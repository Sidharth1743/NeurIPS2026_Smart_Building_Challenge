/* Canonical layout + on-device base controller (pid_water) + residual action decode.
 *
 * Included by smart_buildings.cu.  Everything here is OFF unless the env is created with env.canonical = 1
 * (and env.residual = 1 for the residual mode); the default path is unchanged.
 *
 * CANONICAL LAYOUT.  A building is described by slots that mean the same thing in every building:
 *   AHU slot k (k < CN_K) = the k-th AHU in numeric id order (ahu_1, ahu_2, ...);
 *   zone slot s = CN_ZPA * k + j = the j-th zone served by AHU k, zones in the order their
 *   VAV_<n>_zone_air_temperature_sensor field first appears in the observation;
 *   one hot-water slot; one global slot.
 * The observation handed to the policy is a fixed-size vector of scaled features per slot (CN_OBS values, zeros
 * for absent slots, an "exists" flag first) computed from the ORACLE observation vector (+ the base controller's
 * proposal), so the submission can compute it from what the gateway hands it.  The policy's continuous heads
 * are per slot too (CN_ACT): zone s -> damper, reheat; AHU k -> supply-air temperature, static pressure,
 * recirculation, run; hot water -> supply temperature, pump dP, run.  No discrete heads are used (run decisions are
 * continuous heads thresholded at 0, so they can be residuals of the base's on/off decision).
 *
 * BASE CONTROLLER.  A device port of starter-kit baselines/pid_plus.PIDWaterResetPolicy (= the `pid_water`
 * baseline), arithmetic in the same precisions (numpy 2 / NEP 50 semantics: zone temperatures, their means and the
 * zone-level commands in float32, the AHU PID loop, economiser and hot-water reset in float64, round-to-nearest per
 * operation, numpy's pairwise sum for means).  One documented difference: zones inside an AHU are taken in
 * observation order (pid_water uses the metadata's room order); that only changes the float32 rounding of an AHU's
 * mean temperature.  Verified against a numpy port that uses the same order (gate `basecheck`).
 *
 * RESIDUAL.  continuous: a = clip(base + res_scale * clip(p, -1, 1), -1, 1);  run bits: on iff
 * (2 base - 1) + res_dscale * clip(p, -1, 1) > 0.  With the policy's action means initialised at 0 the agent starts
 * as the base controller.  Comfort guard (guard_k): p is first multiplied by the action's trust.  Absolute canonical mode (residual = 0): a = clip(p, -1, 1), run iff p > -0.5, so the neutral output (p = 0, the zero_init_actions start) is a running device and p < -0.5 is the policy deliberately stopping it (a threshold at 0 makes a zero-mean policy flip every air handler and pump on every resample: measured r8 failure mode).
 */
#define CN_K 9
#define CN_ZPA 10
#define CN_ZS (CN_K * CN_ZPA)
#define CN_FZ 9
#define CN_FA 16
#define CN_FH 10
#define CN_FG 19
#define CN_OA (CN_ZS * CN_FZ)
#define CN_OH (CN_OA + CN_K * CN_FA)
#define CN_OG (CN_OH + CN_FH)
#define CN_OBS (CN_OG + CN_FG)
#define CN_AA (2 * CN_ZS)
#define CN_AH (CN_AA + 4 * CN_K)
#define CN_ACT (CN_AH + 3)

/* per group, read-only, device */
struct CanonMap {
    int nz_total;           /* zones in the building (all AHUs) */
    int K;                  /* AHUs */
    /* zone slots (CN_ZS): observation indices and native action indices, -1 = absent */
    int zT[CN_ZS], zH[CN_ZS], zC[CN_ZS], zD[CN_ZS], zR[CN_ZS], zAd[CN_ZS], zAr[CN_ZS];
    int zc[CN_K];           /* zones of AHU k (slots CN_ZPA*k .. +zc[k]-1) */
    int aOAT[CN_K], aSP[CN_K], aSF[CN_K], aOF[CN_K], aSAT[CN_K], aRC[CN_K], aCR[CN_K], aRUNo[CN_K];
    int aT[CN_K], aP[CN_K], aRec[CN_K], aRun[CN_K];      /* native action indices */
    int hWT, hWS, hDP, hDS, hRS, hHR, hW, hDPa, hRun;
    int g_hs[4], g_hc[4], g_ds[4], g_dc[4], g_cn, g_cs, g_oc;
    float tmean, tstd;                                      /* zone temperature normalisation (oracle obs) */
    double comfort_lo, comfort_hi, eco_lo, eco_hi;
    int morning, evening;
    int zV[CN_ZS];          /* slot -> zone index, -1 absent */
    int n_holiday;
    int holiday[32];
    int start_doy;          /* local day-of-year at step 0, 1-based */
    /* field native-value reconstruction for the features: v = obs * fstd + fmean (fstd = 1, fmean = 0 if raw) */
    float fmean[SB_OBS], fstd[SB_OBS];
    int slot_of_action[SB_NATIVE_MAX];                      /* native action -> canonical action slot */
};

/* numpy pairwise sum (float32 / float64), n < 128 (the zone counts here); then / n as numpy's mean */
__device__ __forceinline__ float np_mean_f32(const float* a, int n) {
    float r;
    if (n < 8) {
        r = 0.0f;
        for (int i = 0; i < n; i++) r = __fadd_rn(r, a[i]);
    } else {
        float q[8];
        for (int j = 0; j < 8; j++) q[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) q[j] = __fadd_rn(q[j], a[i + j]);
        r = __fadd_rn(__fadd_rn(__fadd_rn(q[0], q[1]), __fadd_rn(q[2], q[3])),
                      __fadd_rn(__fadd_rn(q[4], q[5]), __fadd_rn(q[6], q[7])));
        for (; i < n; i++) r = __fadd_rn(r, a[i]);
    }
    return __fdiv_rn(r, (float)n);
}
__device__ __forceinline__ double np_mean_f64(const double* a, int n) {
    double r;
    if (n < 8) {
        r = 0.0;
        for (int i = 0; i < n; i++) r = __dadd_rn(r, a[i]);
    } else {
        double q[8];
        for (int j = 0; j < 8; j++) q[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) q[j] = __dadd_rn(q[j], a[i + j]);
        r = __dadd_rn(__dadd_rn(__dadd_rn(q[0], q[1]), __dadd_rn(q[2], q[3])),
                      __dadd_rn(__dadd_rn(q[4], q[5]), __dadd_rn(q[6], q[7])));
        for (; i < n; i++) r = __dadd_rn(r, a[i]);
    }
    return __ddiv_rn(r, (double)n);
}
/* pid._cyclic_feature_to_index with numpy float32 arctan2 (correctly rounded through double) */
__device__ __forceinline__ int cn_cyclic(float s, float c, int period) {
    float ang = (float)atan2((double)s, (double)c);
    if (ang < 0.0f) ang = __fadd_rn(ang, (float)(2.0 * 3.141592653589793));
    float v = __fdiv_rn(__fmul_rn((float)period, ang), (float)(2.0 * 3.141592653589793));
    return (int)floorf(v);
}
__device__ __forceinline__ double cn_clip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* One thread per env.  obs: oracle observation rows [nenv][SB_OBS]; pid state [nenv][CN_K][2] (integral, prev);
 * out: base normalized action per native index [nenv][stride] (continuous in [-1,1]; discrete 0/1), and the per-AHU
 * effort / mean zone temperature for the features [nenv][CN_K][2]. */
__global__ void base_pidw_k(const CanonMap* __restrict__ cm, const float* __restrict__ obs, int nenv, int n_action,
                            double* pid, float* base, float* ahu_info) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    const float* o = obs + (size_t)e * SB_OBS;
    float* b = base + (size_t)e * n_action;
    double* ps = pid + (size_t)e * CN_K * 2;
    float* ai = ahu_info + (size_t)e * CN_K * 2;
    for (int i = 0; i < n_action; i++) b[i] = 0.0f;
    const int hour = cn_cyclic(o[cm->g_hs[0]], o[cm->g_hc[0]], 24);
    const int day = cn_cyclic(o[cm->g_ds[0]], o[cm->g_dc[0]], 7);
    const bool work = day < 5 && cm->morning <= hour && hour < cm->evening;
    const double lo = work ? cm->comfort_lo : cm->eco_lo, hi = work ? cm->comfort_hi : cm->eco_hi;
    const float lof = (float)lo, hif = (float)hi;
    for (int k = 0; k < cm->K; k++) if (cm->aRec[k] >= 0) b[cm->aRec[k]] = 1.0f;
    bool any_heating = false, any_active = false;
    double efforts[CN_K];
    int ne = 0;
    double short_[CN_ZS];
    int nsh = 0;
    for (int k = 0; k < cm->K; k++) {
        ai[2 * k] = 0.0f; ai[2 * k + 1] = 0.0f;
        const int nz = cm->zc[k];
        if (nz == 0) continue;
        float T[CN_ZPA];
        for (int j = 0; j < nz; j++) {
            const int s = CN_ZPA * k + j;
            T[j] = __fadd_rn(__fmul_rn(o[cm->zT[s]], cm->tstd), cm->tmean);
            /* zone commands (PIDProportionalReheatPolicy._command_zones), float32 */
            float err = 0.0f;
            if (T[j] < lof) err = __fsub_rn(lof, T[j]);
            else if (T[j] > hif) err = __fsub_rn(hif, T[j]);
            float damper = fminf(1.0f, __fdiv_rn(fabsf(err), 2.0f));
            if (cm->zAd[s] >= 0) b[cm->zAd[s]] = __fsub_rn(__fmul_rn(2.0f, damper), 1.0f);
            float opening = fminf(1.0f, __fdiv_rn(fmaxf(0.0f, err), 2.0f));
            if (cm->zAr[s] >= 0) b[cm->zAr[s]] = __fsub_rn(__fmul_rn(2.0f, opening), 1.0f);
        }
        const double avg = (double)np_mean_f32(T, nz);
        ai[2 * k + 1] = (float)avg;
        double error = 0.0;
        if (avg < lo) error = __dsub_rn(lo, avg);
        else if (avg > hi) error = __dsub_rn(hi, avg);
        double* st = ps + 2 * k;
        if (error == 0.0) {
            st[0] = __dmul_rn(st[0], 0.5);
            st[1] = 0.0;
            if (cm->aT[k] >= 0) b[cm->aT[k]] = -1.0f;
            if (cm->aP[k] >= 0) b[cm->aP[k]] = -1.0f;
            if (cm->aRec[k] >= 0) b[cm->aRec[k]] = 1.0f;
            if (cm->aRun[k] >= 0) b[cm->aRun[k]] = 0.0f;
            continue;
        }
        st[0] = cn_clip(__dadd_rn(st[0], error), -10.0, 10.0);
        const double deriv = __dsub_rn(error, st[1]);
        st[1] = error;
        const double u = cn_clip(__dadd_rn(__dadd_rn(__dmul_rn(0.5, error), __dmul_rn(0.02, st[0])), __dmul_rn(0.1, deriv)), -1.0, 1.0);
        ai[2 * k] = (float)u;
        efforts[ne++] = u;
        const double pressure = cn_clip(__dsub_rn(__dmul_rn(2.0, fabs(u)), 1.0), -1.0, 1.0);
        if (cm->aT[k] >= 0) b[cm->aT[k]] = (float)u;
        if (cm->aP[k] >= 0) b[cm->aP[k]] = (float)pressure;
        if (cm->aRun[k] >= 0) b[cm->aRun[k]] = 1.0f;
        if (cm->aRec[k] >= 0) {     /* PIDEconomizerPolicy._recirculation */
            double rec = 1.0;
            if (!(u >= 0.0) && cm->aOAT[k] >= 0) {
                const double outside = (double)o[cm->aOAT[k]], inside = avg;
                if (outside < inside) {
                    double pa = __dmul_rn(__dmul_rn(__dadd_rn(pressure, 1.0), 0.5), 2000.0);
                    pa = fmax(pa, 200.0);
                    const double breakeven = __dmul_rn(pa, 0.00322);
                    if (__dsub_rn(inside, outside) > breakeven) {
                        double target = __dadd_rn(285.0, __dmul_rn(__dmul_rn(__dadd_rn(u, 1.0), 0.5), 20.0));
                        target = fmax(target, lo);
                        if (inside > target) {
                            const double f = __ddiv_rn(__dsub_rn(inside, target), __dsub_rn(inside, outside));
                            rec = __dsub_rn(__dmul_rn(2.0, cn_clip(__dsub_rn(1.0, f), 0.0, 1.0)), 1.0);
                        }
                    }
                }
            }
            b[cm->aRec[k]] = (float)rec;
        }
        any_active = true;
        if (error > 0.0) any_heating = true;
    }
    if (cm->hW >= 0) b[cm->hW] = any_heating ? 1.0f : -1.0f;
    if (cm->hDPa >= 0) b[cm->hDPa] = any_active ? 1.0f : -1.0f;
    if (cm->hRun >= 0) b[cm->hRun] = any_active ? 1.0f : 0.0f;
    /* PIDWaterResetPolicy: hot-water reset (integral gain 0) */
    if (cm->hW >= 0 && b[cm->hW] > 0.0f) {
        for (int k = 0; k < cm->K; k++)
            for (int j = 0; j < cm->zc[k]; j++) {
                const int s = CN_ZPA * k + j;
                const float T = __fadd_rn(__fmul_rn(o[cm->zT[s]], cm->tstd), cm->tmean);
                const float d = __fsub_rn(lof, T);
                short_[nsh++] = d > 0.0f ? (double)d : 0.0;
            }
        const double shortfall = nsh ? np_mean_f64(short_, nsh) : 0.0;
        const double demand = __ddiv_rn(shortfall, 4.0);
        double sat = 0.0;
        if (ne > 0) {
            double h[CN_K];
            for (int i = 0; i < ne; i++) h[i] = fmax(0.0, efforts[i]);
            for (int i = 1; i < ne; i++) {            /* insertion sort (<= 9) */
                double x = h[i]; int j = i - 1;
                while (j >= 0 && h[j] > x) { h[j + 1] = h[j]; j--; }
                h[j + 1] = x;
            }
            /* numpy quantile 'linear', q = 0.25: virtual index (n q + (1 - q)) - 1, numpy _lerp */
            const double vi = __dsub_rn(__dadd_rn(__dmul_rn((double)ne, 0.25), 0.75), 1.0);
            const int i0 = (int)floor(vi);
            const double t = __dsub_rn(vi, (double)i0);
            const double a0 = h[i0], a1 = h[i0 + 1 < ne ? i0 + 1 : ne - 1];
            const double diff = __dsub_rn(a1, a0);
            double q = (t >= 0.5) ? __dsub_rn(a1, __dmul_rn(diff, __dsub_rn(1.0, t))) : __dadd_rn(a0, __dmul_rn(diff, t));
            if (a0 == a1) q = a0;
            sat = q;
        }
        const double frac = cn_clip(fmax(demand, sat) + 0.0, 0.0, 1.0);
        b[cm->hW] = (float)__dsub_rn(__dmul_rn(2.0, frac), 1.0);
    }
}

/* COMFORT GUARD (env.guard_d1 > 0): the residual's authority per native action, trust in [0, 1], falls from 1 to 0 as
 * the zone's distance below the base's band (guard_hot = 1: below or above) grows from guard_d0 to guard_d0 + guard_d1 K; an AHU's
 * actions use the worst of its zones, the hot-water actions the worst zone of the building.  Far outside the band the
 * action is the base controller's own; the residual cannot hold a building in a state the base would leave. */
__global__ void guard_k(const CanonMap* __restrict__ cm, const float* __restrict__ obs, int nenv, int n_action,
                        float d0, float d1, int hot, float* trust) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    const float* o = obs + (size_t)e * SB_OBS;
    float* w = trust + (size_t)e * n_action;
    for (int i = 0; i < n_action; i++) w[i] = 1.0f;
    const int hour = cn_cyclic(o[cm->g_hs[0]], o[cm->g_hc[0]], 24);
    const int day = cn_cyclic(o[cm->g_ds[0]], o[cm->g_dc[0]], 7);
    const bool work = day < 5 && cm->morning <= hour && hour < cm->evening;
    const float lof = (float)(work ? cm->comfort_lo : cm->eco_lo), hif = (float)(work ? cm->comfort_hi : cm->eco_hi);
    float wb = 1.0f;
    for (int k = 0; k < cm->K; k++) {
        float wk = 1.0f;
        for (int j = 0; j < cm->zc[k]; j++) {
            const int s = CN_ZPA * k + j;
            const float T = __fadd_rn(__fmul_rn(o[cm->zT[s]], cm->tstd), cm->tmean);
            const float dev = T < lof ? lof - T : (hot && T > hif ? T - hif : 0.0f);
            const float t = fminf(1.0f, fmaxf(0.0f, 1.0f - (dev - d0) / d1));
            if (cm->zAd[s] >= 0) w[cm->zAd[s]] = t;
            if (cm->zAr[s] >= 0) w[cm->zAr[s]] = t;
            wk = fminf(wk, t);
        }
        if (cm->aT[k] >= 0) w[cm->aT[k]] = wk;
        if (cm->aP[k] >= 0) w[cm->aP[k]] = wk;
        if (cm->aRec[k] >= 0) w[cm->aRec[k]] = wk;
        if (cm->aRun[k] >= 0) w[cm->aRun[k]] = wk;
        wb = fminf(wb, wk);
    }
    if (cm->hW >= 0) w[cm->hW] = wb;
    if (cm->hDPa >= 0) w[cm->hDPa] = wb;
    if (cm->hRun >= 0) w[cm->hRun] = wb;
}

/* PID state of restarted envs back to 0 (the policy object is rebuilt per scenario/episode) */
__global__ void base_reset_k(double* pid, int nenv, const int* gate) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0)) return;
    for (int i = 0; i < 2 * CN_K; i++) pid[(size_t)e * 2 * CN_K + i] = 0.0;
}

__device__ __forceinline__ float cn_nat(const CanonMap* cm, const float* o, int idx) {
    return __fadd_rn(__fmul_rn(o[idx], cm->fstd[idx]), cm->fmean[idx]);
}
/* The policy's observation: CN_OBS scaled features (zeros for absent slots), out stride SB_OBS. */
__global__ void canon_obs_k(const CanonMap* __restrict__ cm, const float* __restrict__ obs, const float* __restrict__ base,
                            const float* __restrict__ ahu_info, const double* __restrict__ pid, int nenv, int n_action,
                            float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * CN_OBS) return;
    const int e = idx / CN_OBS, f = idx % CN_OBS;
    const float* o = obs + (size_t)e * SB_OBS;
    const float* b = base + (size_t)e * n_action;
    float v = 0.0f;
    if (f < CN_OA) {
        const int s = f / CN_FZ, c = f % CN_FZ;
        if (cm->zT[s] >= 0) {
            const float T = cn_nat(cm, o, cm->zT[s]);
            const float Hs = cn_nat(cm, o, cm->zH[s]), Cs = cn_nat(cm, o, cm->zC[s]);
            switch (c) {
                case 0: v = 1.0f; break;
                case 1: v = (T - 295.0f) / 5.0f; break;
                case 2: v = (T - Hs) / 2.0f; break;
                case 3: v = (Cs - T) / 2.0f; break;
                case 4: v = (Hs - 294.0f) / 3.0f; break;
                case 5: v = cm->zD[s] >= 0 ? 2.0f * cn_nat(cm, o, cm->zD[s]) - 1.0f : 0.0f; break;
                case 6: v = cm->zR[s] >= 0 ? 2.0f * cn_nat(cm, o, cm->zR[s]) - 1.0f : 0.0f; break;
                case 7: v = cm->zAd[s] >= 0 ? b[cm->zAd[s]] : 0.0f; break;
                case 8: v = cm->zAr[s] >= 0 ? b[cm->zAr[s]] : 0.0f; break;
            }
        }
    } else if (f < CN_OH) {
        const int k = (f - CN_OA) / CN_FA, c = (f - CN_OA) % CN_FA;
        if (k < cm->K) {
            const float* aif = ahu_info + (size_t)e * CN_K * 2;
            const double* ps = pid + (size_t)e * CN_K * 2;
            auto g = [&](int i) { return i >= 0 ? cn_nat(cm, o, i) : 0.0f; };
            switch (c) {
                case 0: v = 1.0f; break;
                case 1: v = (g(cm->aOAT[k]) - 283.0f) / 15.0f; break;
                case 2: v = g(cm->aSP[k]) / 1000.0f; break;
                case 3: v = g(cm->aSF[k]) / 10.0f; break;
                case 4: v = g(cm->aOF[k]) / 10.0f; break;
                case 5: v = (cm->aSAT[k] >= 0 ? g(cm->aSAT[k]) - 295.0f : 0.0f) / 5.0f; break;
                case 6: v = 2.0f * g(cm->aRC[k]) - 1.0f; break;
                case 7: v = g(cm->aCR[k]) / 10.0f; break;
                case 8: v = 2.0f * g(cm->aRUNo[k]) - 1.0f; break;
                case 9: v = cm->aT[k] >= 0 ? b[cm->aT[k]] : 0.0f; break;
                case 10: v = cm->aP[k] >= 0 ? b[cm->aP[k]] : 0.0f; break;
                case 11: v = cm->aRec[k] >= 0 ? b[cm->aRec[k]] : 0.0f; break;
                case 12: v = cm->aRun[k] >= 0 ? 2.0f * b[cm->aRun[k]] - 1.0f : 0.0f; break;
                case 13: v = cm->zc[k] > 0 ? (aif[2 * k + 1] - 295.0f) / 5.0f : 0.0f; break;
                case 14: v = (float)(ps[2 * k] / 10.0); break;
                case 15: v = aif[2 * k]; break;
            }
        }
    } else if (f < CN_OG) {
        const int c = f - CN_OH;
        auto g = [&](int i) { return i >= 0 ? cn_nat(cm, o, i) : 0.0f; };
        switch (c) {
            case 0: v = 1.0f; break;
            case 1: v = (g(cm->hWT) - 330.0f) / 20.0f; break;
            case 2: v = (g(cm->hWS) - 330.0f) / 20.0f; break;
            case 3: v = g(cm->hDP) / 10.0f; break;
            case 4: v = g(cm->hDS) / 10.0f; break;
            case 5: v = 2.0f * g(cm->hRS) - 1.0f; break;
            case 6: v = g(cm->hHR) / 10.0f; break;
            case 7: v = cm->hW >= 0 ? b[cm->hW] : 0.0f; break;
            case 8: v = cm->hDPa >= 0 ? b[cm->hDPa] : 0.0f; break;
            case 9: v = cm->hRun >= 0 ? 2.0f * b[cm->hRun] - 1.0f : 0.0f; break;
        }
    } else {
        const int c = f - CN_OG;
        int i = -1;
        if (c < 4) i = cm->g_hs[c];
        else if (c < 8) i = cm->g_hc[c - 4];
        else if (c < 12) i = cm->g_ds[c - 8];
        else if (c < 16) i = cm->g_dc[c - 12];
        else if (c == 16) i = cm->g_cn;
        else if (c == 17) i = cm->g_cs;
        else i = cm->g_oc;
        v = i >= 0 ? o[i] : 0.0f;
    }
    out[(size_t)e * SB_OBS + f] = v;
}

/* Canonical / residual action decode: the framework's continuous heads (canonical slots) + the base proposal ->
 * native setpoints, with the oracle's arithmetic (float32 ((m - lo) / in) * out + mn, op by op; discrete
 * 1 -> +1 / 0 -> -1 then the double affine map). */
__global__ void decode_res_k(const CanonMap* __restrict__ cm, const float* __restrict__ base, const float* __restrict__ actions,
                             int astride, int n_action, int nenv, const uint8_t* is_disc, const double* mn,
                             const double* mx, const double* mnr, const double* mxr, int resid, float scale,
                             float dscale, const float* __restrict__ trust, double* native) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * n_action) return;
    const int e = idx / n_action, i = idx % n_action;
    const int sl = cm->slot_of_action[i];
    float p = 0.0f;
    if (actions != NULL && sl >= 0) {
        p = actions[(size_t)e * astride + sl];
        p = isfinite(p) ? fminf(1.0f, fmaxf(-1.0f, p)) : 0.0f;
        if (trust != NULL) p = __fmul_rn(p, trust[(size_t)e * n_action + i]);
    }
    const float bse = base[(size_t)e * n_action + i];
    double out;
    if (is_disc[i]) {
        /* Absolute canonical mode: the run bit is a thresholded continuous head, and the neutral
         * output (p = 0) is ON - a building's equipment runs by default and the policy has to learn
         * to stop it.  p < -0.5 is the deliberate OFF command, so the decision is stable under the
         * exploration noise (sigma 0.2: P(flip from the neutral start) ~ 0.6%).  Threshold 0 made a
         * zero-mean policy flip every air handler and pump 50/50 every resample. */
        const bool on = resid ? (__fadd_rn(__fsub_rn(__fmul_rn(2.0f, bse), 1.0f), __fmul_rn(dscale, p)) > 0.0f) : (p > -0.5f);
        const double merged = on ? 1.0 : -1.0;
        out = (merged - mnr[i]) / (mxr[i] - mnr[i]) * (mx[i] - mn[i]) + mn[i];
    } else {
        float m = resid ? __fadd_rn(bse, __fmul_rn(scale, p)) : p;
        m = fminf(1.0f, fmaxf(-1.0f, m));
        const float lo = (float)mnr[i], rin = (float)(mxr[i] - mnr[i]), rout = (float)(mx[i] - mn[i]), m0 = (float)mn[i];
        out = (double)__fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(m, lo), rin), rout), m0);
    }
    native[(size_t)e * n_action + i] = out;
}

#ifdef SB_ENTITY
/* Entity mode packs the canonical features into one row per zone slot: the 9 zone features, then the 19 global
 * features (time / occupancy).  AHU and hot-water residuals stay 0, so those actions remain the base controller's. */
__global__ void entity_pack_k(const float* __restrict__ canon, float* __restrict__ ent, int nenv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * CN_ZS) return;
    const int e = idx / CN_ZS, s = idx % CN_ZS;
    const float* src = canon + (size_t)e * SB_OBS;
    float* dst = ent + (size_t)idx * (CN_FZ + CN_FG);
    for (int c = 0; c < CN_FZ; c++) dst[c] = src[s * CN_FZ + c];
    for (int c = 0; c < CN_FG; c++) dst[CN_FZ + c] = src[CN_OG + c];
}
/* Zone-agent actions (2 per slot) -> the canonical action vector decode_res_k reads.  AHU / hot-water slots stay 0. */
__global__ void entity_gather_k(const float* __restrict__ ent, float* __restrict__ canon, int nenv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    float* dst = canon + (size_t)e * CN_ACT;
    for (int i = 0; i < CN_ACT; i++) dst[i] = 0.0f;
    const float* src = ent + (size_t)e * CN_ZS * 2;
    for (int s = 0; s < CN_ZS; s++) {
        dst[2 * s] = src[2 * s];
        dst[2 * s + 1] = src[2 * s + 1];
    }
}
/* Building reward, split evenly over the zones that exist, so the shared policy's gradient has the same scale as
 * one agent per building.  Empty slots get reward 0; every slot shares the building's terminal. */
__global__ void entity_scatter_k(const CanonMap* __restrict__ cm, const float* __restrict__ rew,
                                 const float* __restrict__ term, float* __restrict__ rew_out,
                                 float* __restrict__ term_out, int nenv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    int np = 0;
    for (int s = 0; s < CN_ZS; s++) if (cm->zT[s] >= 0) np++;
    const float share = np > 0 ? rew[e] / (float)np : 0.0f;
    for (int s = 0; s < CN_ZS; s++) {
        rew_out[(size_t)e * CN_ZS + s] = cm->zT[s] >= 0 ? share : 0.0f;
        term_out[(size_t)e * CN_ZS + s] = term[e];
    }
}
__global__ void entity_mask_k(const CanonMap* __restrict__ cm, unsigned char* __restrict__ mask, int nenv, int width) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    for (int s = 0; s < CN_ZS; s++) {
        unsigned char* m = mask + ((size_t)e * CN_ZS + s) * width;
        m[0] = cm->zAd[s] >= 0 ? 1 : 0;
        if (width > 1) m[1] = cm->zAr[s] >= 0 ? 1 : 0;
    }
}
#endif

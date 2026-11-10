/* Anticipation and per-zone memory, computed from the observation the gateway
 * already produces. Off unless env.zone_memory = 1. The actor reads the buffer;
 * the physics and the observation vector are unchanged. */
#pragma once
#include "sb_zone_reward.cuh"

/* Nine anticipation features, plus where the zone will be when the delayed
 * productivity is billed, and whether the air path is already the backup heater. */
constexpr int ZM_FEAT = 11;

#ifdef SB_SHARED_POLICY
extern "C" void sb_set_feat(float* p);
#endif

static int g_zone_memory = 0;
// env.zone_signed (r99): the memory's deficit features carry their sign (cold positive, hot negative)
// instead of max(0, |cold|, |hot|), so a zone that is too HOT cannot excite the cold shortfall that
// opens its reheat valve. The critic's delay summary keeps magnitudes (fabsf), so its meaning is
// unchanged; the actor's f[4], f[5], f[6], f[9] become signed.
static int g_zone_signed = 0;
// env.zone_night_share (r99): at total_occ == 0 the per-zone target is the equal share of the empty
// building's energy and carbon bill instead of 0, so the valves are paid for the gas and fan power they
// cause while nobody is in the building. Sum over present slots stays the building's fuel term.
static int g_zone_night_share = 0;
/* device copies: both flags are read inside kernels (batch-wide, set once at env creation) */
__device__ int d_zone_signed = 0;
__device__ int d_zone_night_share = 0;
static int g_zone_adv = 0;
static int g_zone_critic = 0;
/* env.ahu_critic (zone_critic only): CN_K more returns per step, one per AHU. */
static int g_ahu_critic = 0;
static int zm_rzw = CN_ZS;
static int g_prestart = 0;

static float* zm_feat = nullptr;
static float* zm_aux = nullptr;
static float* zm_critic = nullptr;
static float* zm_ringT = nullptr;
static float* zm_ringD = nullptr;
static int* zm_count = nullptr;
static float* zm_oat = nullptr;
static int* zm_fresh = nullptr;
static unsigned long long* zm_rng = nullptr;
static float* zm_traj = nullptr;
static float* zm_feat_tr = nullptr;
static float* zm_aux_tr = nullptr;
static float* zm_critic_tr = nullptr;
static float* zm_feat_bt = nullptr;
static float* zm_aux_bt = nullptr;
static float* zm_critic_bt = nullptr;
static float* zm_parts_bt = nullptr;
static float* zm_zshare = nullptr;
static float* zm_zshare_tr = nullptr;
static float* zm_zshare_bt = nullptr;
static float* zm_rz = nullptr;
static float* zm_rz_tr = nullptr;
static float* zm_rz_bt = nullptr;
static int zm_cap = 0;
static int zm_total = 0;   /* envs in the whole batch; the live buffers are indexed by that */
static int zm_base = 0;    /* this group's first env; set before each group's launches */
static int zm_traj_h = 0;
static int zm_trace_on = 0;
static FILE* zm_traj_f = nullptr;

__device__ int zm_holiday(const CanonMap* cm, int doy) {
    for (int i = 0; i < cm->n_holiday; i++) if (cm->holiday[i] == doy) return 1;
    return 0;
}
__device__ float zm_nat(const CanonMap* cm, const float* o, int i) {
    if (i < 0) return 0.0f;
    return o[i] * cm->fstd[i] + cm->fmean[i];
}

__global__ void zm_reset_k(int* count, int* fresh, int nenv, int zones, const int* gate) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0)) return;
    fresh[e] = 1;
    for (int z = 0; z < zones; z++) count[(size_t)e * zones + z] = 0;
}

/* One block per env, one thread per zone slot. Thread 0 adds the building sums in slot order. */
constexpr int ZM_THREADS = (CN_ZS + 31) / 32 * 32;
__global__ void __launch_bounds__(ZM_THREADS)
zm_update_k(const CanonMap* cm, const float* obs, const int* stepv, const double* occ,
                            float* feat, float* ringT, float* ringD, int* count, float* last_oat, int* fresh,
                            float* aux, float* critic, int nenv, int n_zones, int obs_stride) {
    const int e = blockIdx.x, z = threadIdx.x;
    if (e >= nenv || cm == nullptr) return;
    __shared__ float s_bounded[CN_ZS];
    __shared__ int s_present[CN_ZS];
    __shared__ float s_last_oat;
    __shared__ int s_fresh;
    const float* o = obs + (size_t)e * obs_stride;
    float hs = zm_nat(cm, o, cm->g_hs[0]), hc = zm_nat(cm, o, cm->g_hc[0]);
    float ds = zm_nat(cm, o, cm->g_ds[0]), dc = zm_nat(cm, o, cm->g_dc[0]);
    float ang = atan2f(hs, hc); if (ang < 0.0f) ang += 6.28318530718f;
    float hour = ang / 6.28318530718f * 24.0f;
    float dang = atan2f(ds, dc); if (dang < 0.0f) dang += 6.28318530718f;
    int weekday = (int)lrintf(dang / 6.28318530718f * 7.0f) % 7;
    if (weekday < 0) weekday += 7;
    int minute = (int)lrintf(hour * 60.0f) % 1440;
    int step = stepv[e];
    int doy = cm->start_doy + (cm->morning * 60 + step * 5) / 1440;
    int start = cm->morning * 60, end = cm->evening * 60;
    int to_start, to_end;
    int today_work = (weekday % 7) < 5 && !zm_holiday(cm, doy);
    if (today_work && minute >= start && minute < end) {
        to_start = 0; to_end = end - minute;
    } else if (today_work && minute < start) {
        to_start = start - minute; to_end = end - minute;
    } else {
        int ahead = 1;
        while (ahead <= 7) {
            int ok = ((weekday + ahead) % 7) < 5 && !zm_holiday(cm, doy + ahead);
            if (ok) break;
            ahead++;
        }
        to_start = ahead * 1440 - minute + start;
        to_end = to_start + (end - start);
    }
    int tom_ok = ((weekday + 1) % 7) < 5 && !zm_holiday(cm, doy + 1);
    float tomorrow = tom_ok ? 1.0f : 0.0f;
    float oat = zm_nat(cm, o, cm->aOAT[0]);
    if (z == 0) { s_fresh = fresh[e]; s_last_oat = last_oat[e]; }
    __syncthreads();
    float oat_delta = s_fresh ? 0.0f : oat - s_last_oat;
    float lo = (float)cm->comfort_lo, hi = (float)cm->comfort_hi;
    int during = today_work && minute >= start && minute < end;
    float ramp = during ? 1.0f : fminf(1.0f, fmaxf(0.0f, (180.0f - (float)to_start) / 180.0f));
    if (z < CN_ZS) do {
        float* f = feat + ((size_t)e * CN_ZS + z) * ZM_FEAT;
        float* cr = critic ? critic + ((size_t)e * CN_ZS + z) * 2 : nullptr;
        s_present[z] = 0;
        if (cm->zT[z] < 0) {
            for (int k = 0; k < ZM_FEAT; k++) f[k] = 0.0f;
            if (cr) { cr[0] = 0.0f; cr[1] = 0.0f; }
            break;
        }
        float temp = zm_nat(cm, o, cm->zT[z]);
        float deficit = fmaxf(0.0f, fmaxf(lo - temp, temp - hi));
        if (d_zone_signed)   /* signed: cold positive, hot negative */
            deficit = (temp < lo) ? (lo - temp) : ((temp > hi) ? -(temp - hi) : 0.0f);
        int nprev = count[(size_t)e * CN_ZS + z];
        int slot = nprev % 48;
        ringT[((size_t)e * CN_ZS + z) * 48 + slot] = temp;
        ringD[((size_t)e * CN_ZS + z) * 48 + slot] = deficit;
        int n = nprev + 1; if (n > 48) n = 48;
        count[(size_t)e * CN_ZS + z] = nprev + 1;
        float d1 = 0, d4 = 0, t4 = 0;
        int n1 = n < 12 ? n : 12;
        for (int k = 0; k < n; k++) {
            int at = (nprev - k) % 48; if (at < 0) at += 48;
            float d = ringD[((size_t)e * CN_ZS + z) * 48 + at];
            float t = ringT[((size_t)e * CN_ZS + z) * 48 + at];
            d4 += d; t4 += t; if (k < n1) d1 += d;
        }
        d1 /= (float)n1; d4 /= (float)n; t4 /= (float)n;
        float hours = (n - 1) * 5.0f / 60.0f;
        int old_at = (nprev - (n - 1)) % 48; if (old_at < 0) old_at += 48;
        float oldest = ringT[((size_t)e * CN_ZS + z) * 48 + old_at];
        float slope = hours > 0.0f ? (temp - oldest) / hours : 0.0f;
        f[0] = to_start / 180.0f; f[1] = to_end / 180.0f; f[2] = tomorrow; f[3] = oat_delta / 2.0f;
        f[4] = deficit / 5.0f; f[5] = d1 / 5.0f; f[6] = d4 / 5.0f; f[7] = slope / 5.0f; f[8] = (t4 - 295.0f) / 5.0f;
        float horizon = during ? 60.0f : fminf((float)to_start, 240.0f);
        float that = temp + slope * (horizon / 60.0f);
        float dhat = fmaxf(0.0f, fmaxf(lo - that, that - hi));
        if (d_zone_signed) dhat = (that < lo) ? (lo - that) : ((that > hi) ? -(that - hi) : 0.0f);
        f[9] = fminf(dhat / 5.0f, 10.0f);
        float damper = 0.0f, reheat = 1.0f;
        int has_d = cm->zD[z] >= 0, has_r = cm->zR[z] >= 0, has_h = cm->zH[z] >= 0;
        if (has_d) damper = fminf(1.0f, fmaxf(0.0f, zm_nat(cm, o, cm->zD[z])));
        if (has_r) reheat = fminf(1.0f, fmaxf(0.0f, zm_nat(cm, o, cm->zR[z])));
        int kahu = z / CN_ZPA;
        int has_sat = kahu < CN_K && cm->aSAT[kahu] >= 0;
        float sat = has_sat ? zm_nat(cm, o, cm->aSAT[kahu]) : temp;
        int idle = 0;
        if (has_r && has_sat) idle = reheat < 0.3f && sat <= temp;
        else if (has_r) idle = reheat < 0.3f;
        else if (has_sat) idle = sat <= temp;
        float heat_sp = has_h ? zm_nat(cm, o, cm->zH[z]) : lo;
        float below = fmaxf(0.0f, heat_sp - temp) / 5.0f;
        f[10] = fminf(damper * below * (idle ? 1.0f : 0.0f), 10.0f);
        float adef = fabsf(deficit);   /* the critic / comfort shortfall stays a magnitude */
        float over = fmaxf(0.0f, adef - 5.0f) / 5.0f;
        float shortfall = adef <= 5.0f ? adef / 5.0f : 1.0f + over * over;
        /* The actor already sees the raw deficit. The critic and the comfort-start target stay
         * inside the linear part of the competition curve, so one frozen zone cannot put tens
         * of units into the value trunk. */
        float bounded = fminf(shortfall, 1.0f);
        int v = cm->zV[z];
        float oc = (occ != nullptr && v >= 0 && v < n_zones) ? (float)occ[(size_t)e * n_zones + v] : 0.0f;
        if (cr) { cr[0] = oc; cr[1] = bounded; }
        s_bounded[z] = bounded; s_present[z] = 1;
    } while (0);
    __syncthreads();
    if (z != 0) return;
    last_oat[e] = oat;
    fresh[e] = 0;
    float phi_sum = 0.0f, sh_sum = 0.0f; int np = 0;
    for (int s = 0; s < CN_ZS; s++) {
        if (!s_present[s]) continue;
        const float bounded = s_bounded[s];
        phi_sum += ramp * bounded; sh_sum += bounded; np++;
    }
    if (aux != nullptr) {
        float inv = np > 0 ? 1.0f / (float)np : 0.0f;
        aux[(size_t)e * 4] = sh_sum * inv;
        aux[(size_t)e * 4 + 1] = (float)to_start / 180.0f;
        aux[(size_t)e * 4 + 2] = during ? 1.0f : 0.0f;
        aux[(size_t)e * 4 + 3] = phi_sum * inv;
    }
}

__global__ void zm_prestart_temp_k(GpuSim* sims, int nenv, int ncv, int nmass, const int* gate, unsigned long long* rng) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0)) return;
    unsigned long long r = rng[e] ? rng[e] : (0x9E3779B97F4A7C15ULL * (unsigned long long)(e + 1));
    GpuSim* s = &sims[e];
    int ncell = ncv > nmass ? ncv : nmass;
    for (int p = 0; p < ncell; p++) {
        r = r * 6364136223846793005ULL + 1ULL;
        float u = (float)((r >> 40) & 0xFFFFFF) / 16777216.0f;
        double shift = (double)(u * 12.0f - 10.0f);
        if (p < ncv) s->temp[(size_t)p * s->ts] += shift;
        if (p < nmass) s->mass_temp[(size_t)p * s->ts] += shift;
    }
    rng[e] = r;
}

__global__ void zm_prestart_clock_k(int* stepv, int nenv, const int* gate, unsigned long long* rng) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0)) return;
    unsigned long long r = rng[e] * 6364136223846793005ULL + 1ULL;
    int day = 1 + (int)(r % 28ull);
    r = r * 6364136223846793005ULL + 1ULL;
    int slot = (int)(r % 48ull);
    stepv[e] = day * 288 - 48 + slot;
    rng[e] = r;
}

__global__ void zm_traj_k(const double* rparts, float* dst, int t, int nenv) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    const double* rp = rparts + (size_t)e * SB_NPARTS;
    float* o = dst + ((size_t)t * nenv + e) * 5;
    o[0] = (float)rp[0]; o[1] = (float)rp[1]; o[2] = (float)rp[2]; o[3] = (float)rp[3]; o[4] = (float)rp[4];
}

__global__ void zm_copy_k(const float* src, float* dst, int t, int nenv, int width) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv || src == nullptr || dst == nullptr) return;
    const float* in = src + (size_t)e * width;
    float* out = dst + ((size_t)t * nenv + e) * width;
    for (int i = 0; i < width; i++) out[i] = in[i];
}
__global__ void zm_transpose_k(const float* src, float* dst, int T, int E, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)T * E * C;
    if (i >= n) return;
    int c = (int)(i % (size_t)C);
    size_t te = i / (size_t)C;
    int e = (int)(te % (size_t)E);
    int t = (int)(te / (size_t)E);
    dst[((size_t)e * T + t) * C + c] = src[i];
}

/* Per-zone productivity share minus the building's productivity, in the same
 * units as the trainer reward. Summed over present slots it is zero, so the
 * plant's full return is unchanged. Absent slots stay zero. */
__global__ void zm_zshare_k(const CanonMap* cm, GpuSim* sims, const double* occV,
                            const int* stepv, float* dst, int nenv, int nz, float rew_scale) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    float* o = dst + (size_t)e * CN_ZS;
    for (int s = 0; s < CN_ZS; s++) o[s] = 0.f;
    if (cm == nullptr || sims == nullptr || occV == nullptr) return;
    GpuSim* sim = &sims[e];
    int kv = stepv[e];
    int zones = sim->n_zones;
    if (zones > nz) zones = nz;
    const double* occ = occV + (size_t)e * nz;
    double dt = sim->dt_sec;
    double inv_hour = 1.0 / 3600.0;
    double max_pph = sim->rw_scalars[0], min_pph = sim->rw_scalars[1], ref = sim->rw_scalars[2];
    double hsp = sim->sched_heat_sp[kv], csp = sim->sched_cool_sp[kv];
    double total_occ = 0.0;
    for (int z = 0; z < zones; z++) total_occ += occ[z];
    double max_prod = max_pph * total_occ * dt * inv_hour;
    double min_prod = min_pph * total_occ * dt * inv_hour;
    double den = max_prod - min_prod;
    double npr = sim->rparts[0];
    double pw = sim->rw_scalars[5], ew = sim->rw_scalars[6], cw = sim->rw_scalars[7];
    double wsum = pw + ew + cw;
    double coef = (wsum != 0.0 && total_occ > 0.0 && den != 0.0) ? (double)rew_scale * pw / wsum : 0.0;
    if (coef == 0.0) return;
    for (int s = 0; s < CN_ZS; s++) {
        int z = cm->zV[s];
        if (z < 0 || z >= zones) continue;
        double dev = sim->avg_temp[z] - csp;
        if (hsp - sim->avg_temp[z] > dev) dev = hsp - sim->avg_temp[z];
        if (dev < 0.0) dev = 0.0;
        double short_ = (dev <= ref) ? dev / ref
                        : 1.0 + ((dev - ref) / ref) * ((dev - ref) / ref);
        double occz = occ[z];
        double Sz = max_pph * (1.0 - short_) * occz * dt * inv_hour;
        double npr_z = (Sz - min_prod * occz / total_occ) / den - occz / total_occ;
        o[s] = (float)(coef * (npr_z - npr));
    }
}

static void zm_zone_share() {
    if (!g_zone_adv || zm_zshare == nullptr) return;
    zm_zshare_k<<<(nenv + 127) / 128, 128, 0, stream>>>(
        d_cmap, d_sims, d_V, d_stepv, zm_zshare + (size_t)zm_base * CN_ZS, nenv, n_zones, g_rew_scale);
}

/* r_z for the zone critic. Separate from the zone_adv trace, which stays
 * coef*(npr_z - npr). Present slots of an occupied building sum to the
 * learning reward. An empty building writes zeros.
 * Under ahu_critic, column CN_ZS + k is AHU k's return: the productivity of its own
 * zone slots plus the whole building's energy and carbon, so an empty building
 * gives every present AHU the building reward. Absent AHUs write zeros. */
__global__ void zm_rz_k(const CanonMap* cm, GpuSim* sims, const double* occV,
                        const int* stepv, float* dst, int nenv, int nz, float rew_scale, int width) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    float* o = dst + (size_t)e * width;
    for (int s = 0; s < width; s++) o[s] = 0.f;
    if (cm == nullptr || sims == nullptr || occV == nullptr) return;
    GpuSim* sim = &sims[e];
    int kv = stepv[e];
    int zones = sim->n_zones;
    if (zones > nz) zones = nz;
    const double* occ = occV + (size_t)e * nz;
    double total_occ = 0.0;
    for (int z = 0; z < zones; z++) total_occ += occ[z];
    double pw = sim->rw_scalars[5], ew = sim->rw_scalars[6], cw = sim->rw_scalars[7];
    double nec = sim->rparts[1], ncar = sim->rparts[2];
    const bool ahu = width > CN_ZS;
    double ec = 0.0;
    if (ahu) {
        ec = sb_zone_target(0.0, 1.0, nec, ncar, pw, ew, cw, (double)rew_scale);
        for (int k = 0; k < CN_K && k < cm->K; k++) o[CN_ZS + k] = (float)ec;
    }
    if (!(total_occ > 0.0)) {
        if (d_zone_night_share) {
            /* equal share of the empty building's energy and carbon, so the zone valves are billed for
             * the gas and fan power they cause at night and at weekends (r99). Sums to the fuel term. */
            int present = 0;
            for (int s = 0; s < CN_ZS; ++s) if (cm->zV[s] >= 0) present++;
            if (present > 0) {
                double fuel = sb_zone_target(0.0, 1.0, nec, ncar, pw, ew, cw, (double)rew_scale);
                for (int s = 0; s < CN_ZS; ++s)
                    if (cm->zV[s] >= 0) o[s] = (float)(fuel / (double)present);
            }
        }
        return;
    }
    double hsp = sim->sched_heat_sp[kv], csp = sim->sched_cool_sp[kv];
    double own[CN_K];
    for (int k = 0; k < CN_K; k++) own[k] = 0.0;
    for (int s = 0; s < CN_ZS; s++) {
        int z = cm->zV[s];
        if (z < 0 || z >= zones) continue;
        double share = occ[z] / total_occ;
        double p_z = sb_zone_productivity(sim->avg_temp[z], hsp, csp, sim->rw_scalars[2],
            sim->rw_scalars[0], sim->rw_scalars[1], occ[z], total_occ, sim->dt_sec);
        o[s] = (float)sb_zone_target(p_z, share, nec, ncar, pw, ew, cw, (double)rew_scale);
        own[s / CN_ZPA] += sb_zone_target(p_z, 0.0, 0.0, 0.0, pw, ew, cw, (double)rew_scale);
    }
    if (ahu)
        for (int k = 0; k < CN_K && k < cm->K; k++) o[CN_ZS + k] = (float)(own[k] + ec);
}

// Exercise the actual zone-reward kernel used by training, with one empty and
// one occupied building and absent slots. Called by the trainer's policy gate.
extern "C" int sb_zone_credit_contract_check() {
    struct Fixture {
        CanonMap map; GpuSim sim[2]; double occupancy[4], temp[4], parts[2][8];
        double scalar[8], heat[1], cool[1]; int step[2]; float rows[2 * CN_ZS];
    };
    Fixture* f=nullptr;
    if(cudaMallocManaged(&f,sizeof(Fixture))!=cudaSuccess)return 1;
    memset(f,0,sizeof(Fixture));
    for(int s=0;s<CN_ZS;++s)f->map.zV[s]=-1;
    f->map.zV[0]=0;f->map.zV[1]=1;
    f->scalar[0]=100;f->scalar[1]=40;f->scalar[2]=1;
    f->scalar[5]=.6;f->scalar[6]=.2;f->scalar[7]=.2;
    f->heat[0]=293;f->cool[0]=297;
    f->occupancy[2]=2;f->occupancy[3]=2;
    f->temp[0]=f->temp[2]=295;f->temp[1]=f->temp[3]=290;
    for(int e=0;e<2;++e){
        f->sim[e].n_zones=2;f->sim[e].dt_sec=300;
        f->sim[e].avg_temp=f->temp+2*e;f->sim[e].rw_scalars=f->scalar;
        f->sim[e].rparts=f->parts[e];f->sim[e].sched_heat_sp=f->heat;f->sim[e].sched_cool_sp=f->cool;
        f->parts[e][1]=.4;f->parts[e][2]=.5;
    }
    int failures=0;
    for(int enabled=0;enabled<=1;++enabled){
        if(cudaMemcpyToSymbol(d_zone_night_share,&enabled,sizeof(enabled))!=cudaSuccess){++failures;break;}
        zm_rz_k<<<1,2>>>(&f->map,f->sim,f->occupancy,f->step,f->rows,2,2,1.0f,CN_ZS);
        if(cudaDeviceSynchronize()!=cudaSuccess){++failures;break;}
        double fuel=sb_zone_target(0.,1.,.4,.5,.6,.2,.2,1.);
        if(fabs((double)f->rows[0]-(enabled?fuel/2:0))>1e-6 ||
           fabs((double)f->rows[1]-(enabled?fuel/2:0))>1e-6)++failures;
        double p0=sb_zone_productivity(295,293,297,1,100,40,2,4,300);
        double p1=sb_zone_productivity(290,293,297,1,100,40,2,4,300);
        double total=sb_zone_target(p0+p1,1.,.4,.5,.6,.2,.2,1.);
        if(fabs((double)f->rows[CN_ZS]+f->rows[CN_ZS+1]-total)>1e-5)++failures;
        for(int e=0;e<2;++e)for(int s=2;s<CN_ZS;++s)if(f->rows[e*CN_ZS+s]!=0.f)++failures;
    }
    int off=0;cudaMemcpyToSymbol(d_zone_night_share,&off,sizeof(off));cudaFree(f);
    printf("%s actual zone reward kernel: empty fuel share, occupied sum and absent slots (failures=%d)\n",
           failures?"FAIL":"PASS",failures);
    return failures;
}

static void zm_zone_reward() {
    if (!g_zone_critic || zm_rz == nullptr) return;
    zm_rz_k<<<(nenv + 127) / 128, 128, 0, stream>>>(
        d_cmap, d_sims, d_V, d_stepv, zm_rz + (size_t)zm_base * zm_rzw, nenv, n_zones, g_rew_scale, zm_rzw);
}


/* SB_ZCHECK=1: every step, for each present slot, (a) zV must be the zone that the slot's VAV serves
 * in the physics tables, and (b) the slot's observed temperature (the sensed value) must equal the
 * previous step's average temperature of zone zV (the observation lags the zone average by one step).
 * [0] slot-steps, [1] (a) failures, [2] (b) failures > 0.01 K, [3] max (b) |difference| in mK. */
static int g_zcheck = -1;
static unsigned long long* zm_zcheck = nullptr;
static double* zm_zprev = nullptr;
__global__ void zm_zcheck_k(const CanonMap* cm, GpuSim* sims, const float* obs, double* prev, int nenv, int nz,
                            unsigned long long* cnt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv || cm == nullptr) return;
    const GpuSim* sim = &sims[e];
    const float* o = obs + (size_t)e * SB_OBS;
    int zones = sim->n_zones < nz ? sim->n_zones : nz;
    for (int s = 0; s < CN_ZS; s++) {
        if (cm->zT[s] < 0) continue;
        atomicAdd(&cnt[0], 1ull);
        int z = cm->zV[s];
        int v = sim->f_idx[cm->zT[s]];
        int zt = (sim->vav_zone_off[v + 1] - sim->vav_zone_off[v] == 1) ? sim->vav_zone_idx[sim->vav_zone_off[v]] : -1;
        if (z != zt || z < 0 || z >= zones) { atomicAdd(&cnt[1], 1ull); continue; }
        double* pv = &prev[(size_t)e * CN_ZS + s];
        if (*pv > 0.0) {
            double d = fabs(*pv - (double)zm_nat(cm, o, cm->zT[s]));
            if (d > 0.01) atomicAdd(&cnt[2], 1ull);
            atomicMax(&cnt[3], (unsigned long long)(d * 1000.0));
        }
        *pv = sim->avg_temp[z];
    }
}
/* Allocated in zm_alloc, before any graph capture. */
static void zm_zone_check_init(int agents) {
    if (g_zcheck >= 0) return;
    const char* v = getenv("SB_ZCHECK");
    g_zcheck = (v != NULL && atoi(v) != 0) ? 1 : 0;
    if (g_zcheck) {
        cudaMalloc(&zm_zcheck, 4 * sizeof(unsigned long long));
        cudaMemset(zm_zcheck, 0, 4 * sizeof(unsigned long long));
        cudaMalloc(&zm_zprev, (size_t)agents * CN_ZS * sizeof(double));
        cudaMemset(zm_zprev, 0, (size_t)agents * CN_ZS * sizeof(double));
    }
}
static void zm_zone_check() {
    if (g_zcheck != 1 || zm_zcheck == nullptr || d_cmap == nullptr) return;
    zm_zcheck_k<<<(nenv + 127) / 128, 128, 0, stream>>>(d_cmap, d_sims, g_obs, zm_zprev + (size_t)zm_base * CN_ZS,
                                                       nenv, n_zones, zm_zcheck);
}
static void zm_zone_check_report() {
    if (zm_zcheck == nullptr) return;
    unsigned long long h[4] = {0, 0, 0, 0};
    cudaDeviceSynchronize();
    cudaMemcpy(h, zm_zcheck, sizeof(h), cudaMemcpyDeviceToHost);
    fprintf(stderr, "smart_buildings: zone map check: %llu slot-steps, %llu wrong zone, %llu temperature mismatches > 0.01 K"
            " (observation vs previous zone average), max |diff| %.3f K\n", h[0], h[1], h[2], (double)h[3] / 1000.0);
}

static void zm_alloc(int agents) {
    zm_zone_check_init(agents);
    zm_total = agents;
    if (agents <= zm_cap) return;
    if (zm_feat) {
        cudaFree(zm_feat); cudaFree(zm_ringT); cudaFree(zm_ringD); cudaFree(zm_count);
        cudaFree(zm_oat); cudaFree(zm_fresh); cudaFree(zm_rng);
        cudaFree(zm_aux); cudaFree(zm_critic);
        if (zm_zshare) { cudaFree(zm_zshare); cudaFree(zm_zshare_tr); cudaFree(zm_zshare_bt); }
        zm_zshare = zm_zshare_tr = zm_zshare_bt = nullptr;
        if (zm_rz) { cudaFree(zm_rz); cudaFree(zm_rz_tr); cudaFree(zm_rz_bt); }
        zm_rz = zm_rz_tr = zm_rz_bt = nullptr;
    }
    zm_cap = agents;
    cudaMalloc(&zm_feat, (size_t)agents * CN_ZS * ZM_FEAT * sizeof(float));
    cudaMalloc(&zm_aux, (size_t)agents * 4 * sizeof(float));
    cudaMalloc(&zm_critic, (size_t)agents * CN_ZS * 2 * sizeof(float));
    cudaMalloc(&zm_ringT, (size_t)agents * CN_ZS * 48 * sizeof(float));
    cudaMalloc(&zm_ringD, (size_t)agents * CN_ZS * 48 * sizeof(float));
    cudaMalloc(&zm_count, (size_t)agents * CN_ZS * sizeof(int));
    cudaMalloc(&zm_oat, (size_t)agents * sizeof(float));
    cudaMalloc(&zm_fresh, (size_t)agents * sizeof(int));
    cudaMalloc(&zm_rng, (size_t)agents * sizeof(unsigned long long));
    cudaMemset(zm_feat, 0, (size_t)agents * CN_ZS * ZM_FEAT * sizeof(float));
    cudaMemset(zm_aux, 0, (size_t)agents * 4 * sizeof(float));
    cudaMemset(zm_critic, 0, (size_t)agents * CN_ZS * 2 * sizeof(float));
    cudaMemset(zm_count, 0, (size_t)agents * CN_ZS * sizeof(int));
    cudaMemset(zm_fresh, 1, (size_t)agents * sizeof(int));
    cudaMemset(zm_rng, 0, (size_t)agents * sizeof(unsigned long long));
#ifdef SB_SHARED_POLICY
    sb_set_feat(zm_feat);
#endif
    zm_traj_h = 1024;
    /* The rollout that trains is train.horizon (288 here). 1024 is the cap the
       single-building runs fit. A 960-env batch at 1024 does not fit in 24 GB,
       and a shorter cap still holds one rollout. SB_ZM_TRAJ sets that cap. */
    const char* traj_cap = getenv("SB_ZM_TRAJ");
    if (traj_cap != NULL && traj_cap[0] != '\0') {
        int h = atoi(traj_cap);
        if (h > 0 && h <= 8192) zm_traj_h = h;
    }
    if (g_zone_memory)
        fprintf(stderr, "smart_buildings: zone trace horizon %d\n", zm_traj_h);
    if (g_zone_memory) {
        cudaMalloc(&zm_feat_tr, (size_t)zm_traj_h * agents * CN_ZS * ZM_FEAT * sizeof(float));
        cudaMalloc(&zm_aux_tr, (size_t)zm_traj_h * agents * 4 * sizeof(float));
        cudaMalloc(&zm_critic_tr, (size_t)zm_traj_h * agents * CN_ZS * 2 * sizeof(float));
        cudaMalloc(&zm_feat_bt, (size_t)zm_traj_h * agents * CN_ZS * ZM_FEAT * sizeof(float));
        cudaMalloc(&zm_aux_bt, (size_t)zm_traj_h * agents * 4 * sizeof(float));
        cudaMalloc(&zm_critic_bt, (size_t)zm_traj_h * agents * CN_ZS * 2 * sizeof(float));
        cudaMalloc(&zm_parts_bt, (size_t)zm_traj_h * agents * 5 * sizeof(float));
        zm_trace_on = 1;
    }
    if (g_zone_adv) {
        cudaMalloc(&zm_zshare, (size_t)agents * CN_ZS * sizeof(float));
        cudaMalloc(&zm_zshare_tr, (size_t)zm_traj_h * agents * CN_ZS * sizeof(float));
        cudaMalloc(&zm_zshare_bt, (size_t)zm_traj_h * agents * CN_ZS * sizeof(float));
        cudaMemset(zm_zshare, 0, (size_t)agents * CN_ZS * sizeof(float));
    }
    if (g_zone_critic) {
        zm_rzw = g_ahu_critic ? CN_ZS + CN_K : CN_ZS;
        cudaMalloc(&zm_rz, (size_t)agents * zm_rzw * sizeof(float));
        cudaMalloc(&zm_rz_tr, (size_t)zm_traj_h * agents * zm_rzw * sizeof(float));
        cudaMalloc(&zm_rz_bt, (size_t)zm_traj_h * agents * zm_rzw * sizeof(float));
        cudaMemset(zm_rz, 0, (size_t)agents * zm_rzw * sizeof(float));
    }
    const char* path = getenv("SB_TRAJ");
    if (path != NULL && path[0] != '\0' && zm_traj_f == nullptr) {
        cudaMalloc(&zm_traj, (size_t)zm_traj_h * agents * 5 * sizeof(float));
        zm_traj_f = fopen(path, "ab");
        if (zm_traj_f != NULL && ftell(zm_traj_f) == 0) {
            unsigned char magic[4] = {'T', 'R', 'J', '1'};
            int width = 5;
            fwrite(magic, 1, 4, zm_traj_f);
            fwrite(&agents, 4, 1, zm_traj_f);
            fwrite(&width, 4, 1, zm_traj_f);
            fflush(zm_traj_f);
        }
        fprintf(stderr, "smart_buildings: trajectory %s (%d envs, 5 reward parts)\n", path, agents);
    }
    if (g_zone_memory && zm_traj == nullptr)
        cudaMalloc(&zm_traj, (size_t)zm_traj_h * agents * 5 * sizeof(float));
}

static void zm_reset(const int* gate) {
    if (!g_zone_memory || zm_count == nullptr) return;
    zm_reset_k<<<(nenv + 127) / 128, 128, 0, stream>>>(
        zm_count + (size_t)zm_base * CN_ZS, zm_fresh + zm_base, nenv, CN_ZS, gate);
}

static void zm_update() {
    if (!g_zone_memory || zm_feat == nullptr || d_cmap == nullptr) return;
    zm_update_k<<<nenv, ZM_THREADS, 0, stream>>>(
        d_cmap, g_obs, d_stepv, d_V,
        zm_feat + (size_t)zm_base * CN_ZS * ZM_FEAT,
        zm_ringT + (size_t)zm_base * CN_ZS * 48,
        zm_ringD + (size_t)zm_base * CN_ZS * 48,
        zm_count + (size_t)zm_base * CN_ZS, zm_oat + zm_base, zm_fresh + zm_base,
        zm_aux + (size_t)zm_base * 4, zm_critic + (size_t)zm_base * CN_ZS * 2,
        nenv, n_zones, SB_OBS);
}

static void zm_pre_temp(const int* gate) {
    if (!g_prestart || zm_rng == nullptr) return;
    zm_prestart_temp_k<<<(nenv + 127) / 128, 128, 0, stream>>>(d_sims, nenv, ncv, n, gate, zm_rng + zm_base);
}

static void zm_pre_clock(const int* gate) {
    if (!g_prestart || zm_rng == nullptr) return;
    zm_prestart_clock_k<<<(nenv + 127) / 128, 128, 0, stream>>>(d_stepv, nenv, gate, zm_rng + zm_base);
#ifndef SB_TEST
    if (g_envs != NULL)
        zm_copy_step_k<<<(nenv + 127) / 128, 128, 0, stream>>>(g_envs, d_stepv, nenv, gate);
#endif
}

static void zm_capture_traj(int t) {
    if (zm_traj == nullptr || t < 0 || t >= zm_traj_h) return;
    zm_traj_k<<<(nenv + 127) / 128, 128, 0, stream>>>(d_rparts, zm_traj, t, nenv);
}

static void zm_flush_traj(int t0, int H) {
    if (zm_traj_f == nullptr || zm_traj == nullptr || H <= 0 || t0 < 0 || t0 + H > zm_traj_h) return;
    std::vector<float> host((size_t)H * nenv * 5);
    cudaMemcpy(host.data(), zm_traj + (size_t)t0 * nenv * 5, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(host.data(), sizeof(float), host.size(), zm_traj_f);
    fflush(zm_traj_f);
}

extern "C" void sb_snap_step(int t, cudaStream_t s) {
    if (t < 0 || t >= zm_traj_h) return;
    /* Groups step concurrently and each writes its own slice of the live buffers.
     * d_rparts belongs to the group that is loaded, so the reward-part copy stays
     * on that group. The feature buffers are the whole batch. */
    int n = zm_total > 0 ? zm_total : nenv;
    if (zm_traj != nullptr)
        zm_traj_k<<<(nenv + 127) / 128, 128, 0, s>>>(d_rparts, zm_traj, t, nenv);
    if (zm_zshare != nullptr)
        zm_copy_k<<<(n + 127) / 128, 128, 0, s>>>(zm_zshare, zm_zshare_tr, t, n, CN_ZS);
    if (zm_rz != nullptr)
        zm_copy_k<<<(n + 127) / 128, 128, 0, s>>>(zm_rz, zm_rz_tr, t, n, zm_rzw);
    if (!zm_trace_on) return;
    zm_copy_k<<<(n + 127) / 128, 128, 0, s>>>(zm_feat, zm_feat_tr, t, n, CN_ZS * ZM_FEAT);
    zm_copy_k<<<(n + 127) / 128, 128, 0, s>>>(zm_aux, zm_aux_tr, t, n, 4);
    zm_copy_k<<<(n + 127) / 128, 128, 0, s>>>(zm_critic, zm_critic_tr, t, n, CN_ZS * 2);
}
extern "C" int sb_trace_on() { return zm_trace_on; }
extern "C" int sb_trace_rows() { return zm_traj_h; }
/* t0 is the first trace row of the rollout slot that trains: the async learner trains slot k
 * while slot 1-k collects, so each slot snaps into its own rows (t0 = slot * horizon). */
extern "C" void sb_trace_prepare(int T, int B, int t0, cudaStream_t s) {
    int expect = zm_total > 0 ? zm_total : nenv;
    if (!zm_trace_on || T <= 0 || t0 < 0 || t0 + T > zm_traj_h || B != expect) return;
    auto go = [&](const float* src, float* dst, int C) {
        size_t n = (size_t)T * B * C;
        zm_transpose_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src + (size_t)t0 * B * C, dst, T, B, C);
    };
    go(zm_feat_tr, zm_feat_bt, CN_ZS * ZM_FEAT);
    go(zm_aux_tr, zm_aux_bt, 4);
    go(zm_critic_tr, zm_critic_bt, CN_ZS * 2);
    if (zm_traj != nullptr) go(zm_traj, zm_parts_bt, 5);
    if (zm_zshare_tr != nullptr) go(zm_zshare_tr, zm_zshare_bt, CN_ZS);
    if (zm_rz_tr != nullptr) go(zm_rz_tr, zm_rz_bt, zm_rzw);
}
extern "C" float* sb_feat_rows(int agent0, int T) { return zm_feat_bt + (size_t)agent0 * T * CN_ZS * ZM_FEAT; }
extern "C" float* sb_aux_rows(int agent0, int T) { return zm_aux_bt + (size_t)agent0 * T * 4; }
extern "C" float* sb_critic_rows(int agent0, int T) { return zm_critic_bt + (size_t)agent0 * T * CN_ZS * 2; }
extern "C" float* sb_parts_rows(int agent0, int T) { return zm_parts_bt + (size_t)agent0 * T * 5; }
extern "C" float* sb_zdelta_rows(int agent0, int T) {
    if (zm_zshare_bt == nullptr) return nullptr;
    return zm_zshare_bt + (size_t)agent0 * T * CN_ZS;
}
extern "C" float* sb_rz_rows(int agent0, int T) {
    if (zm_rz_bt == nullptr) return nullptr;
    return zm_rz_bt + (size_t)agent0 * T * zm_rzw;
}
extern "C" int sb_rz_width() { return zm_rzw; }
extern "C" void sb_traj_flush(int t0, int H) { zm_flush_traj(t0, H); }

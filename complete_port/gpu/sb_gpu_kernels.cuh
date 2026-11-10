/* Device kernels and state for the Smart Buildings port. Shared verbatim by
 * the parity gate (gpu/sb_gpu_sim.cu) and the PufferLib env
 * (ocean/smart_buildings/smart_buildings.cu) so the two can never drift. */
#pragma once
/* M5 device sim step (HVAC + FDM + convection), gated on the M2 actions-only
 * trace. Structure, dictated by measurement:
 *   - per-env sequential work (HVAC chain, occupancy, convection, boiler ramp):
 *     one thread per env, envs in parallel;
 *   - field-sized loops (sensing, applied power, sensed means): global
 *     grid-stride kernels over nenv*n_cvs so the GPU is actually filled;
 *   - the two triangular solves: cuDSS, batched over the group's envs.
 * usage: sb_gpu_sim <blob.sbb> <trace.step> [nenv]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>

extern "C" {
#include "sb_baked.h"
#include "sb_rng.h"

namespace sbgpu {
}

#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + 1e-6 * (double)t.tv_nsec;
}

/* ---- physical constants (mirror sb_sim.c) ---- */
#define RHO_AIR 1.2
#define CP_AIR 1006.0
#define RHO_W 1000.0
#define CP_W 4180.0
#define GRAV 9.8
#define PA_PER_BAR 100000.0
#define ON 1
#define MODE_OFF 0
#define MODE_HEAT 1
#define MODE_COOL 2
#define MODE_PASSIVE 3

#define MT_N 624
#define MT_M 397
struct GpuRng { unsigned int mt[MT_N]; unsigned int idx; };

__device__ __forceinline__ void grng_seed(GpuRng* r, unsigned int seed) {
    r->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        unsigned int prev = r->mt[i - 1];
        r->mt[i] = 1812433253u * (prev ^ (prev >> 30)) + (unsigned int)i;
    }
    r->idx = MT_N;
}
/* The twist, written without the modulo (integer division is very slow on the
 * device): identical arithmetic to the %-form, split at the wrap. */
__device__ __forceinline__ void grng_generate(GpuRng* r) {
    for (int i = 0; i < MT_N - MT_M; i++) {
        unsigned int y = (r->mt[i] & 0x80000000u) | (r->mt[i + 1] & 0x7fffffffu);
        r->mt[i] = r->mt[i + MT_M] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    for (int i = MT_N - MT_M; i < MT_N - 1; i++) {
        unsigned int y = (r->mt[i] & 0x80000000u) | (r->mt[i + 1] & 0x7fffffffu);
        r->mt[i] = r->mt[i - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    {
        int i = MT_N - 1;
        unsigned int y = (r->mt[i] & 0x80000000u) | (r->mt[0] & 0x7fffffffu);
        r->mt[i] = r->mt[MT_M - 1] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    r->idx = 0;
}
__device__ __forceinline__ unsigned int grng_temper(unsigned int y) {
    y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
    return y;
}
__device__ __forceinline__ unsigned int grng_u32(GpuRng* r) {
    if (r->idx >= MT_N) grng_generate(r);
    return grng_temper(r->mt[r->idx++]);
}
__device__ __forceinline__ double grng_double(GpuRng* r) {
    unsigned int a = grng_u32(r) >> 5, b = grng_u32(r) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
__device__ __forceinline__ unsigned int grng_interval(GpuRng* r, unsigned int max) {
    if (max == 0) return 0;
    unsigned int mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16;
    unsigned int v = grng_u32(r) & mask;
    while (v > max) v = grng_u32(r) & mask;
    return v;
}

/* The convection's word stream, generated in parallel and consumed in order.
 * The generator advances the MT state a round at a time; a round's tempered
 * words are a parallel map and the twist is three parallel passes (element
 * i >= MT_N-MT_M reads i-(MT_N-MT_M), already updated by an earlier pass), so
 * the emitted stream is identical to calling grng_u32 sequentially. The serial
 * consumer only pops registers. */
__device__ __forceinline__ void mt_twist_passes(unsigned int* A, unsigned int* B, int nthr) {
    for (int i = threadIdx.x; i < MT_N; i += nthr) {
        if (i == MT_N - 1) { B[i] = 0u; continue; }
        int ia = i + 1;   /* i < MT_N-1, so the neighbour is not rewritten yet */
        unsigned int y = (A[i] & 0x80000000u) | (A[ia] & 0x7fffffffu);
        B[i] = (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        /* for i < MT_N-MT_M the source row is an un-updated old element */
        if (i < MT_N - MT_M) B[i] ^= A[i + MT_M];
    }
    __syncthreads();
    /* The C's in-place loop wraps at i = MT_N-1 to mt[0], which by then is the
     * *updated* element, so this one needs element 0 of the new array. */
    if (threadIdx.x == 0) {
        unsigned int y = (A[MT_N - 1] & 0x80000000u) | (B[0] & 0x7fffffffu);
        B[MT_N - 1] = (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    __syncthreads();
    for (int i = MT_N - MT_M + threadIdx.x; i < 2 * (MT_N - MT_M); i += nthr)
        B[i] ^= B[i - (MT_N - MT_M)];
    __syncthreads();
    for (int i = 2 * (MT_N - MT_M) + threadIdx.x; i < MT_N; i += nthr)
        B[i] ^= B[i - (MT_N - MT_M)];
    __syncthreads();
    for (int i = threadIdx.x; i < MT_N; i += nthr) A[i] = B[i];
    __syncthreads();
}

__global__ void conv_gen_k(const GpuRng* state, unsigned int* out, int n_words) {
    int e = blockIdx.x;
    __shared__ unsigned int A[MT_N], B[MT_N];
    int nthr = blockDim.x;
    for (int i = threadIdx.x; i < MT_N; i += nthr) A[i] = state[e].mt[i];
    __syncthreads();
    int idx = state[e].idx;
    int w = 0;
    for (;;) {
        int avail = MT_N - idx;
        int cnt = (n_words - w < avail) ? (n_words - w) : avail;
        if (cnt > 0) {
            for (int k = threadIdx.x; k < cnt; k += nthr)
                out[(size_t)e * n_words + w + k] = grng_temper(A[idx + k]);
            w += cnt;
            idx += cnt;
        }
        if (w >= n_words) break;
        mt_twist_passes(A, B, nthr);
        idx = 0;
    }
}

/* The saved state must advance by exactly the words the consumer took: one
 * twist per completed round, with the cursor landing where the C's idx would.
 */
__global__ void conv_advance_k(GpuRng* state, const int* consumed) {
    int e = blockIdx.x;
    __shared__ unsigned int A[MT_N], B[MT_N];
    int nthr = blockDim.x;
    int idx = state[e].idx;
    int c = consumed[e];
    int rem = MT_N - idx;
    int twists;
    if (c < rem) {
        idx += c;
        twists = 0;
    } else {
        c -= rem;
        twists = 1 + c / MT_N;
        idx = c % MT_N;
        if (idx == 0) idx = MT_N;
    }
    for (int i = threadIdx.x; i < MT_N; i += nthr) A[i] = state[e].mt[i];
    __syncthreads();
    for (int t = 0; t < twists; t++) mt_twist_passes(A, B, nthr);
    for (int i = threadIdx.x; i < MT_N; i += nthr) state[e].mt[i] = A[i];
    if (threadIdx.x == 0) state[e].idx = idx;
}

/* One tempered word of this step's stream.  The serial collection consumes
 * ~3e5 words per env-step on one thread, so a global read costs its full
 * latency (measured ~250 clocks per pop, 96% of the convection): the block
 * stages the stream in shared memory per cell-chunk and the pop is an LDS.
 * A pop past the staged window falls back to the global buffer, so the result
 * is exact whether or not the window covers it. */
/* One tempered word at a stream position: the shared window when it covers the
 * position, else the global buffer (exact either way).  Takes and returns no
 * state - passing the cursor by reference put the whole cell loop in local
 * memory, which is what made the collection cost ~530 clocks per cell. */
__device__ __forceinline__ unsigned int sbw_at(const unsigned int* gw,
                                               const unsigned int* sw, int pos,
                                               int wb, int wn, int nw) {
    int i = pos - wb;
    if (i >= 0 && i < wn) return sw[i];
    return (pos < nw) ? gw[pos] : 0u;
}

/* Shift-register prefetch of the strictly sequential word stream.  The
 * consumer is one thread, so the depth has to cover a global load's latency
 * (~350-500 clocks) across the pops: 16 keeps the buffer resident in registers
 * (constant indices after unrolling) and the loads in flight. */
#define SB_PREF 16
#define SB_CH 1024
#define SB_W 4096
/* The room-cell -> control-volume offset table (see setup's d_coff) is passed to
 * the per-cell kernels as an argument: every gather would otherwise chase
 * room_cells[2t] and room_cells[2t+1] -- two dependent loads -- per access,
 * which the profile showed dominating them. */

struct Pref {
    const unsigned int* w;
    int n;
    int cur;
    unsigned int b[SB_PREF];
};
__device__ __forceinline__ void pref_init(Pref* p, const unsigned int* w, int n) {
    p->w = w;
    p->n = n;
    #pragma unroll
    for (int k = 0; k < SB_PREF; k++) p->b[k] = (k < n) ? w[k] : 0u;
    p->cur = SB_PREF;
}
__device__ __forceinline__ unsigned int pref_pop(Pref* p) {
    unsigned int v = p->b[0];
    #pragma unroll
    for (int k = 0; k < SB_PREF - 1; k++) p->b[k] = p->b[k + 1];
    p->b[SB_PREF - 1] = (p->cur < p->n) ? p->w[p->cur] : 0u;
    p->cur++;
    return v;
}
/* The buffer already holds tempered words (grng_u32's output), so the pop is
 * the whole of a draw - no second temper. */
__device__ __forceinline__ unsigned int fw_u32(Pref* p) { return pref_pop(p); }
__device__ __forceinline__ double fw_double(Pref* p) {
    unsigned int a = fw_u32(p) >> 5, b = fw_u32(p) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
__device__ __forceinline__ unsigned int fw_interval(Pref* p, unsigned int max) {
    if (max == 0) return 0;
    unsigned int mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    unsigned int v = fw_u32(p) & mask;
    while (v > max) v = fw_u32(p) & mask;
    return v;
}

/* Per-env simulation state + shared (blob) tables. Mirrors SbSim. */
struct GpuSim {
    int n_ahu, n_vav, n_zones, n_rooms, k, nr, nc, n_cvs, n_solved, n_action;
    int step_index, start_index, dwell;
    const int *vav_zone_off, *vav_zone_idx, *vav_ahu_idx, *coupled_idx;
    const int *phys_room_idx, *room_off, *room_cells;
    const double *zone_area, *ahu_const, *vav_const;
    const uint8_t *act_kind, *act_act;
    const int *act_idx;
    const double *sched_comfort, *sched_heat_sp, *sched_cool_sp, *amb;
    const double *boiler_c;
    double internal_w_m2, standby_frac, occupant_w, cv_area_m2, h, header_res, pump_head;
    double dt_sec;
    /* per-env mutable */
    int *ahu_run, *ahu_coolreq, *vav_mode, *vav_prev_valid, *vav_dcmd_set, *vav_rcmd_set;
    double *ahu_recirc, *ahu_sat, *ahu_fan, *ahu_flow, *vav_damper, *vav_reheat;
    double *vav_ratio, *vav_heatsp, *vav_coolsp, *vav_dcmd, *vav_rcmd;
    double *avg_temp, *sensing, *vav_area, *ahu_wsum, *ahu_atotal, *ahu_satv;
    double *zone_q, *zq_div, *sensed_mean, *air_g, *air_T, *coil;
    const int* cell_zone;      /* cell -> zone whose gain heats it, -1 for none */
    /* the field and the interior-mass temperatures are stored cell-major for the whole batch:
     * value of cell p of this env = temp[p * ts] (temp points at the env's column) */
    double *temp, *mass_temp;
    int ts;
    double flow_factor_sum, total_reheat_w, header_res_v;
    double boiler_setpoint, boiler_return, boiler_current, boiler_step_change;
    int hws_run, pump_run, hws_heating_requests;
    /* reward / observation (shared tables + per-env scratch) */
    const int *zone_vav_off, *zone_vav_idx;
    const int64_t *rew_hour, *rew_month;
    const uint8_t *rew_wd;
    const double *rw_scalars, *elec_wd, *elec_we, *elec_c, *gas_p;
    const uint8_t *f_kind, *f_obs_id;
    const int *f_idx;
    const double *f_mean, *f_var, *aux_time;
    int n_dev, n_aux, obs_len;
    double gas_carb_rate, pump_eff;
    double *zo_sp, *zo_act, *rew_out;
    double *rparts;   /* SB_NPARTS diagnostics per env (never feeds the reward) */
};

#define SB_NPARTS 8

#include "sb_gpu_lusolve.cuh"

__device__ double d_clampf(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

__device__ double d_ahu_static(GpuSim* s, int a) {
    if (s->ahu_run[a] != ON) return 0.0;
    double sp = s->ahu_fan[a], mn = s->ahu_const[a * 9 + 2];
    return sp > mn ? sp : mn;
}
__device__ double d_pump_dpres(GpuSim* s) {
    if (s->pump_run != ON) return 0.0;
    return GRAV * RHO_W * s->pump_head / PA_PER_BAR;
}
__device__ double d_total_flow_rate(GpuSim* s) {
    double ff = s->flow_factor_sum, dp = d_pump_dpres(s);
    return ff * sqrt(dp / (1.0 + ff * ff * s->header_res)) / 3600.0;
}
__device__ double d_vav_unrestricted(GpuSim* s, int v) {
    const double* c = &s->vav_const[v * 9];
    double stat = d_ahu_static(s, s->vav_ahu_idx[v]);
    double d = s->vav_damper[v] * c[0] * sqrt(stat / c[4]);
    return d > 0.00001 ? d : 0.00001;
}
__device__ double d_vav_flow_demand(GpuSim* s, int v) {
    return d_vav_unrestricted(s, v) * s->vav_ratio[v];
}
__device__ double d_reheat_flow_rate(GpuSim* s, int v) {
    if (s->flow_factor_sum == 0.0) return 0.0;
    const double* c = &s->vav_const[v * 9];
    double ff = s->vav_reheat[v] * c[1];
    return d_total_flow_rate(s) * (ff / s->flow_factor_sum);
}
__device__ double d_reheat_eff(GpuSim* s, int v, double sat, double iwt) {
    double air_cap = CP_AIR * RHO_AIR * d_vav_flow_demand(s, v);
    if (air_cap <= 0.0) return 0.0;
    double water_cap = CP_W * RHO_W * d_reheat_flow_rate(s, v);
    double e = water_cap / air_cap;
    if (e > 1.0) e = 1.0;
    double gap = iwt - sat;
    if (gap <= 0.0) return e;
    const double* c = &s->vav_const[v * 9];
    double headroom = c[3] - sat;
    if (headroom <= 0.0) return 0.0;
    double hr = headroom / gap;
    return e < hr ? e : hr;
}
__device__ double d_reheat_energy(GpuSim* s, int v, double sat, double iwt) {
    return d_reheat_eff(s, v, sat, iwt) * CP_AIR * RHO_AIR * d_vav_flow_demand(s, v)
         * (iwt - sat);
}
__device__ int d_default_control(GpuSim* s, int v, double zt, double hsp, double csp) {
    double mid = 0.5 * (csp - hsp) + hsp;
    if (zt < hsp) s->vav_mode[v] = MODE_HEAT;
    else if (zt > csp) s->vav_mode[v] = MODE_COOL;
    else if (zt < mid && s->vav_mode[v] == MODE_HEAT) s->vav_mode[v] = MODE_HEAT;
    else if (zt > mid && s->vav_mode[v] == MODE_COOL) s->vav_mode[v] = MODE_COOL;
    else s->vav_mode[v] = MODE_OFF;
    return s->vav_mode[v];
}
__device__ int d_thermostat(GpuSim* s, int v, double zt, int kk, int comfort) {
    double hsp = s->sched_heat_sp[kk], csp = s->sched_cool_sp[kk];
    if (comfort) {
        d_default_control(s, v, zt, hsp, csp);
    } else if (s->vav_prev_valid[v] && s->sched_comfort[kk - 1] != 0.0) {
        s->vav_mode[v] = MODE_PASSIVE;
    } else if (s->vav_mode[v] == MODE_PASSIVE && zt > hsp && zt < csp) {
        s->vav_mode[v] = MODE_PASSIVE;
    } else {
        d_default_control(s, v, zt, hsp, csp);
    }
    s->vav_prev_valid[v] = 1;
    return s->vav_mode[v];
}
__device__ void d_vav_update(GpuSim* s, int v, double sensing, int kk, int comfort) {
    const double* c = &s->vav_const[v * 9];
    s->vav_heatsp[v] = s->sched_heat_sp[kk];
    s->vav_coolsp[v] = s->sched_cool_sp[kk];
    int mode = d_thermostat(s, v, sensing, kk, comfort);
    double damper, reheat;
    if (mode == MODE_HEAT) { damper = 1.0; reheat = 1.0; }
    else if (mode == MODE_COOL) { damper = 1.0; reheat = 0.0; }
    else if (mode == MODE_OFF || mode == MODE_PASSIVE) { damper = c[2]; reheat = 0.0; }
    else { damper = s->vav_damper[v]; reheat = s->vav_reheat[v]; }
    if (s->vav_dcmd_set[v]) damper = s->vav_dcmd[v];
    if (s->vav_rcmd_set[v]) reheat = s->vav_rcmd[v];
    s->vav_damper[v] = damper > c[2] ? damper : c[2];
    s->vav_reheat[v] = reheat;
}

/* occupancy: one thread per env, exact port of sb_occup_advance */
__global__ void occup_k(const int64_t* hour_tab, long hour_stride, const unsigned char* wd_tab,
                        const int* envw, long wd_stride, const int* step_of_env, int n_zones, int nassign,
                        int ea, int la, int ed, int ld, double p_arr, double p_dep,
                        int* state, double* V, GpuRng* rngs, const int* guard) {
    if (guard != NULL && guard[0] == 0) return;
    if (threadIdx.x) return;
    int e = blockIdx.x;
    if (guard != NULL && guard[1 + e] == 0) return;   /* per-env restart flag */
    GpuRng r = rngs[e];
    int step = step_of_env[e];
    int hour = (int)hour_tab[(size_t)envw[e] * hour_stride + step];
    int wd = (wd_tab[(size_t)envw[e] * wd_stride + step] != 0);   /* the env's weather's calendar */
    int* st = state + (size_t)e * n_zones * nassign;
    double* v = V + (size_t)e * n_zones;
    for (int z = 0; z < n_zones; z++) {
        int cnt = 0;
        for (int a = 0; a < nassign; a++) {
            int* s = &st[(size_t)z * nassign + a];
            if (!wd) {
                *s = 0;
            } else if (*s == 0) {
                if (hour >= ea && hour <= la) {
                    if (grng_double(&r) < p_arr) *s = 1;
                }
            } else {
                if (hour < ed) {
                } else if (hour >= ld) {
                    *s = 0;
                } else if (grng_double(&r) < p_dep) {
                    *s = 0;
                }
            }
            cnt += (*s != 0);
        }
        v[z] = (double)cnt;
    }
    rngs[e] = r;
}

/* ---- per-env sequential HVAC phase (thread 0 of each env's block) ---- */
__global__ void seq_k(GpuSim* sims, const double* occV,
                      const double* act_native, int n_act_all, int apply_only) {
    int e = blockIdx.x;
    if (threadIdx.x != 0) return;
    GpuSim* s = &sims[e];
    const double* native = act_native + (size_t)e * n_act_all;
    if (apply_only) {
        for (int i = 0; i < s->n_action; i++) {
            double val = native[i];
            int kind = s->act_kind[i], idx = s->act_idx[i], act = s->act_act[i];
            if (kind == 0) {
                const double* c = &s->ahu_const[idx * 9];
                if (act == 0) s->ahu_sat[idx] = val;
                else if (act == 1) s->ahu_fan[idx] = val;
                else if (act == 2) s->ahu_recirc[idx] = d_clampf(val, 0.0, c[3]);
                else s->ahu_run[idx] = (val == 1.0) ? ON : 0;
            } else if (kind == 1) {
                if (act == 0) { s->vav_dcmd[idx] = d_clampf(val, 0.0, 1.0); s->vav_dcmd_set[idx] = 1; }
                else { s->vav_rcmd[idx] = d_clampf(val, 0.0, 1.0); s->vav_rcmd_set[idx] = 1; }
            } else {
                if (act == 0) s->boiler_setpoint = val;
                else if (act == 1) s->pump_head = val / (GRAV * RHO_W / PA_PER_BAR);
                else { int on = (val == 1.0) ? ON : 0; s->hws_run = on; s->pump_run = on; }
            }
        }
        return;
    }
    int kk = s->step_index;
    double ambient = s->amb[kk];
    int comfort = (s->sched_comfort[kk] != 0.0);
    int na = s->n_ahu, nv = s->n_vav, nz = s->n_zones;
    s->hws_heating_requests = nv;
    for (int a = 0; a < na; a++) { s->ahu_wsum[a] = 0.0; s->ahu_atotal[a] = 0.0; }
    for (int v = 0; v < nv; v++) {
        int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
        double area = 0.0, wt = 0.0;
        for (int t = z0; t < z1; t++) {
            int z = s->vav_zone_idx[t];
            area += s->zone_area[z];
            wt += s->avg_temp[z] * s->zone_area[z];
        }
        s->vav_area[v] = area;
        if (area <= 0.0) { s->sensing[v] = 0.0; continue; }
        double sensing = wt / area;
        s->sensing[v] = sensing;
        d_vav_update(s, v, sensing, kk, comfort);
        int a = s->vav_ahu_idx[v];
        s->ahu_wsum[a] += sensing * area;
        s->ahu_atotal[a] += area;
    }
    for (int a = 0; a < na; a++) {
        double recirc;
        if (s->ahu_atotal[a] > 0.0) recirc = s->ahu_wsum[a] / s->ahu_atotal[a];
        else {
            double mean = 0.0;
            for (int p = 0; p < s->n_cvs; p++) mean += s->temp[(size_t)p * s->ts];
            recirc = mean / (double)s->n_cvs;
        }
        double mixed = s->ahu_recirc[a] * recirc + (1.0 - s->ahu_recirc[a]) * ambient;
        s->ahu_satv[a] = (mixed > s->ahu_sat[a] && s->ahu_run[a] == ON) ? s->ahu_sat[a] : mixed;
    }
    for (int a = 0; a < na; a++) { s->ahu_flow[a] = 0.0; s->ahu_coolreq[a] = 0; }
    s->flow_factor_sum = 0.0;
    for (int v = 0; v < nv; v++) s->flow_factor_sum += s->vav_reheat[v] * s->vav_const[v * 9 + 1];
    double demand[256];
    for (int a = 0; a < na; a++) demand[a] = 0.0;
    for (int v = 0; v < nv; v++) demand[s->vav_ahu_idx[v]] += d_vav_unrestricted(s, v);
    for (int v = 0; v < nv; v++) {
        int a = s->vav_ahu_idx[v];
        double d = demand[a];
        if (d > 0.0) { double r = s->ahu_const[a * 9 + 1] / d; s->vav_ratio[v] = r < 1.0 ? r : 1.0; }
        else s->vav_ratio[v] = 1.0;
    }
    for (int z = 0; z < nz; z++) s->zone_q[z] = 0.0;
    s->total_reheat_w = 0.0;
    for (int col = 0; col < s->k; col++) {
        int v = s->coupled_idx[col];
        if (s->vav_area[v] <= 0.0) { s->air_g[col] = 0.0; s->air_T[col] = 0.0; continue; }
        int a = s->vav_ahu_idx[v];
        double sat = s->ahu_satv[a];
        s->air_T[col] = sat;
        double rw = d_reheat_energy(s, v, sat, s->boiler_current);
        s->total_reheat_w += rw;
        s->coil[col] = sat;
        double fd = d_vav_flow_demand(s, v);
        if (s->vav_damper[v] > 0.0 && s->vav_const[v * 9] > 0.0) {
            s->air_g[col] = fd * RHO_AIR * CP_AIR;
            int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
            for (int t = z0; t < z1; t++) {
                int z = s->vav_zone_idx[t];
                s->zone_q[z] += rw * s->zone_area[z] / s->vav_area[v];
            }
        } else s->air_g[col] = 0.0;
        if (fd > 0.0) {
            s->ahu_flow[a] += fd;
            double mf = s->ahu_const[a * 9 + 1];
            if (s->ahu_flow[a] > mf) s->ahu_flow[a] = mf;
            s->ahu_coolreq[a] += 1;
        }
    }
    if (s->internal_w_m2 > 0.0) {
        double frac = comfort ? 1.0 : s->standby_frac;
        double wpc = s->internal_w_m2 * s->cv_area_m2 * frac;
        for (int z = 0; z < nz; z++) {
            int rm = s->phys_room_idx[z];
            s->zone_q[z] += wpc * (double)(s->room_off[rm + 1] - s->room_off[rm]);
        }
    }
    if (s->occupant_w > 0.0)
        for (int z = 0; z < nz; z++) s->zone_q[z] += occV[(size_t)e * s->n_zones + z] * s->occupant_w;
}

/* avg_temp: one thread per (env, zone).  The sum over the zone's cells must stay
 * in ascending cell order (floating-point addition is not associative, and the
 * CPU core's order is the reference), so each zone is one serial chain - the
 * chains run in parallel across (env, zone), a warp's 32 lanes adding at once
 * (a lone active lane would still occupy the whole FP64 pipe slot).  What the
 * old loop could not do is overlap its loads: every cell cost a dependent
 * coff -> temp pair (~390 clocks each).  The next SB_PF cells' operands are now
 * fetched together, then added in order: same additions, same order,
 * bit-identical. */
#define SB_PF 32
__global__ void avg_temp_k(GpuSim* sims, int nenv, const int* coff, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * sims[0].n_zones) return;
    /* zone-major: the lanes of a warp are envs of one zone, so the cell-offset loads are uniform */
    int z = idx / nenv, e = idx % nenv;
    if (gate != NULL && gate[1 + e] == 0) return;
    GpuSim* s = &sims[e];
    int rm = s->phys_room_idx[z];
    int a = s->room_off[rm], en = s->room_off[rm + 1];
    const double* __restrict__ T = s->temp;
    const size_t ts = s->ts;
    double sum = 0.0;
    int t = a;
    for (; t + SB_PF <= en; t += SB_PF) {
        int ci[SB_PF];
        double v[SB_PF];
#pragma unroll
        for (int i = 0; i < SB_PF; i++) ci[i] = coff[t + i];
#pragma unroll
        for (int i = 0; i < SB_PF; i++) v[i] = T[(size_t)ci[i] * ts];
#pragma unroll
        for (int i = 0; i < SB_PF; i++) sum += v[i];
    }
    for (; t < en; t++) sum += T[(size_t)coff[t] * ts];
    s->avg_temp[z] = sum / (double)(en - a);
}

/* The internal gain of a zone is spread evenly over the zone's cells: q = zone_q / n_cells.
 * The per-cell array this used to fill (and the forcing re-read) is replaced by the one
 * division per zone, and the forcing looks the value up through cell_zone - the same
 * double, computed once. */
__global__ void zone_qdiv_k(GpuSim* sims, int nenv) {
    const int nz = sims[0].n_zones;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * nz) return;
    const int e = idx / nz, z = idx % nz;
    GpuSim* s = &sims[e];
    const int rm = s->phys_room_idx[z];
    const int a = s->room_off[rm], en = s->room_off[rm + 1];
    s->zq_div[z] = s->zone_q[z] / (double)(en - a);
}

/* ---- FDM phase: forcing -> batched cuDSS solve -> air side -> mass/scatter ---- */
/* The ambient-exposure term of the forcing depends only on the scenario (face
 * areas, conduction coefficients, the constant convection coefficient h), not on
 * the state, so  dir[t] + exposure[t]  is evaluated once at setup with exactly the
 * expression the per-step kernel used to evaluate, instead of re-deriving it for
 * every env every step (five table gathers and two divisions per cell). */
__global__ void exdir_k(int n, const int* solved, const double* dir, const double* a0,
                        const double* a1, const double* horiz, const double* vert,
                        const double* cond, double h, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int p = solved[t];
    double ex = 0.0;
    double x0 = a0[p];
    if (x0 > 0.0) ex += h * x0 / (1.0 + h * 0.5 * horiz[p] / cond[p]);
    double x1 = a1[p];
    if (x1 > 0.0) ex += h * x1 / (1.0 + h * 0.5 * vert[p] / cond[p]);
    out[t] = dir[t] + ex;
}
/* The forcing, written straight into the solver's layout: work[row][ld] with row =
 * perm_r[t] (the solve's input permutation) and the envs along the row.  Done as a
 * 32 x 32 tile transpose through shared memory so both the reads (a cell's values
 * along t, per env) and the writes (a row's envs, contiguous) are coalesced.
 * Columns >= nenv (padding up to ld) are written as 0 by the tiles that cover them
 * and otherwise stay at their zero initialisation. */
__global__ void __launch_bounds__(256)
fdm_forcing_k(GpuSim* sims, const int* solved, const double* cap, const double* exdir,
              const int* envw, const double* mc, const int* rowpos, int n, int nenv, int ld, sb_wt* work) {
    /* thread = (cell t, env e), envs fastest: the field, the mass temperatures and the solver's
     * work array are all [cell][env], so every access is coalesced.  Padding columns get 0. */
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * ld) return;
    const int t = (int)(idx / ld), e = (int)(idx % ld);
    double f = 0.0;
    if (e < nenv) {
        GpuSim* s = &sims[e];
        const int p = solved[t];
        const double amb = s->amb[s->step_index];
        const int zc = s->cell_zone[p];
        const double qv = zc >= 0 ? s->zq_div[zc] : 0.0;
        f = cap[t] * s->temp[(size_t)p * ld] + qv + amb * exdir[(size_t)envw[e] * n + t] + mc[t] * s->mass_temp[(size_t)t * ld];
    }
    work[(size_t)rowpos[t] * ld + e] = (sb_wt)f;
}
/* The solve's output with the epilogue of the step's FDM phase: for solved cell t (row pc[t] of the
 * backward result, already air-side corrected) the interior-mass update and the field write.  Thread
 * = (cell, env), envs fastest; everything is [cell][env], so all accesses are coalesced. */
__global__ void __launch_bounds__(256)
lu_out_k(GpuSim* sims, const sb_wt* zout, const int* pc, const int* solved, const double* mret,
         const double* mgain, int n, int nenv, int ld, double* temp, double* mass) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * ld) return;
    const int t = (int)(idx / ld), e = (int)(idx % ld);
    if (e >= nenv) return;
    const double yv = zout[(size_t)pc[t] * ld + e];
    const size_t mi = (size_t)t * ld + e;
    mass[mi] = mass[mi] * mret[t] + mgain[t] * yv;
    temp[(size_t)solved[t] * ld + e] = yv;
    (void)sims;
}
__global__ void lu_perm_out_k(const sb_wt* zout, const int* pc, double* y, int n, int ncol, int ld) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * ncol) return;
    int e = (int)(i / n), kx = (int)(i % n);
    y[i] = zout[(size_t)pc[kx] * ld + e];
}
/* setup-time variants (k columns, not performance critical) */
__global__ void lu_perm_in_k(const double* b, const int* rowpos, sb_wt* work, int n, int ncol, int ld) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * ld) return;
    int t = (int)(i / ld), e = (int)(i % ld);
    work[(size_t)rowpos[t] * ld + e] = (sb_wt)(e < ncol ? b[(size_t)e * n + t] : 0.0);
}

/* S = sensed @ A^-1 @ applied (k x k) -- shared by all envs (one matrix) */
__global__ void applied_cols_k(const int* ap, const int* ai, const double* ax,
                               double* R, int k, int n) {
    int j = blockIdx.x;
    if (j >= k) return;
    for (int p = ap[j] + threadIdx.x; p < ap[j + 1]; p += blockDim.x)
        R[(size_t)j * n + ai[p]] = ax[p];
}
__global__ void sensed_resp_k(const int* sp, const int* si, const double* sx,
                              const double* yr, double* SR, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * k) return;
    int i = idx / k, j = idx % k;
    const double* col = yr + (size_t)j * n;
    double a = 0.0;
    for (int p = sp[i]; p < sp[i + 1]; p++) a += sx[p] * col[si[p]];
    SR[i * k + j] = a;
}

/* The air side's sensed temperatures of the uncorrected solve, sensed (A^-1 b), read off the
 * FORWARD result z = L^-1 P_r b:  sensed_i = sum_row R[i,row] z[row] with R = sensed P_c U^-1
 * (precomputed, CSR over the k sensors, solver-row columns ascending).  Each sensor row is cut into
 * fixed segments of SB_RSEG entries (seg_p: [start, end) of each segment, row_seg: the segments of
 * each sensor; the cut depends only on the pattern); one warp per (segment, 32 envs), lane = env (z is [row][ld], so every
 * load is one coalesced row segment), four partial sums in a fixed interleave, combined in a fixed
 * order; then the segments of a sensor are added in order.  Deterministic. */
#define SB_RSEG 256
__global__ void sensed_R_part_k(const int* seg_p, const int* rc, const double* rx0, long nnz,
                                const int* envw, const sb_wt* z, int ld, double* part) {
    const int sg = blockIdx.x;
    const int e = blockIdx.y * 32 + threadIdx.x;
    if (e >= ld) return;
    const double* rx = rx0 + (size_t)envw[e] * nnz;    /* this env's weather's values */
    const int pb = seg_p[2 * sg], pe = seg_p[2 * sg + 1];
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    int p = pb;
    for (; p + 8 <= pe; p += 8) {
        int c[8]; double x[8], v[8];
#pragma unroll
        for (int t = 0; t < 8; t++) { c[t] = rc[p + t]; x[t] = rx[p + t]; }
#pragma unroll
        for (int t = 0; t < 8; t++) v[t] = z[(size_t)c[t] * ld + e];
        a0 = __fma_rn(x[0], v[0], a0); a1 = __fma_rn(x[1], v[1], a1);
        a2 = __fma_rn(x[2], v[2], a2); a3 = __fma_rn(x[3], v[3], a3);
        a0 = __fma_rn(x[4], v[4], a0); a1 = __fma_rn(x[5], v[5], a1);
        a2 = __fma_rn(x[6], v[6], a2); a3 = __fma_rn(x[7], v[7], a3);
    }
    for (; p < pe; p++) a0 = __fma_rn(rx[p], (double)z[(size_t)rc[p] * ld + e], a0);
    part[(size_t)sg * ld + e] = (a0 + a1) + (a2 + a3);
}
__global__ void sensed_R_sum_k(GpuSim* sims, const int* row_seg, const double* part, int nenv, int k, int ld) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * nenv) return;
    const int i = idx / nenv, e = idx % nenv;
    double a = 0.0;
    for (int sg = row_seg[i]; sg < row_seg[i + 1]; sg++) a += part[(size_t)sg * ld + e];
    sims[e].sensed_mean[i] = a;
}

/* air side, one block per env: power, k x k solve, hand q to the backward solve */
#define AIR_NAME airside_k
#define AIR_GLOBAL 0
#include "sb_gpu_airside.inc"
#undef AIR_NAME
#undef AIR_GLOBAL
#define AIR_NAME airside_gk
#define AIR_GLOBAL 1
#include "sb_gpu_airside.inc"
#undef AIR_NAME
#undef AIR_GLOBAL

/* The cells the FDM does not solve (the exterior / Dirichlet ones) hold the ambient
 * temperature; the solved ones are written by ycorr_k.  Same final field as filling
 * every cell and then overwriting the solved ones, without writing them twice. */
__global__ void fill_amb_k(GpuSim* sims, const int* cells, int ncells, int nenv) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)nenv * ncells) return;
    int i = (int)(idx / nenv), e = (int)(idx % nenv);      /* envs fastest: coalesced */
    GpuSim* s = &sims[e];
    s->temp[(size_t)cells[i] * s->ts] = s->amb[s->step_index];
}

/* ---- post phase ---- */
/* sensed_mean: one thread per column j, summing over p in ASCENDING order
 * (precomputed CSR of the transposed air-side sensed matrix) so the order
 * matches the CPU exactly. */
__global__ void sensed_mean_k(GpuSim* sims, const int* tp, const int* ti,
                              const double* tx, int nenv, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * k) return;
    int j = idx / nenv, e = idx % nenv;   /* sensor-major: uniform index / weight loads per warp */
    GpuSim* s = &sims[e];
    const double* __restrict__ T = s->temp;
    const size_t ts = s->ts;
    double a = 0.0;
    int p = tp[j];
    const int pe = tp[j + 1];
    for (; p + SB_PF <= pe; p += SB_PF) {
        int ii[SB_PF];
        double xx[SB_PF], v[SB_PF];
#pragma unroll
        for (int i = 0; i < SB_PF; i++) { ii[i] = ti[p + i]; xx[i] = tx[p + i]; }
#pragma unroll
        for (int i = 0; i < SB_PF; i++) v[i] = T[(size_t)ii[i] * ts];
#pragma unroll
        for (int i = 0; i < SB_PF; i++) a = __fma_rn(xx[i], v[i], a);
    }
    for (; p < pe; p++) a = __fma_rn(tx[p], T[(size_t)ti[p] * ts], a);
    s->sensed_mean[j] = a;
}
__global__ void post_seq_k(GpuSim* sims, int nenv) {
    int e = blockIdx.x;
    if (threadIdx.x != 0) return;
    GpuSim* s = &sims[e];
    int nz = s->n_zones;
    if (s->k > 0) {
        for (int col = 0; col < s->k; col++) {
            int v = s->coupled_idx[col];
            if (s->vav_area[v] <= 0.0) continue;
            double q = s->air_g[col] * (s->air_T[col] - s->sensed_mean[col]);
            int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
            for (int t = z0; t < z1; t++) {
                int z = s->vav_zone_idx[t];
                s->zone_q[z] += q * s->zone_area[z] / s->vav_area[v];
            }
        }
    }
    /* Boiler return temperature, computed before the ramp: the reward's gas
     * rate is a function of it, so leaving it at its initial value made the
     * reported gas 1.95x too high. */
    {
        double wcap_rate = CP_W * RHO_W * d_total_flow_rate(s);
        if (wcap_rate > 0.0 && s->k > 0) {
            double supply_temp = s->boiler_current;
            double rt = supply_temp - s->total_reheat_w / wcap_rate;
            double at = s->coil[0];
            for (int c = 1; c < s->k; c++) {
                if (s->total_reheat_w >= 0.0) { if (s->coil[c] < at) at = s->coil[c]; }
                else { if (s->coil[c] > at) at = s->coil[c]; }
            }
            double lo = supply_temp < at ? supply_temp : at;
            double hi = supply_temp > at ? supply_temp : at;
            if (rt < lo) rt = lo;
            if (rt > hi) rt = hi;
            s->boiler_return = rt;
        }
    }
    double begin = s->boiler_current;
    double sp = s->boiler_setpoint;
    double dt_min = s->dt_sec / 60.0;
    if (sp > begin) {
        double v = begin + s->boiler_c[2] * dt_min;
        s->boiler_current = v < sp ? v : sp;
    } else if (sp < begin) {
        double v = begin - s->boiler_c[3] * dt_min;
        s->boiler_current = v > sp ? v : sp;
    } else {
        s->boiler_current = sp;
    }
    s->boiler_step_change = s->boiler_current - begin;
    s->step_index += 1;
}
/* convection, one thread per env (exact port) */
/* Convection. sb_conv_apply is a single-threaded per-room chain, and on the
 * device every dependent *global* access on that chain is exposed (~350-500
 * clocks). Two things remove those without changing a single value:
 *   - the room's candidate offsets are staged in shared memory (the serial
 *     collection no longer waits on global loads), and
 *   - instead of moving temperatures one swap at a time (4 global accesses per
 *     pair, each serialised against the previous pair), the swap order is
 *     composed into a permutation in shared memory and then applied in place by
 *     rotating each cycle: every value is moved, never recomputed, so the
 *     result is bit-identical.  Only the touches actually moved touch global
 *     memory (~2m values, not 4 accesses per pair).
 * The candidate-id picks are the one data-dependent global read; they are
 * independent of each other and are resolved for the whole room in parallel.
 */
/* Convection.  sb_conv_apply is a single-threaded per-room chain, and on the
 * device every dependent *global* access on it is exposed (~350-500 clocks):
 * moving temperatures one swap at a time costs four serialised global accesses
 * per pair.  So each room's temperatures are staged in shared memory by the
 * whole block, the C's serial chain (draws, collection, Fisher-Yates, swaps)
 * runs entirely in shared, and the result is scattered back in parallel.  The
 * only global traffic left in the chain is the candidate-id pick, which is
 * independent per entry and therefore resolved for the whole room in parallel,
 * and the candidate offsets, which are prefetched a few cells ahead so their
 * latency hides behind the draw chain.  Values are only ever moved, never
 * recomputed, so the result is bit-identical to the CPU core's.
 */
__global__ void convect_k(const int* room_off, const int* room_cells,
                          const int* cand_off, const int* cand_idx,
                          int n_rooms, int nc, int ncv, int max_room, double p,
                          int distance, double* temp, int* left, int* right,
                          const unsigned int* words, int n_words, int* consumed, int ts) {
    int e = blockIdx.x;
    if (p == 0.0 || distance == 0) return;
    extern __shared__ double s_tmp[];   /* max_room doubles */
    /* Statically typed shared arrays: deriving these from the double[] above
     * made the accesses generic loads rather than LDS. */
    __shared__ unsigned int s_word[SB_W];
    __shared__ int s_cand[SB_CH + 8];
    int* L = left + (size_t)e * max_room;
    int* R = right + (size_t)e * max_room;
    double* T = temp + e;                        /* field is [cell][ts] */
    int nthr = blockDim.x, tid = threadIdx.x;
    __shared__ int s_m, s_wb;
    const unsigned int* gw = words + (size_t)e * n_words;
    int wpos = 0;                 /* words consumed from this step's stream */
    /* sb_rng_double's value is D/2^53 with D = (a<<26)|b an exact 53-bit
     * integer, and it is only ever compared with p, so compare integers:
     * D > p*2^53.  p*2^53 is an exact double, so the truncation is exact. */
    const unsigned long long p_int = (unsigned long long)(p * 9007199254740992.0);
    long long t_g_ = 0, t_c_ = 0, t_r_ = 0, t_s_ = 0, t_x_ = 0, tk_ = 0;

    for (int rm = 0; rm < n_rooms; rm++) {
        int base = room_off[rm];
        int n = room_off[rm + 1] - base;
        if (n <= 0) continue;
        if (tid == 0) tk_ = clock64();
        for (int j = tid; j < n; j += nthr) {
            int q = base + j;
            s_tmp[j] = T[(size_t)(room_cells[2 * q] * nc + room_cells[2 * q + 1]) * ts];
        }
        __syncthreads();
        if (tid == 0) { t_g_ += clock64() - tk_; tk_ = clock64(); }
        {
            /* the cand-offset lookahead lives across chunk boundaries in
             * registers; only thread 0 uses it */
            int mcol = 0;
            for (int c0 = 0; c0 < n; c0 += SB_CH) {
                int cn = (n - c0 < SB_CH) ? (n - c0) : SB_CH;
                if (tid == 0) s_wb = wpos;
                __syncthreads();
                int wb = s_wb, wn = n_words - wb;
                if (wn > SB_W) wn = SB_W;
                if (wn < 0) wn = 0;
                for (int q = tid; q < wn; q += nthr) s_word[q] = gw[wb + q];
                /* The candidate offsets are the *other* per-cell global read and
                 * a 4-cell register lookahead leaves its ~250-clock latency
                 * exposed on every cell; stage this chunk's slice instead. */
                int cend = c0 + cn + 5;
                if (cend > n + 1) cend = n + 1;
                for (int q = c0 + tid; q < cend; q += nthr) s_cand[q - c0] = cand_off[base + q];
                __syncthreads();
                if (tid == 0) {
                    for (int j = c0; j < c0 + cn; j++) {
                        int start = s_cand[j - c0];
                        int end = s_cand[j + 1 - c0];
                        unsigned int wa = ((unsigned)((wpos) - wb) < (unsigned)wn ? s_word[(wpos) - wb] : ((wpos) < n_words ? gw[wpos] : 0u));
                        unsigned int wb2 = ((unsigned)((wpos + 1) - wb) < (unsigned)wn ? s_word[(wpos + 1) - wb] : ((wpos + 1) < n_words ? gw[wpos + 1] : 0u));
                        wpos += 2;
                        unsigned long long D = ((unsigned long long)(wa >> 5) << 26)
                                             | (unsigned long long)(wb2 >> 6);
                        if (D > p_int) continue;
                        if (end <= start) continue;
                        L[mcol] = base + j;
                        {
                            unsigned int mx = (unsigned)(end - start - 1), v = 0;
                            if (mx != 0) {
                                unsigned int msk = mx;
                                msk |= msk >> 1; msk |= msk >> 2; msk |= msk >> 4;
                                msk |= msk >> 8; msk |= msk >> 16;
                                do {
                                    v = ((unsigned)((wpos) - wb) < (unsigned)wn ? s_word[(wpos) - wb] : ((wpos) < n_words ? gw[wpos] : 0u)) & msk;
                                    wpos++;
                                } while (v > mx);
                            }
                            R[mcol] = (int)v;
                        }
                        mcol++;
                    }
                }
                __syncthreads();
            }
            /* the Fisher-Yates draws, with one more window */
            if (tid == 0) s_wb = wpos;
            __syncthreads();
            int wb = s_wb, wn = n_words - wb;
            if (wn > SB_W) wn = SB_W;
            if (wn < 0) wn = 0;
            for (int q = tid; q < wn; q += nthr) s_word[q] = gw[wb + q];
            __syncthreads();
            if (tid == 0) {
                s_m = mcol;
                for (int t = mcol - 1; t > 0; t--) {
                    unsigned int mx = (unsigned)t, v = 0;
                    { unsigned int msk = mx;
                      msk |= msk >> 1; msk |= msk >> 2; msk |= msk >> 4;
                      msk |= msk >> 8; msk |= msk >> 16;
                      do { v = ((unsigned)((wpos) - wb) < (unsigned)wn ? s_word[(wpos) - wb] : ((wpos) < n_words ? gw[wpos] : 0u)) & msk;
                           wpos++; } while (v > mx); }
                    int j2 = (int)v;
                    int tmp = L[t]; L[t] = L[j2]; L[j2] = tmp;
                    tmp = R[t]; R[t] = R[j2]; R[j2] = tmp;
                }
                t_c_ += clock64() - tk_;
            }
        }
        __syncthreads();
        int m = s_m;
        /* the candidate-id picks: independent, so resolve the room's in parallel */
        if (tid == 0) tk_ = clock64();
        for (int t = tid; t < m; t += nthr) {
            int a = L[t], kk = R[t];
            R[t] = cand_idx[cand_off[a] + kk] - base;
        }
        __syncthreads();
        if (tid == 0) { t_r_ += clock64() - tk_; tk_ = clock64();
            for (int t = 0; t < m; t++) {
                int a = L[t] - base, b = R[t];
                double held = s_tmp[a];
                s_tmp[a] = s_tmp[b];
                s_tmp[b] = held;
            }
            t_s_ += clock64() - tk_;
        }
        __syncthreads();
        for (int j = tid; j < n; j += nthr) {
            int q = base + j;
            T[(size_t)(room_cells[2 * q] * nc + room_cells[2 * q + 1]) * ts] = s_tmp[j];
        }
        __syncthreads();
        if (tid == 0) t_x_ += clock64() - tk_;
    }
    (void)t_g_; (void)t_c_; (void)t_r_; (void)t_s_; (void)t_x_; (void)tk_;
    if (tid == 0) consumed[e] = wpos;
}


/* ===========================================================================
 * Convection v2: the same serial MT19937 chain as sb_conv_apply, restructured so
 * the thread that walks it does O(1) work per *accepted* cell instead of per
 * cell, and the MT stream is produced in-kernel (no word buffer, no fixed
 * stream bound, no separate generate/advance kernels).
 *
 * What is exact, and why:
 *   - A cell consumes 2 words (the accept test) and, only if accepted and it has
 *     candidates, 1+ more (the interval draw's rejection loop).  Whether a pair
 *     of words (g, g+1) is an accept is a pure function of those two words, so
 *     the flag for every pair start g is computed in parallel as the words are
 *     generated and kept as a bit ring.  Rejected cells then cost nothing: the
 *     chain jumps straight to the next set bit of the right parity.
 *   - The words are the MT19937 output in order: the twist is the same three
 *     parallel passes as before (mt_twist_passes), the ring holds the tempered
 *     words, the chain pops them in order.  Draws that run past the generated
 *     frontier are retried from their start after more words are generated, so
 *     no word is ever read before it exists and none is skipped.
 *   - The room's swaps are applied to a permutation of the room's cells and
 *     the temperatures are then moved once (read-all, then write-all), which is
 *     the composition of the same transpositions in the same order.  Values are
 *     only moved, never recomputed, so the field is bit-identical.
 *   - The saved generator state advances by exactly the words consumed (the
 *     array holding the last consumed word and the cursor into it).
 * =========================================================================== */
#define CV_RING 4096
#define CV_RING_MASK (CV_RING - 1)
#define CV_BW (CV_RING / 64)            /* 32-bit bit-words per parity */

struct CvChain {
    int j, pos, mcol, phase, t;
};

/* mask of the smallest 2^L-1 covering mx (mx > 0): one clz and one shift */
__device__ __forceinline__ unsigned int cv_mask(unsigned int mx) {
    return 0xFFFFFFFFu >> __clz((int)mx);
}

/* One rejection-sampled draw of numpy's randint(0, mx) from stream position q.
 * The ring carries four mirror words past its end, so the next four words are
 * read with one address and four loads; the draw takes the first with
 * probability >= 1/2, so >99.9% of draws finish inside that window.  On
 * success returns the number of words consumed (>= 1) and the value; a draw that
 * does not finish inside the window falls to cv_draw_slow, which walks on word
 * by word.  Returns -1 when the generated frontier F is reached first; nothing
 * is consumed then and the caller retries from the same q. */
__device__ __noinline__ long long cv_draw_slow(const unsigned int* ring, int F, int q,
                                               unsigned int mx, unsigned int msk) {
    const int q0 = q;
    for (;;) {
        if (q >= F) return -1ll;
        const unsigned int v = ring[q & CV_RING_MASK] & msk;
        q++;
        if (v <= mx) return ((long long)(q - q0) << 32) | (long long)v;
    }
}
__device__ __forceinline__ int cv_draw(const unsigned int* ring, int F, int q,
                                       unsigned int mx, unsigned int msk,
                                       unsigned int& v) {
    if (q + 3 < F) {
        const unsigned int* w = ring + (q & CV_RING_MASK);
        const unsigned int v0 = w[0] & msk, v1 = w[1] & msk, v2 = w[2] & msk,
                           v3 = w[3] & msk;
        if (v0 <= mx) { v = v0; return 1; }
        if (v1 <= mx) { v = v1; return 2; }
        if (v2 <= mx) { v = v2; return 3; }
        if (v3 <= mx) { v = v3; return 4; }
        const long long r = cv_draw_slow(ring, F, q + 4, mx, msk);
        if (r < 0) return -1;
        v = (unsigned int)(r & 0xFFFFFFFFll);
        return (int)(r >> 32) + 4;
    }
    const long long r = cv_draw_slow(ring, F, q, mx, msk);
    if (r < 0) return -1;
    v = (unsigned int)(r & 0xFFFFFFFFll);
    return (int)(r >> 32);
}

/* One resumable burst of the serial chain.  Returns 0 when the room's
 * collection and Fisher-Yates are complete, 1 when it needs words beyond the
 * generated frontier F (state is left at a point from which resuming is exact).
 * Entries of LR hold the cell index in the low half and the interval draw in the
 * high half, so the Fisher-Yates swap moves both with one exchange. */
__device__ __forceinline__ int cv_chain_run(
    CvChain& c, const unsigned int* ring, const unsigned int* bitsE,
    const unsigned int* bitsO, const unsigned short* cnt, unsigned int* LR, int n,
    int F) {
    if (c.phase == 0) {
        const int Fv = F - 2;                              /* last valid pair */
        int cw_idx = -1, cw_par = -1;
        unsigned int cw = 0u;
        while (c.j < n) {
            const int par = c.pos & 1;
            const int gmax = c.pos + 2 * (n - c.j - 1);    /* last cell's start */
            const int glim = gmax < Fv ? gmax : Fv;
            int h = c.pos >> 1;
            const int hlim = (glim - par) >> 1;            /* glim >= pos-1 here */
            int hfound = -1;
            if (glim >= c.pos) {
                const unsigned int* B = par ? bitsO : bitsE;
                while (h <= hlim) {
                    const int wi = (h >> 5) & (CV_BW - 1);
                    if (wi != cw_idx || par != cw_par) { cw = B[wi]; cw_idx = wi; cw_par = par; }
                    const unsigned int x = cw >> (h & 31);
                    if (x != 0u) {
                        const int hh = h + (__ffs((int)x) - 1);
                        if (hh <= hlim) hfound = hh;
                        break;
                    }
                    h = (h | 31) + 1;
                }
            }
            if (hfound < 0) {
                if (glim == gmax) {                 /* no accept to the room end */
                    c.pos += 2 * (n - c.j);
                    c.j = n;
                    break;
                }
                if (glim >= c.pos) {                /* commit the verified rejects */
                    const int ncell = ((glim - c.pos) >> 1) + 1;
                    c.j += ncell;
                    c.pos += 2 * ncell;
                }
                return 1;
            }
            const int gfound = 2 * hfound + par;
            const int jj = c.j + (hfound - (c.pos >> 1));
            const int cc = cnt[jj];
            int q = gfound + 2;
            if (cc == 0) {                          /* accepted, but no candidates */
                c.j = jj + 1;
                c.pos = q;
                continue;
            }
            unsigned int v = 0;
            if (cc > 1) {
                const unsigned int mx = (unsigned int)(cc - 1);
                const int len = cv_draw(ring, F, q, mx, cv_mask(mx), v);
                if (len < 0) { c.j = jj; c.pos = gfound; return 1; }
                q += len;
            }
            LR[c.mcol++] = (unsigned int)jj | (v << 16);
            c.j = jj + 1;
            c.pos = q;
        }
        c.phase = 1;
        c.t = c.mcol - 1;
    }
    while (c.t > 0) {
        const unsigned int mx = (unsigned int)c.t;
        unsigned int v;
        const int len = cv_draw(ring, F, c.pos, mx, cv_mask(mx), v);
        if (len < 0) return 1;
        c.pos += len;
        const int t = c.t;
        const unsigned int a = LR[t], b = LR[v];
        LR[t] = b;
        LR[v] = a;
        c.t--;
    }
    return 0;
}

#define CV2_NAME convect2_k
#define CV2_GLOBAL 0
#include "sb_gpu_convect2.inc"
#undef CV2_NAME
#undef CV2_GLOBAL
#define CV2_NAME convect2g_k
#define CV2_GLOBAL 1
#include "sb_gpu_convect2.inc"
#undef CV2_NAME
#undef CV2_GLOBAL

/* ===========================================================================
 * Episode reset on the device: sb_sim_reset, for every env (or only when the
 * gate flag is set, which is how the in-step restart uses it).
 *   - device state: AHU and VAV controller state back to their baked initial
 *     values, boiler / pump / header scalars, step index 0;
 *   - the temperature and interior-mass fields back to temp_init /
 *     mass_temp_init, the applied internal gains to zero;
 *   - the sensed zone temperatures primed from the initial field, because the
 *     reset observation reports them (the CPU core's sense_avg_temp +
 *     sense_zone_temps) and the post-step path that normally writes them has
 *     not run yet.
 * The convection and occupancy generators are deliberately NOT touched: like the
 * CPU env (and the oracle, whose building is not rebuilt per episode) they keep
 * their state across episodes.  pump_head is not touched either, as in the CPU
 * core.  =========================================================================== */
__global__ void reset_sim_k(GpuSim* sims, int nenv, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    const int e = blockIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && gate[1 + e] == 0) return;
    GpuSim* s = &sims[e];
    for (int a = threadIdx.x; a < s->n_ahu; a += blockDim.x) {
        const double* c = &s->ahu_const[a * 9];
        s->ahu_run[a] = (int)c[8];
        s->ahu_recirc[a] = c[0];
        s->ahu_sat[a] = c[6];
        s->ahu_fan[a] = c[7];
        s->ahu_flow[a] = 0.0;
        s->ahu_coolreq[a] = 0;
    }
    for (int v = threadIdx.x; v < s->n_vav; v += blockDim.x) {
        const double* c = &s->vav_const[v * 9];
        s->vav_damper[v] = c[7];
        s->vav_reheat[v] = c[8];
        s->vav_ratio[v] = 1.0;
        s->vav_heatsp[v] = c[5];
        s->vav_coolsp[v] = c[6];
        s->vav_mode[v] = MODE_OFF;
        s->vav_prev_valid[v] = 0;
        s->vav_dcmd_set[v] = 0;
        s->vav_rcmd_set[v] = 0;
        s->vav_dcmd[v] = 0.0;
        s->vav_rcmd[v] = 0.0;
    }
    if (threadIdx.x == 0) {
        s->flow_factor_sum = 0.0;
        s->hws_heating_requests = 0;
        s->hws_run = ON;
        s->pump_run = ON;
        s->boiler_setpoint = s->boiler_c[0];
        s->boiler_return = s->boiler_c[1];
        s->boiler_current = s->boiler_c[0];
        s->boiler_step_change = 0.0;
        s->step_index = 0;
        s->start_index = 0;
    }
}
__global__ void reset_field_k(GpuSim* sims, int nenv, int ncv, int n, const double* tinit,
                              const double* minit, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    const size_t total = (size_t)nenv * (ncv > n ? ncv : n);
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += stride) {
        const int p = (int)(i / nenv);          /* envs fastest: coalesced */
        const int e = (int)(i % nenv);
        if (gate != NULL && gate[1 + e] == 0) continue;
        GpuSim* s = &sims[e];
        if (p < ncv) s->temp[(size_t)p * s->ts] = tinit[p];
        if (p < n) s->mass_temp[(size_t)p * s->ts] = minit[p];
    }
}
__global__ void reset_stepv_k(int* stepv, int nenv, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < nenv && (gate == NULL || gate[1 + e] != 0)) stepv[e] = 0;
}
/* sense_zone_temps: each VAV's area-weighted zone temperature.  Plain multiply
 * then add, as the CPU core evaluates it. */
__global__ void sense_k(GpuSim* sims, int nenv, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    int nv = sims[0].n_vav;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * nv) return;
    int e = idx / nv, v = idx % nv;
    if (gate != NULL && gate[1 + e] == 0) return;
    GpuSim* s = &sims[e];
    int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
    double area = 0.0, wt = 0.0;
    for (int t = z0; t < z1; t++) {
        int z = s->vav_zone_idx[t];
        area = __dadd_rn(area, s->zone_area[z]);
        wt = __dadd_rn(wt, __dmul_rn(s->avg_temp[z], s->zone_area[z]));
    }
    s->vav_area[v] = area;
    s->sensing[v] = (area > 0.0) ? wt / area : 0.0;
}

/* Apply the producer's permutation to every env: read all sources, then write
 * all destinations (the two phases cannot be one kernel: a cell can be both).
 * `shared` = every env uses stream 0's list (identical generator streams). */
/* The shared-list (one producer for the batch) permutation apply on the [cell][ld] field: thread =
 * (list entry, env), envs fastest, so both the gather and the scatter move whole cell rows. */
__global__ void conv_gather_sh_k(const double* temp, int ld, int nenv, const int* src, const int* count,
                                 double* scr) {
    const size_t tot = (size_t)count[0] * ld;
    for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < tot; x += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(x / ld), e = (int)(x % ld);
        if (e < nenv) scr[x] = temp[(size_t)src[i] * ld + e];
    }
}
__global__ void conv_scatter_sh_k(double* temp, int ld, int nenv, const int* dst, const int* count,
                                  const double* scr) {
    const size_t tot = (size_t)count[0] * ld;
    for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < tot; x += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(x / ld), e = (int)(x % ld);
        if (e < nenv) temp[(size_t)dst[i] * ld + e] = scr[x];
    }
}
__global__ void conv_gather_k(const double* temp, int ncv, const int* src, const int* count,
                              int list_cap, int shared, double* scr, int scr_stride) {
    const int e = blockIdx.y;
    const int ls = shared ? 0 : e;
    const int cnt = count[ls];
    const int* l = src + (size_t)ls * list_cap;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += gridDim.x * blockDim.x)
        scr[(size_t)e * scr_stride + i] = temp[(size_t)l[i] * ncv + e];   /* ncv = field stride (ld) */
}
__global__ void conv_scatter_k(double* temp, int ncv, const int* dst, const int* count,
                               int list_cap, int shared, const double* scr, int scr_stride) {
    const int e = blockIdx.y;
    const int ls = shared ? 0 : e;
    const int cnt = count[ls];
    const int* l = dst + (size_t)ls * list_cap;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += gridDim.x * blockDim.x)
        temp[(size_t)l[i] * ncv + e] = scr[(size_t)e * scr_stride + i];
}

/* ---- reward: exact port of sb_reward --------------------------------- */
/* The field mean is only reached when an AHU has no VAV area (the C computes
 * it unconditionally); it is reduced in parallel, so the value there differs
 * from the CPU's sequential sum at round-off, which is measured and reported. */
__global__ void tmean_k(GpuSim* sims, double* out) {
    int e = blockIdx.x;
    GpuSim* s = &sims[e];
    extern __shared__ double sbreduce[];
    double acc = 0.0;
    const double* T = s->temp;
    for (int p = threadIdx.x; p < s->n_cvs; p += blockDim.x) acc += T[(size_t)p * s->ts];
    sbreduce[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
        if (threadIdx.x < w) sbreduce[threadIdx.x] += sbreduce[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[e] = sbreduce[0] / (double)s->n_cvs;
}

__global__ void reward_k(GpuSim* sims, const double* tmean, const double* occV,
                         int nenv, const int* stepv, double* phi, float* shape,
                         float shape_k, float shape_gamma) {
    int e = blockIdx.x;
    if (threadIdx.x) return;
    GpuSim* s = &sims[e];
    int kv = stepv[e];
    int nz = s->n_zones;
    double dt = s->dt_sec;
    double inv_hour = 1.0 / 3600.0;
    const double* occ = occV + (size_t)e * nz;

    for (int z = 0; z < nz; z++) {
        double fsp = 0.0, fact = 0.0;
        for (int t = s->zone_vav_off[z]; t < s->zone_vav_off[z + 1]; t++) {
            int v = s->zone_vav_idx[t];
            double va = s->vav_area[v];
            if (va <= 0.0) continue;
            double share = s->zone_area[z] / va;
            fsp += s->vav_const[v * 9] * share;
            fact += d_vav_flow_demand(s, v) * share;
        }
        s->zo_sp[z] = fsp;
        s->zo_act[z] = fact;
    }

    double ambient = s->amb[kv];
    int hour = (int)s->rew_hour[kv];
    int month = (int)s->rew_month[kv];
    int wd = (int)s->rew_wd[kv];

    for (int a2 = 0; a2 < s->n_ahu; a2++) { s->ahu_wsum[a2] = 0.0; s->ahu_atotal[a2] = 0.0; }
    for (int v = 0; v < s->n_vav; v++) {
        int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
        double area = 0.0, wt = 0.0;
        for (int t = z0; t < z1; t++) {
            int z = s->vav_zone_idx[t];
            area += s->zone_area[z];
            wt += s->avg_temp[z] * s->zone_area[z];
        }
        if (area > 0.0) {
            int a = s->vav_ahu_idx[v];
            s->ahu_wsum[a] += (wt / area) * area;
            s->ahu_atotal[a] += area;
        }
    }
    double intake = 0.0, exhaust = 0.0, cooling = 0.0;
    for (int a = 0; a < s->n_ahu; a++) {
        double recirc = (s->ahu_atotal[a] > 0.0) ? (s->ahu_wsum[a] / s->ahu_atotal[a])
                                                 : tmean[e];
        double stat = d_ahu_static(s, a);
        double fe = s->ahu_const[a * 9 + 5];
        double fl = (s->ahu_run[a] == ON) ? s->ahu_flow[a] : 0.0;
        intake += fl * stat / fe;
        exhaust += fl * (1.0 - s->ahu_recirc[a]) * stat / fe;
        double mixed = s->ahu_recirc[a] * recirc + (1.0 - s->ahu_recirc[a]) * ambient;
        double sat = (mixed > s->ahu_sat[a] && s->ahu_run[a] == ON) ? s->ahu_sat[a] : mixed;
        cooling += fl * RHO_AIR * CP_AIR * (sat - mixed) / s->ahu_const[a * 9 + 4];
    }
    double blower = intake + exhaust;

    double tf = d_total_flow_rate(s);
    const double* c = s->boiler_c;
    double rwt = s->boiler_return;
    double supply = (s->boiler_setpoint > rwt) ? s->boiler_setpoint : rwt;
    double gflow = CP_W * RHO_W * tf * (supply - rwt);
    double diss = 0.0;
    if (supply >= ambient) {
        double num = c[5] * 2.0 * 3.14159265358979323846 * (supply - ambient);
        double r2 = c[6] + c[9];
        double cond = log(r2 / c[6]) / c[8];
        double conv = 1.0 / c[4] / r2;
        diss = num / (cond + conv);
    }
    double tank = 0.0;
    if (dt > 0.0) tank = CP_W * RHO_W * c[7] * s->boiler_step_change / dt;
    double tot = gflow + diss + tank;
    double gas_rate = tot > 0.0 ? tot : 0.0;
    double pump_e = tf * RHO_W * GRAV * s->pump_head / s->pump_eff;

    double max_pph = s->rw_scalars[0], min_pph = s->rw_scalars[1];
    double ref = s->rw_scalars[2];
    double total_occ = 0.0, cum_prod = 0.0;
    double dsum = 0.0, dmax = 0.0, ssum = 0.0;
    int n_oob = 0;
    double hsp = s->sched_heat_sp[kv], csp = s->sched_cool_sp[kv];
    for (int z = 0; z < nz; z++) {
        double occz = occ[z];
        total_occ += occz;
        double dev = s->avg_temp[z] - csp;
        if (hsp - s->avg_temp[z] > dev) dev = hsp - s->avg_temp[z];
        if (dev < 0.0) dev = 0.0;
        dsum += dev;
        if (dev > dmax) dmax = dev;
        if (dev > 0.0) n_oob++;
        double short_ = (dev <= ref) ? dev / ref
                        : 1.0 + ((dev - ref) / ref) * ((dev - ref) / ref);
        cum_prod += max_pph * (1.0 - short_) * occz * dt * inv_hour;
        if (shape != NULL) ssum += short_;   /* occupancy-free shortfall (env.comfort_shape) */
    }
    double max_prod = max_pph * total_occ * dt * inv_hour;
    double min_prod = min_pph * total_occ * dt * inv_hour;
    double npr;
    if (total_occ > 0.0 && (max_prod - min_prod) != 0.0)
        npr = (cum_prod - min_prod) / (max_prod - min_prod) - 1.0;
    else
        npr = 0.0;

    double elec_rate = blower + fabs(cooling) + pump_e;
    double price = (wd ? s->elec_wd[hour] : s->elec_we[hour]);
    double crate = s->elec_c[hour];
    double elec_cost = price * fabs(elec_rate) * dt;
    double elec_cost_max = price * s->rw_scalars[3] * dt;
    double elec_carb = crate * fabs(elec_rate) * dt;
    double elec_carb_max = crate * s->rw_scalars[3] * dt;
    double gprice = s->gas_p[month - 1];
    double g = gas_rate > 0.0 ? gas_rate : 0.0;
    double gas_cost = gprice * g * dt;
    double gas_cost_max = gprice * s->rw_scalars[4] * dt;
    double gas_carb = s->gas_carb_rate * g * dt;
    double gas_carb_max = s->gas_carb_rate * s->rw_scalars[4] * dt;

    double den_e = elec_cost_max + gas_cost_max;
    double nec = (den_e > 0.0) ? (elec_cost + gas_cost) / den_e : 0.0;
    double den_c = elec_carb_max + gas_carb_max;
    double ncar = (den_c > 0.0) ? (elec_carb + gas_carb) / den_c : 0.0;

    double pw = s->rw_scalars[5], ew = s->rw_scalars[6], cw = s->rw_scalars[7];
    double raw = npr * pw - nec * ew - ncar * cw;
    s->rew_out[0] = raw / (pw + ew + cw);
    /* env.comfort_shape: the train-only potential-based bonus for THIS step, F_t = gamma*Phi_t -
     * Phi_{t-1}, with Phi = -K*mean_z shortfall_z (occupancy-free).  The official reward above is
     * untouched (rew_out/rparts feed the eval dump and the live metrics); finalize_k adds this to
     * the trainer's reward only.  kv == 1 is the first transition of an episode (no previous
     * state); phi holds NaN until the first step runs, so both guards skip exactly that step. */
    if (shape != NULL) {
        double q = nz > 0 ? ssum / nz : 0.0;
        double prev = phi[e];
        shape[e] = (float)((kv > 1 && prev >= 0.0) ? shape_k * (prev - shape_gamma * q) : 0.0);
        phi[e] = q;
    }
    double* rp = s->rparts;
    rp[0] = npr;               /* normalized productivity term (before weight) */
    rp[1] = nec;               /* normalized energy cost */
    rp[2] = ncar;              /* normalized carbon */
    rp[3] = nz > 0 ? dsum / nz : 0.0;           /* mean zone comfort deviation, K */
    rp[4] = dmax;                               /* worst zone deviation, K */
    rp[5] = nz > 0 ? (double)n_oob / nz : 0.0;  /* fraction of zones outside the band */
    rp[6] = fabs(elec_rate) * dt / 3.6e6;       /* electricity, kWh this step */
    rp[7] = g * dt / 3.6e6;                     /* gas, kWh this step */
}

/* ---- observation: exact port of sb_obs ------------------------------- */
__device__ __forceinline__ double d_obs_value(GpuSim* s, int kind, int idx, int id,
                                              int j) {
    if (kind == 0) {
        int a = idx;
        const double* c = &s->ahu_const[a * 9];
        switch (id) {
            case 0: return (double)s->ahu_coolreq[a];
            case 1: return d_ahu_static(s, a);
            case 2: return (s->ahu_run[a] == ON) ? s->ahu_flow[a] : 0.0;
            case 3: return s->ahu_sat[a];
            case 4: return (double)s->ahu_run[a];
            case 5: return s->ahu_flow[a] / c[1];
            case 6: return (1.0 - s->ahu_recirc[a]) * s->ahu_flow[a];
            case 7: return (double)s->ahu_run[a];
            case 8: return s->ahu_recirc[a];
            default: return s->amb[j];
        }
    } else if (kind == 1) {
        int v = idx;
        switch (id) {
            case 0: return s->vav_reheat[v];
            case 1: return s->vav_damper[v];
            case 2: return s->vav_const[v * 9];
            case 3: return s->vav_coolsp[v];
            case 4: return s->vav_heatsp[v];
            default: return s->sensing[v];
        }
    }
    double pdp = (s->pump_run == ON) ? GRAV * RHO_W * s->pump_head / PA_PER_BAR : 0.0;
    switch (id) {
        case 0: return pdp;
        case 1: return s->boiler_current;
        case 2: return pdp;
        case 3: return (double)s->hws_run;
        case 4: return (double)s->hws_heating_requests;
        case 5: return (double)s->hws_run;
        default: return s->boiler_setpoint;
    }
}

__global__ void obs_dev_k(GpuSim* sims, int nenv, const int* jv, float* out,
                          const int* guard) {
    if (guard != NULL && guard[0] == 0) return;
    int nd = sims[0].n_dev;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * nd) return;
    int e = idx / nd, i = idx % nd;
    if (guard != NULL && guard[1 + e] == 0) return;
    int j = jv[e];
    GpuSim* s = &sims[e];
    double v = d_obs_value(s, s->f_kind[i], s->f_idx[i], s->f_obs_id[i], j);
    float v32 = (float)v;
    double var = s->f_var[i];
    out[(size_t)e * s->obs_len + i] =
        (float)((var > 0.0) ? ((double)v32 - s->f_mean[i]) / sqrt(var) : 0.0);
}

__global__ void obs_aux_k(GpuSim* sims, const double* occV, int nenv, const int* jv,
                          float* out, const int* guard) {
    if (guard != NULL && guard[0] == 0) return;
    int na = sims[0].n_aux;
    if (na <= 0) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nenv * na) return;
    int e = idx / na, k = idx % na;
    if (guard != NULL && guard[1 + e] == 0) return;
    GpuSim* s = &sims[e];
    int j = jv[e];
    float val;
    if (k < na - 1) {
        val = (float)s->aux_time[(size_t)j * (na - 1) + k];
    } else {
        double occ = 0.0;
        for (int z = 0; z < s->n_zones; z++) occ += occV[(size_t)e * s->n_zones + z];
        val = (float)(int)occ;
    }
    out[(size_t)e * s->obs_len + s->n_dev + k] = val;
}

#define NE 64

static double* g_keep_apc = nullptr;
static double* g_keep_yrk = nullptr;
}  /* namespace sbgpu */

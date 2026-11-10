/* M5 device occupancy + convection, gated against csim bit-for-bit.
 *
 * Both consume a SINGLE MT19937 stream per environment in a fixed order, so
 * they are inherently sequential *within* an env and are parallelised *across*
 * envs: one block per env, one thread doing the chain. That is the same
 * batching axis the solves use.
 * usage: sb_gpu_rng <blob.sbb> [nsteps] [nenv]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

extern "C" {
#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_occup.h"
}

#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while (0)

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfu
#define MT_UPPER_MASK 0x80000000u
#define MT_LOWER_MASK 0x7fffffffu

struct GpuRng { unsigned int mt[MT_N]; unsigned int idx; };

__device__ void grng_seed(GpuRng* r, unsigned int seed) {
    r->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        unsigned int prev = r->mt[i - 1];
        r->mt[i] = 1812433253u * (prev ^ (prev >> 30)) + (unsigned int)i;
    }
    r->idx = MT_N;
}

__device__ void grng_generate(GpuRng* r) {
    for (int i = 0; i < MT_N; i++) {
        unsigned int y = (r->mt[i] & MT_UPPER_MASK)
                       | (r->mt[(i + 1) % MT_N] & MT_LOWER_MASK);
        r->mt[i] = r->mt[(i + MT_M) % MT_N] ^ (y >> 1)
                 ^ ((y & 1u) ? MT_MATRIX_A : 0u);
    }
    r->idx = 0;
}

__device__ unsigned int grng_u32(GpuRng* r) {
    if (r->idx >= MT_N) grng_generate(r);
    unsigned int y = r->mt[r->idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

__device__ double grng_double(GpuRng* r) {
    unsigned int a = grng_u32(r) >> 5;
    unsigned int b = grng_u32(r) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

__device__ unsigned int grng_interval(GpuRng* r, unsigned int max) {
    if (max == 0) return 0;
    unsigned int mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    unsigned int v = grng_u32(r) & mask;
    while (v > max) v = grng_u32(r) & mask;
    return v;
}

/* occupancy: one thread per env */
__global__ void occup_k(const long long* hour_tab, const unsigned char* wd_tab,
                        const int* step_of_env, int n_zones, int nassign,
                        int ea, int la, int ed, int ld, double p_arr, double p_dep,
                        int* state, double* V, GpuRng* rngs) {
    if (threadIdx.x) return;
    int e = blockIdx.x;
    GpuRng r = rngs[e];
    int step = step_of_env[e];
    int hour = (int)hour_tab[step];
    int wd = (wd_tab[step] != 0);
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

/* convection: one thread per env, exact port of sb_conv_apply */
__global__ void convect_k(const int* room_off, const int* room_cells,
                          const int* cand_off, const int* cand_idx,
                          int n_rooms, int nc, int ncv, int max_room, double p,
                          int distance, double* temp, int* left, int* right,
                          GpuRng* rngs) {
    if (threadIdx.x) return;
    int e = blockIdx.x;
    if (p == 0.0 || distance == 0) return;
    GpuRng r = rngs[e];
    double* T = temp + (size_t)e * ncv;   /* this env's field */
    int* L = left + (size_t)e * max_room;
    int* R = right + (size_t)e * max_room;
    for (int rm = 0; rm < n_rooms; rm++) {
        int base = room_off[rm];
        int n = room_off[rm + 1] - base;
        if (n <= 0) continue;
        int m = 0;
        for (int j = 0; j < n; j++) {
            if (grng_double(&r) > p) continue;
            int cg = base + j;
            int start = cand_off[cg], end = cand_off[cg + 1];
            if (end <= start) continue;
            int jj = cand_idx[start + (int)grng_interval(&r, (unsigned)(end - start - 1))];
            L[m] = cg; R[m] = jj; m++;
        }
        for (int t = m - 1; t > 0; t--) {
            int j2 = (int)grng_interval(&r, (unsigned)t);
            int tmp = L[t]; L[t] = L[j2]; L[j2] = tmp;
            tmp = R[t]; R[t] = R[j2]; R[j2] = tmp;
        }
        for (int t = 0; t < m; t++) {
            int a = L[t], bb = R[t];
            int ia = room_cells[2 * a] * nc + room_cells[2 * a + 1];
            int ib = room_cells[2 * bb] * nc + room_cells[2 * bb + 1];
            double held = T[ia]; T[ia] = T[ib]; T[ib] = held;
        }
    }
    rngs[e] = r;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc < 2) { printf("usage: %s <blob.sbb> [nsteps] [nenv]\n", argv[0]); return 2; }
    int nsteps = argc > 2 ? atoi(argv[2]) : 10;
    int nenv = argc > 3 ? atoi(argv[3]) : 1;
    SbBlob blob;
    if (sb_open(&blob, argv[1]) != 0) return 1;

    /* ---- shared scenario data ---- */
    static SbOccup occ0;
    static SbConv conv0;
    if (sb_occup_init(&occ0, &blob) || sb_conv_init(&conv0, &blob)) {
        printf("init FAIL\n"); return 1;
    }
    const int n_zones = occ0.n_zones, nassign = occ0.zone_assignment;
    const int n_rooms = conv0.n_rooms, nc = conv0.nc;
    const int ncv = conv0.n_cvs;
    const int nsteps_tab = (int)sb_find(&blob, "occ_hour")->count;
    printf("blob: zones=%d assign=%d conv rooms=%d nc=%d cells=%d tab=%d\n",
           n_zones, nassign, n_rooms, nc, ncv, nsteps_tab);
    /* max room size for the swap scratch */
    int max_room = 1;
    for (int i = 0; i < n_rooms; i++) {
        int n = sb_i32(&blob, "conv_room_offsets")[i + 1] - sb_i32(&blob, "conv_room_offsets")[i];
        if (n > max_room) max_room = n;
    }
    printf("max room cells=%d\n", max_room);

    /* ---- GPU buffers ---- */
    long long* d_hour; unsigned char* d_wd;
    int *d_state, *d_left, *d_right, *d_step;
    int *d_ro, *d_rc, *d_co, *d_ci;
    double *d_V, *d_temp;
    GpuRng* d_rng_occ; GpuRng* d_rng_conv;
    CU(cudaMalloc(&d_hour, (size_t)nsteps_tab * 8));
    CU(cudaMemcpy(d_hour, sb_i64(&blob, "occ_hour"), (size_t)nsteps_tab * 8, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_wd, (size_t)nsteps_tab));
    CU(cudaMemcpy(d_wd, sb_u8(&blob, "occ_workday"), (size_t)nsteps_tab, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_state, (size_t)nenv * n_zones * nassign * 4));
    CU(cudaMemset(d_state, 0, (size_t)nenv * n_zones * nassign * 4));
    CU(cudaMalloc(&d_V, (size_t)nenv * n_zones * 8));
    CU(cudaMalloc(&d_step, (size_t)nenv * 4));
    CU(cudaMalloc(&d_rng_occ, sizeof(GpuRng) * nenv));
    CU(cudaMalloc(&d_rng_conv, sizeof(GpuRng) * nenv));
    CU(cudaMalloc(&d_temp, (size_t)nenv * ncv * 8));
    /* the swap scratch must hold every selected cell of the LARGEST room */
    CU(cudaMalloc(&d_left, (size_t)nenv * max_room * 4));
    CU(cudaMalloc(&d_right, (size_t)nenv * max_room * 4));
    int nro = (int)sb_find(&blob, "conv_room_offsets")->count;
    int nrc = (int)sb_find(&blob, "conv_room_cells")->count;
    int nco = (int)sb_find(&blob, "conv_cand_offsets")->count;
    int nci = (int)sb_find(&blob, "conv_cand_idx")->count;
    CU(cudaMalloc(&d_ro, (size_t)nro * 4));
    CU(cudaMemcpy(d_ro, sb_i32(&blob, "conv_room_offsets"), (size_t)nro * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_rc, (size_t)nrc * 4));
    CU(cudaMemcpy(d_rc, sb_i32(&blob, "conv_room_cells"), (size_t)nrc * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_co, (size_t)nco * 4));
    CU(cudaMemcpy(d_co, sb_i32(&blob, "conv_cand_offsets"), (size_t)nco * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_ci, (size_t)nci * 4));
    CU(cudaMemcpy(d_ci, sb_i32(&blob, "conv_cand_idx"), (size_t)nci * 4, cudaMemcpyHostToDevice));

    /* seed the device RNGs exactly as the CPU does (per env, same seeds) */
    {
        std::vector<GpuRng> ho(nenv), hc(nenv);
        uint32_t so = (uint32_t)sb_i64_1(&blob, "occ_seed", 17321);
        uint32_t sc = (uint32_t)sb_i64_1(&blob, "conv_seed", 0);
        for (int e = 0; e < nenv; e++) {
            /* reuse the CPU seeding code by running it into a temp struct */
            SbRng t; sb_rng_seed(&t, so);
            memcpy(ho[e].mt, t.mt, sizeof(t.mt)); ho[e].idx = t.idx;
            sb_rng_seed(&t, sc);
            memcpy(hc[e].mt, t.mt, sizeof(t.mt)); hc[e].idx = t.idx;
        }
        CU(cudaMemcpy(d_rng_occ, ho.data(), sizeof(GpuRng) * nenv, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_rng_conv, hc.data(), sizeof(GpuRng) * nenv, cudaMemcpyHostToDevice));
        std::vector<int> hs(nenv);
        for (int e = 0; e < nenv; e++) hs[e] = 0;
        CU(cudaMemcpy(d_step, hs.data(), nenv * 4, cudaMemcpyHostToDevice));
    }
    /* initial temperature = the blob's initial field, replicated per env */
    {
        std::vector<double> t0(ncv);
        const double* ti = sb_f64(&blob, "temp_init");
        for (int p = 0; p < ncv; p++) t0[p] = ti[p];
        for (int e = 0; e < nenv; e++)
            CU(cudaMemcpy(d_temp + (size_t)e * ncv, t0.data(), (size_t)ncv * 8, cudaMemcpyHostToDevice));
    }

    /* ---- CPU reference: independent copies, same seeds ---- */
    std::vector<SbOccup> occ(nenv);
    std::vector<SbConv> conv(nenv);
    std::vector<std::vector<double>> cpu_temp(nenv);
    for (int e = 0; e < nenv; e++) {
        sb_occup_init(&occ[e], &blob);
        sb_conv_init(&conv[e], &blob);
        cpu_temp[e].resize(ncv);
        const double* ti = sb_f64(&blob, "temp_init");
        for (int p = 0; p < ncv; p++) cpu_temp[e][p] = ti[p];
    }

    double worstV = 0.0, worstT = 0.0;
    double p_arr = occ0.p_arrival, p_dep = occ0.p_departure;
    for (int st = 0; st < nsteps; st++) {
        /* device: occupancy for this step, then convection */
        CU(cudaMemcpy(d_step, std::vector<int>(nenv, st).data(), nenv * 4, cudaMemcpyHostToDevice));
        occup_k<<<nenv, 32>>>(d_hour, d_wd, d_step, n_zones, nassign, occ0.ea,
                              occ0.la, occ0.ed, occ0.ld, p_arr, p_dep, d_state,
                              d_V, d_rng_occ);
        CU(cudaDeviceSynchronize());
        convect_k<<<nenv, 32>>>(d_ro, d_rc, d_co, d_ci, n_rooms, nc, ncv, max_room,
                                conv0.p, conv0.distance, d_temp, d_left, d_right,
                                d_rng_conv);
        CU(cudaDeviceSynchronize());

        /* CPU: same step */
        for (int e = 0; e < nenv; e++) {
            sb_occup_advance(&occ[e], st);
            sb_conv_apply(&conv[e], cpu_temp[e].data());
        }
        std::vector<double> hV((size_t)nenv * n_zones);
        std::vector<double> hT((size_t)nenv * ncv);
        CU(cudaMemcpy(hV.data(), d_V, (size_t)nenv * n_zones * 8, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(hT.data(), d_temp, (size_t)nenv * ncv * 8, cudaMemcpyDeviceToHost));
        double eV = 0.0, eT = 0.0;
        for (size_t i = 0; i < hV.size(); i++)
            eV = fmax(eV, fabs(hV[i] - occ[i / n_zones].V[i % n_zones]));
        for (size_t e = 0; e < (size_t)nenv; e++)
            for (int p = 0; p < ncv; p++)
                eT = fmax(eT, fabs(hT[e * ncv + p] - cpu_temp[e][p]));
        worstV = fmax(worstV, eV);
        worstT = fmax(worstT, eT);
        if (st < 3 || eV > 0 || eT > 0)
            printf("  step %2d: occupancy err=%.3e temp err=%.3e\n", st, eV, eT);
    }
    printf("worst: occupancy=%.3e temp=%.3e (nenv=%d steps=%d)\n", worstV, worstT,
           nenv, nsteps);
    int bad = (worstV != 0.0) || (worstT != 0.0);
    printf(bad ? "GPU RNG/OCC/CONV: FAIL\n" : "GPU RNG/OCC/CONV: OK (bit-exact)\n");
    return bad;
}

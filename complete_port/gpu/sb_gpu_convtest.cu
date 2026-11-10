/* Parity gate for the device convection (convect2_k, and the original convect_k
 * pipeline) against the verified CPU core's sb_conv_apply, bit for bit.
 *   usage: sb_gpu_convtest <blob.sbb> <steps> <p> [nenv] [old]
 * Every step refills the field with fresh distinct values, applies the CPU
 * convection once per env (each env owns an RNG stream seeded differently) and
 * the device convection once for all envs, then compares the field (memcmp) and
 * the generator state after the step.  `p` overrides the blob's probability so
 * event-dense and stream-boundary cases are exercised. */
#include "sb_gpu_kernels.cuh"
#include "sb_convect.h"
#include <vector>
using namespace sbgpu;

static unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <blob> <steps> <p> [nenv] [old]\n", argv[0]); return 2; }
    const int steps = atoi(argv[2]);
    const double p = atof(argv[3]);
    const int nenv = argc > 4 ? atoi(argv[4]) : 4;
    const bool old = argc > 5;
    SbBlob blob;
    if (sb_open(&blob, argv[1]) != 0) return 2;
    std::vector<SbConv> cc(nenv);
    for (int e = 0; e < nenv; e++) {
        if (sb_conv_init(&cc[e], &blob) != 0) return 2;
        cc[e].p = p;
        sb_rng_seed(&cc[e].rng, 1000u + 7u * (unsigned)e);
    }
    const int nc = cc[0].nc, ncv = cc[0].n_cvs, nrooms = cc[0].n_rooms;
    const int ncell = (int)sb_find(&blob, "conv_cand_offsets")->count - 1;
    int max_room = 1;
    for (int r = 0; r < nrooms; r++) max_room = std::max(max_room, cc[0].room_off[r + 1] - cc[0].room_off[r]);
    int *d_roo, *d_rcc, *d_coo, *d_cii;
    CU(cudaMalloc(&d_roo, 4 * (nrooms + 1)));
    CU(cudaMemcpy(d_roo, cc[0].room_off, 4 * (nrooms + 1), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_rcc, 8 * (size_t)ncell));
    CU(cudaMemcpy(d_rcc, cc[0].room_cells, 8 * (size_t)ncell, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_coo, 4 * (size_t)(ncell + 1)));
    CU(cudaMemcpy(d_coo, cc[0].cand_off, 4 * (size_t)(ncell + 1), cudaMemcpyHostToDevice));
    const int nci = cc[0].cand_off[ncell];
    CU(cudaMalloc(&d_cii, 4 * (size_t)nci));
    CU(cudaMemcpy(d_cii, cc[0].cand_idx, 4 * (size_t)nci, cudaMemcpyHostToDevice));
    std::vector<GpuRng> hr(nenv);
    for (int e = 0; e < nenv; e++) {
        memcpy(hr[e].mt, cc[e].rng.mt, sizeof(hr[e].mt));
        hr[e].idx = (unsigned)cc[e].rng.idx;
    }
    GpuRng* d_rng; CU(cudaMalloc(&d_rng, sizeof(GpuRng) * nenv));
    CU(cudaMemcpy(d_rng, hr.data(), sizeof(GpuRng) * nenv, cudaMemcpyHostToDevice));
    double *d_T, *d_scr; CU(cudaMalloc(&d_T, 8 * (size_t)nenv * ncv));
    const int cap = 2 * ncell;
    int *d_ld, *d_ls, *d_lc;
    CU(cudaMalloc(&d_ld, 4 * (size_t)nenv * cap)); CU(cudaMalloc(&d_ls, 4 * (size_t)nenv * cap));
    CU(cudaMalloc(&d_lc, 4 * nenv));
    CU(cudaMalloc(&d_scr, 8 * (size_t)nenv * cap));
    int *d_l, *d_r, *d_cons; unsigned int* d_w;
    const int nwords = 6 * ncv;
    CU(cudaMalloc(&d_l, 4 * (size_t)nenv * max_room));
    CU(cudaMalloc(&d_r, 4 * (size_t)nenv * max_room));
    CU(cudaMalloc(&d_cons, 4 * nenv));
    CU(cudaMalloc(&d_w, 4 * (size_t)nenv * nwords));
    const size_t smem2 = (size_t)(CV_RING + 4 + 2 * CV_BW + 2 * MT_N + max_room) * 4 + (size_t)2 * ((max_room + 1) & ~1);
    const size_t smem1 = (size_t)max_room * 8;
    const size_t smem2g = (size_t)(CV_RING + 4 + 2 * CV_BW + 2 * MT_N) * 4 + (size_t)2 * ((max_room + 1) & ~1);
    unsigned int* d_lrg; unsigned short* d_pg;
    CU(cudaMalloc(&d_lrg, 4 * (size_t)nenv * max_room)); CU(cudaMalloc(&d_pg, 2 * (size_t)nenv * max_room));
    CU(cudaFuncSetAttribute(convect2g_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2g));
    if (smem2 <= 99 * 1024) CU(cudaFuncSetAttribute(convect2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2));   /* rooms too big: use CV_GLOBAL=1 */
    if (smem1 <= 99 * 1024) CU(cudaFuncSetAttribute(convect_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1));
    std::vector<double> hT((size_t)nenv * ncv), gT((size_t)nenv * ncv);
    long long bad = 0, badrng = 0, words = 0;
    for (int st = 0; st < steps; st++) {
        for (int e = 0; e < nenv; e++)
            for (int i = 0; i < ncv; i++)
                hT[(size_t)e * ncv + i] =
                    (double)(mix64(((unsigned long long)st << 40) ^ ((unsigned long long)e << 32) ^ (unsigned)i) >> 11) / 9007199254740992.0 + 250.0;
        /* the env keeps the field [cell][env] (stride = nenv here) */
        for (int e = 0; e < nenv; e++)
            for (int i = 0; i < ncv; i++) gT[(size_t)i * nenv + e] = hT[(size_t)e * ncv + i];
        CU(cudaMemcpy(d_T, gT.data(), 8 * (size_t)nenv * ncv, cudaMemcpyHostToDevice));
        for (int e = 0; e < nenv; e++) sb_conv_apply(&cc[e], &hT[(size_t)e * ncv]);
        if (!old) {
            if (getenv("CV_GLOBAL"))
                convect2g_k<<<nenv, 256, smem2g>>>(d_roo, d_rcc, d_coo, d_cii, nrooms, nc, max_room, p,
                                                   2, d_rng, d_ld, d_ls, d_lc, cap, d_lrg, d_pg);
            else
                convect2_k<<<nenv, 256, smem2>>>(d_roo, d_rcc, d_coo, d_cii, nrooms, nc, max_room, p,
                                                 2, d_rng, d_ld, d_ls, d_lc, cap, NULL, NULL);
            conv_gather_k<<<dim3(32, nenv), 256>>>(d_T, nenv, d_ls, d_lc, cap, 0, d_scr, cap);
            conv_scatter_k<<<dim3(32, nenv), 256>>>(d_T, nenv, d_ld, d_lc, cap, 0, d_scr, cap);
        } else {
            CU(cudaMemset(d_cons, 0, 4 * nenv));
            conv_gen_k<<<nenv, 256>>>(d_rng, d_w, nwords);
            convect_k<<<nenv, 256, smem1>>>(d_roo, d_rcc, d_coo, d_cii, nrooms, nc, ncv, max_room, p,
                                            2, d_T, d_l, d_r, d_w, nwords, d_cons, nenv);
            conv_advance_k<<<nenv, 256>>>(d_rng, d_cons);
        }
        CU(cudaDeviceSynchronize());
        {
            std::vector<double> cm((size_t)nenv * ncv);
            CU(cudaMemcpy(cm.data(), d_T, 8 * cm.size(), cudaMemcpyDeviceToHost));
            for (int e = 0; e < nenv; e++)
                for (int i = 0; i < ncv; i++) gT[(size_t)e * ncv + i] = cm[(size_t)i * nenv + e];
        }
        CU(cudaMemcpy(hr.data(), d_rng, sizeof(GpuRng) * nenv, cudaMemcpyDeviceToHost));
        for (int e = 0; e < nenv; e++) {
            if (memcmp(&hT[(size_t)e * ncv], &gT[(size_t)e * ncv], 8 * (size_t)ncv) != 0) {
                if (bad < 3) printf("FIELD MISMATCH step %d env %d\n", st, e);
                bad++;
            }
            bool same = (hr[e].idx == (unsigned)cc[e].rng.idx) &&
                        memcmp(hr[e].mt, cc[e].rng.mt, sizeof(hr[e].mt)) == 0;
            if (!same) {
                if (badrng < 3) printf("RNG STATE MISMATCH step %d env %d (gpu idx %u cpu idx %d)\n", st, e, hr[e].idx, cc[e].rng.idx);
                badrng++;
            }
        }
        words += 1;
    }
    printf("convtest p=%.3f nenv=%d steps=%d %s: field mismatches=%lld rng mismatches=%lld -> %s\n",
           p, nenv, steps, old ? "convect_k(old)" : "convect2_k", bad, badrng,
           (bad || badrng) ? "FAIL" : "OK");
    return (bad || badrng) ? 1 : 0;
}

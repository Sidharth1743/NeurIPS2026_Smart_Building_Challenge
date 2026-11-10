/* Device replay of the baked SuperLU factorization (the solve of csim/sb_lu.c),
 * batched over envs, as a synchronisation-free sparse triangular solve.
 *
 * The CPU core solves  y = b[ir];  L z = y (unit lower, CSC push);  U w = z (upper,
 * CSC push);  x = w[pc].  A push update  work[i] -= l_ij * work[j]  visits, for a
 * fixed row i, the columns j in ascending order (forward) / descending order
 * (backward), so the same additions can be done row-by-row as a pull:
 *     w_i = b_i;  for j in the row's pattern, in that order:  w_i = w_i - l_ij * w_j
 * with the multiply and the subtraction rounded separately (no fma), which makes
 * every value BIT-IDENTICAL to sb_lu_solve, and independent of scheduling.
 *
 * Layout: work[row][ld] (forward result y) and zout[row][ld] (backward result z, kept apart
 * because a forward row may still be reading y_j when backward row j finishes), ld = 32 * RPL envs per row (lane l owns env l + 32 r).
 * One warp owns one row.  Rows are claimed from an atomic ticket counter in
 * dependency-LEVEL order (rows of one level are adjacent, so the rows on the critical path are never
 * queued behind blocked rows of deeper levels) (forward: ascending, then backward: descending, one
 * counter 0..2n-1), a row waits until the rows it reads have published their
 * epoch flag, so a waiting warp only ever waits on warps that already started:
 * deadlock-free.  Flags carry the solve's epoch, so they never need clearing. */
#pragma once
/* Two builds of the same solve.
 *   default:        one fused multiply-add per term (fl(acc - l*x), a single rounding) and a
 *                   multiply by 1/U_ii.  Deterministic.  Measured against the CPU replay of the
 *                   same factors: ~2e-14 absolute on values of order 300, and over a full
 *                   8640-step episode zero differing observation values and a per-step reward
 *                   difference <= 9e-13 (the reference's own reward tolerance is 7.5e-9).
 *                   About 1.2-1.4x faster than exact where the solve is FP64-issue bound.
 *   SB_LU_EXACT:    the multiply and the subtraction are rounded separately and the division
 *                   is a division: bit-identical to csim's sb_lu_solve (two FP64 issue slots
 *                   per term instead of one). */
#ifdef SB_LU_EXACT
#define LU_EXACT
#endif

/* SB_SOLVE_F32 (training-only build: SB_F32=1 tools/build.sh, or SB_F32=1 ./build.sh
 * smart_buildings in the PufferLib checkout.  The macro is SB_SOLVE_F32 because SB_F32 is a blob
 * dtype.)  The solve's state vectors - work (forward result y) and zout (backward result z) - and
 * its arithmetic are fp32 instead of fp64.  The state stream is the solve's whole cost (8 bytes per
 * term per env, twice the FP64 pipe's floor, docs/HISTORY.md "P0/P1"), so halving it halves the
 * step's dominant traffic.  fp32 *storage alone* was measured SLOWER than fp64 (each load needs an
 * F2F conversion on the FP64 pipe: train_1 @256, 23.6 vs 19.3 ms/step), so the accumulator, the
 * row's coefficient and the fma are fp32 as well; the coefficients (lcx/ucx/zcx/uinv), qz and the
 * temperatures stay double, and the per-row arithmetic order is unchanged.  Only rounding changes:
 * a row's sum accumulates in fp32, so a trajectory differs from the exact solve at ~1e-5 relative,
 * bounded and non-diverging.  Training only: the default build is unchanged and remains the one
 * used for evaluation and the submission. */
#ifdef SB_SOLVE_F32
typedef float sb_wt;    /* state vectors: work (y) / zout (z) */
typedef float sb_ws;    /* solve arithmetic: accumulator and per-term operands */
#else
typedef double sb_wt;
typedef double sb_ws;
#endif

struct LuDev {
    int n;
    const int *lrp, *lci;  const double* lcx;      /* L strict-lower CSR, ascending columns   */
    const int *urp, *uci;  const double* ucx;      /* U strict-upper CSR, DESCENDING columns  */
    const double* udiag;                           /* U_ii                                    */
    const double* uinv;                            /* 1 / U_ii (fast mode only)               */
    const int *ordL, *ordU;                        /* rows in dependency-level order (ticket order) */
    /* Optional air-side correction folded into the backward solve (NULL = none): row i of the
     * backward pass starts from  z_i + sum_j Z_ij q_j  (Z = L^-1 P_r applied, CSR over solver rows,
     * ascending j; q laid out [j][ld], one column per env, padding lanes 0). */
    const int* zrp; const int* zci; const double* zcx;
    /* Several weathers of one building (identical sparsity pattern and ordering, different values)
     * in one solve: nw > 1, envw[col] = weather of solver column col (padding columns 0), and the
     * value arrays hold nw consecutive copies (strides nnzL, nnzU, n, nnzZ).  Each lane then loads
     * its own weather's values; the arithmetic per value is the same as with one weather. */
    int nw;
    const int* envw;
    int slab_uniform;          /* 1: every 32-column slab holds one weather (values can be shuffled) */
    long nnzL, nnzU, nnzZ;
};

__device__ __forceinline__ unsigned int ld_cg_flag(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void st_release_flag(unsigned int* p, unsigned int v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ double ld_cg_f64(const double* p) {
    double v;
    asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}
/* the solve state load: fp64 by default, fp32 under SB_SOLVE_F32 (loads scale with the state type, so
 * every state read goes through this one macro) */
#ifdef SB_SOLVE_F32
#define SB_WLD(p) ld_cg_f32(p)
#else
#define SB_WLD(p) ld_cg_f64(p)
#endif

/* LU_EXACT: multiply and subtract rounded separately = bit-identical to sb_lu_solve.
 * Otherwise one fused multiply-add per term (half the FP64 issue slots; deterministic,
 * differs from the CPU replay at round-off only).
 * The fma intrinsic is type-specific: __fma_rn is the double one and silently promotes float
 * arguments, which under SB_SOLVE_F32 would put every term back on the FP64 pipe (measured 2x
 * slower than the fp64 build).  LU_FMA picks the intrinsic of sb_ws. */
#ifdef SB_SOLVE_F32
#define LU_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#else
#define LU_FMA(a, b, c) __fma_rn((a), (b), (c))
#endif
#ifdef LU_EXACT
#define LU_SUB(acc, l, x) __dsub_rn((acc), __dmul_rn((l), (x)))
#else
#define LU_SUB(acc, l, x) LU_FMA(-(l), (x), (acc))
#endif
#ifdef LU_EXACT
#define LU_DIV(acc, i) __ddiv_rn((acc), lu.udiag[(i)])
#else
#define LU_DIV(acc, i) ((acc) * lu.uinv[(i)])
#endif
extern "C++" {   /* templates (the includers wrap this header in extern "C") */
/* One row of the solve: group-local ticket t (< n: forward row ordL[t]; else backward row
 * ordU[t - n]), every lane carrying RPL envs of the row.  wv: this lane's weather per env slot. */
template <int RPL, int LU_CHUNK, int MW>
__device__ __forceinline__ void lu_row(const LuDev& lu, sb_wt* work, sb_wt* zout, int ld,
                                       unsigned int* flagL, unsigned int* flagU, unsigned int epoch,
                                       unsigned long long* ts, const double* qz, unsigned int t,
                                       const int* wv, int lane) {
    const int n = lu.n;
    sb_ws acc[RPL];
    if (t < (unsigned)n) {                                  /* ---- forward, row i ---- */
        const int i = lu.ordL[t];
        sb_wt* wi = work + (size_t)i * ld;
#pragma unroll
        for (int r = 0; r < RPL; r++) acc[r] = wi[lane + 32 * r];
        const int pb = lu.lrp[i], pe = lu.lrp[i + 1];
        for (int p0 = pb; p0 < pe; p0 += LU_CHUNK) {
            const int cn = min(LU_CHUNK, pe - p0);
            int cj = 0; sb_ws cx = 0.0;
            sb_ws cxr[RPL];   /* MW == 2: lane q holds term q's value of slab r's weather */
            if (lane < cn) {
                cj = lu.lci[p0 + lane];
                if (MW == 0) cx = (sb_ws)lu.lcx[p0 + lane];
                if (MW == 2) {
#pragma unroll
                    for (int r = 0; r < RPL; r++) cxr[r] = (sb_ws)lu.lcx[(size_t)wv[r] * lu.nnzL + p0 + lane];
                }
            }
            /* wait until every row of the chunk is published */
            int ready = 0;
            while (!ready) {
                int ok = 1;
                if (lane < cn) ok = (ld_cg_flag(flagL + cj) == epoch);
                ready = __all_sync(0xffffffffu, ok);
            }
            fence_acqrel();
            sb_ws xv[LU_CHUNK][RPL];
#pragma unroll
            for (int q = 0; q < LU_CHUNK; q++) {
                if (q < cn) {
                    const int j = __shfl_sync(0xffffffffu, cj, q);
#pragma unroll
                    for (int r = 0; r < RPL; r++)
                        xv[q][r] = (sb_ws)SB_WLD(work + (size_t)j * ld + lane + 32 * r);
                }
            }
#pragma unroll
            for (int q = 0; q < LU_CHUNK; q++) {
                if (q < cn) {
                    const sb_ws lx = MW ? 0.0 : __shfl_sync(0xffffffffu, cx, q);
#pragma unroll
                    for (int r = 0; r < RPL; r++) {
                        const sb_ws l_ = MW == 1 ? (sb_ws)lu.lcx[wv[r] * lu.nnzL + p0 + q]
                                        : (MW == 2 ? __shfl_sync(0xffffffffu, cxr[r], q) : lx);
                        acc[r] = LU_SUB(acc[r], l_, xv[q][r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPL; r++) wi[lane + 32 * r] = (sb_wt)acc[r];
        fence_acqrel(); __syncwarp();
        if (lane == 0) {
            if (ts) { unsigned long long g_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g_)); ts[i] = g_; }
            st_release_flag(flagL + i, epoch);
        }
    } else {                                                /* ---- backward, row i ---- */
        const int i = lu.ordU[t - (unsigned)n];
        if (ts && lane == 0) { unsigned long long g_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g_)); ts[2 * (size_t)n + i] = g_; }
        sb_wt* wi = work + (size_t)i * ld;
        /* wait for this row's own forward result */
        while (ld_cg_flag(flagL + i) != epoch) { }
        fence_acqrel();
#pragma unroll
        for (int r = 0; r < RPL; r++) acc[r] = (sb_ws)SB_WLD(wi + lane + 32 * r);
        if (qz != NULL)
            for (int p = lu.zrp[i]; p < lu.zrp[i + 1]; p++) {
                const int j = lu.zci[p];
#pragma unroll
                for (int r = 0; r < RPL; r++) {
                    const sb_ws zx = (sb_ws)lu.zcx[(MW ? wv[r] * lu.nnzZ : 0) + p];
                    acc[r] = LU_FMA(zx, (sb_ws)qz[(size_t)j * ld + lane + 32 * r], acc[r]);
                }
            }
        const int pb = lu.urp[i], pe = lu.urp[i + 1];
        for (int p0 = pb; p0 < pe; p0 += LU_CHUNK) {
            const int cn = min(LU_CHUNK, pe - p0);
            int cj = 0; sb_ws cx = 0.0;
            sb_ws cxr[RPL];
            if (lane < cn) {
                cj = lu.uci[p0 + lane];
                if (MW == 0) cx = (sb_ws)lu.ucx[p0 + lane];
                if (MW == 2) {
#pragma unroll
                    for (int r = 0; r < RPL; r++) cxr[r] = (sb_ws)lu.ucx[(size_t)wv[r] * lu.nnzU + p0 + lane];
                }
            }
            int ready = 0;
            while (!ready) {
                int ok = 1;
                if (lane < cn) ok = (ld_cg_flag(flagU + cj) == epoch);
                ready = __all_sync(0xffffffffu, ok);
            }
            fence_acqrel();
            sb_ws xv[LU_CHUNK][RPL];
#pragma unroll
            for (int q = 0; q < LU_CHUNK; q++) {
                if (q < cn) {
                    const int j = __shfl_sync(0xffffffffu, cj, q);
#pragma unroll
                    for (int r = 0; r < RPL; r++)
                        xv[q][r] = (sb_ws)SB_WLD(zout + (size_t)j * ld + lane + 32 * r);
                }
            }
#pragma unroll
            for (int q = 0; q < LU_CHUNK; q++) {
                if (q < cn) {
                    const sb_ws ux = MW ? 0.0 : __shfl_sync(0xffffffffu, cx, q);
#pragma unroll
                    for (int r = 0; r < RPL; r++) {
                        const sb_ws u_ = MW == 1 ? (sb_ws)lu.ucx[wv[r] * lu.nnzU + p0 + q]
                                        : (MW == 2 ? __shfl_sync(0xffffffffu, cxr[r], q) : ux);
                        acc[r] = LU_SUB(acc[r], u_, xv[q][r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPL; r++)
            zout[(size_t)i * ld + lane + 32 * r] = (sb_wt)LU_DIV(acc[r], (MW ? (long)wv[r] * n : 0L) + i);
        fence_acqrel(); __syncwarp();
        if (lane == 0) {
            if (ts) { unsigned long long g_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g_)); ts[n + i] = g_; }
            st_release_flag(flagU + i, epoch);
        }
    }
}
template <int RPL, int LU_CHUNK, int MW>
__global__ void __launch_bounds__(256)
lu_solve_t(LuDev lu, sb_wt* work, sb_wt* zout, int ld, unsigned int* flagL, unsigned int* flagU,
           unsigned int* ticket, const unsigned int* epoch_p, unsigned long long* ts,
           unsigned int tbeg, unsigned int tend, const double* qz) {
    const int lane = threadIdx.x & 31;
    const unsigned int epoch = *(volatile const unsigned int*)epoch_p;
    int wv[RPL];                                   /* this lane's weather per env slot (MW only) */
#pragma unroll
    for (int r = 0; r < RPL; r++) wv[r] = MW ? lu.envw[lane + 32 * r] : 0;
    for (;;) {
        unsigned int t = 0;
        if (lane == 0) t = atomicAdd(ticket, 1u) + tbeg;
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= tend) return;
        lu_row<RPL, LU_CHUNK, MW>(lu, work, zout, ld, flagL, flagU, epoch, ts, qz, t, wv, lane);
    }
}
/* Several groups (scenario groups of one batch, same envs per lane) in ONE persistent solve: the
 * tickets walk a merged order of (group, group-local ticket) sorted by dependency level, so the
 * groups' independent DAGs fill each other's narrow levels.  A row only ever waits on rows of its own
 * group with a lower level, which hold earlier tickets: deadlock-free.  Per-row arithmetic is that of
 * lu_solve_t, so every value is bit-identical to solving the groups one by one. */
struct LuGrp {
    LuDev lu;
    sb_wt *work, *zout;
    unsigned int *flagL, *flagU;
    const double* qz;
    int ld;
};
#define LU_MG_MAX 64
/* the group table lives in constant memory: the group index is warp-uniform, so every field read
 * is a constant-cache broadcast (as cheap as a kernel parameter) */
__constant__ LuGrp c_lugrp[LU_MG_MAX];
template <int RPL, int LU_CHUNK, int MW>
__global__ void __launch_bounds__(256)
lu_solve_mg(const int2* order, unsigned int* ticket, const unsigned int* epoch_p, unsigned int nt) {
    const int lane = threadIdx.x & 31;
    const unsigned int epoch = *(volatile const unsigned int*)epoch_p;
    for (;;) {
        unsigned int t = 0;
        if (lane == 0) t = atomicAdd(ticket, 1u);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= nt) return;
        const int2 o = order[t];
        const LuGrp& G = c_lugrp[o.x];
        int wv[RPL];
#pragma unroll
        for (int r = 0; r < RPL; r++) wv[r] = MW ? G.lu.envw[lane + 32 * r] : 0;
        lu_row<RPL, LU_CHUNK, MW>(G.lu, G.work, G.zout, G.ld, G.flagL, G.flagU, epoch, NULL,
                                  G.qz, (unsigned int)o.y, wv, lane);
    }
}
/* chunk (terms per flag wait) by envs per lane: 8 for 1-2, 4 for 3-8, 2 for 9-16.  A chunk's
 * cn * RPL state loads are issued back to back between two waits, so the chunk size sets the
 * loads in flight per warp; the fp32 build can afford to double it (SB_LU_CHUNK2). */
template <int RPL> struct LuChunk { static const int v = RPL <= 2 ? 8 : (RPL <= 8 ? 4 : 2); };
}  /* extern "C++" */

/* launch the instantiation for rpl envs per lane (columns = 32 * rpl) */
#define LU_LAUNCH_RC(R, C) \
    if (mw == 1) lu_solve_t<R, C, 1><<<blocks, tpb, 0, st>>>(lu, work, zout, ld, fl, fu, tk, ep, ts, tbeg, tend, qz); \
    else if (mw == 2) lu_solve_t<R, C, 2><<<blocks, tpb, 0, st>>>(lu, work, zout, ld, fl, fu, tk, ep, ts, tbeg, tend, qz); \
    else lu_solve_t<R, C, 0><<<blocks, tpb, 0, st>>>(lu, work, zout, ld, fl, fu, tk, ep, ts, tbeg, tend, qz);
/* SB_LU_CHUNK2=1 (fp32 build only) doubles the chunk, i.e. the loads in flight per flag wait
 * (cn * RPL independent loads): the fp32 solve stalls on memory latency, not on bytes. */
#ifdef SB_SOLVE_F32
#define LU_CASE(R) case R: if (chunk2) { LU_LAUNCH_RC(R, 2 * LuChunk<R>::v) } else { LU_LAUNCH_RC(R, LuChunk<R>::v) } break;
#else
#define LU_CASE(R) case R: LU_LAUNCH_RC(R, LuChunk<R>::v) break;
#endif
/* Tickets tbeg..tend-1 (0..n-1 forward rows, n..2n-1 backward rows) claimed from the counter *tk
 * (which must start at 0).  The whole solve is [0, 2n); the env splits it into a forward launch and
 * a backward launch (with the air-side correction qz) around the air-side computation. */
static inline void lu_launch_range(int rpl, LuDev lu, sb_wt* work, sb_wt* zout, int ld, unsigned int* fl,
                                   unsigned int* fu, unsigned int* tk, const unsigned int* ep,
                                   unsigned long long* ts, int blocks, int tpb, cudaStream_t st,
                                   unsigned int tbeg, unsigned int tend, const double* qz) {
    /* 0: one weather; 2: several, one per 32-column slab (values shuffled); 1: per-lane values.
     * SB_FORCE_MW=1|2 forces a mode for A/B timing (any envw is valid for one weather). */
    static const int force_mw = getenv("SB_FORCE_MW") ? atoi(getenv("SB_FORCE_MW")) : 0;
#ifdef SB_SOLVE_F32
    static const int chunk2 = getenv("SB_LU_CHUNK2") != NULL;
#endif
    int mw = lu.nw > 1 ? (lu.slab_uniform ? 2 : 1) : 0;
    if (force_mw && lu.envw != NULL) mw = force_mw;
    switch (rpl) {
        LU_CASE(1) LU_CASE(2) LU_CASE(3) LU_CASE(4) LU_CASE(5) LU_CASE(6) LU_CASE(7) LU_CASE(8)
        LU_CASE(9) LU_CASE(10) LU_CASE(11) LU_CASE(12) LU_CASE(13) LU_CASE(14) LU_CASE(15)
        default: LU_CASE(16)
    }
}
#define LU_MG_CASE(R) case R: if (mw == 1) lu_solve_mg<R, LuChunk<R>::v, 1><<<blocks, tpb, 0, st>>>(order, tk, ep, nt); \
                              else if (mw == 2) lu_solve_mg<R, LuChunk<R>::v, 2><<<blocks, tpb, 0, st>>>(order, tk, ep, nt); \
                              else lu_solve_mg<R, LuChunk<R>::v, 0><<<blocks, tpb, 0, st>>>(order, tk, ep, nt); break;
static inline void lu_launch_mg(int rpl, int mw, const int2* order, unsigned int* tk,
                                const unsigned int* ep, unsigned int nt, int blocks, int tpb, cudaStream_t st) {
    switch (rpl) {
        LU_MG_CASE(1) LU_MG_CASE(2) LU_MG_CASE(3) LU_MG_CASE(4) LU_MG_CASE(5) LU_MG_CASE(6) LU_MG_CASE(7)
        LU_MG_CASE(8) LU_MG_CASE(9) LU_MG_CASE(10) LU_MG_CASE(11) LU_MG_CASE(12) LU_MG_CASE(13)
        LU_MG_CASE(14) LU_MG_CASE(15)
        default: LU_MG_CASE(16)
    }
}
static inline void lu_launch_solve(int rpl, LuDev lu, sb_wt* work, sb_wt* zout, int ld, unsigned int* fl,
                                   unsigned int* fu, unsigned int* tk, const unsigned int* ep,
                                   unsigned long long* ts, int blocks, int tpb, cudaStream_t st) {
    lu_launch_range(rpl, lu, work, zout, ld, fl, fu, tk, ep, ts, blocks, tpb, st, 0u, 2u * (unsigned)lu.n, NULL);
}
/* per-solve prologue: new epoch (the flags of the previous solve become stale) and a
 * fresh ticket counter; on the device so a captured launch sequence can be replayed */
__global__ void lu_prep_k(unsigned int* epoch, unsigned int* ticket, unsigned int* flagL,
                          unsigned int* flagU, int n, int two = 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int e = *epoch + 1u;
        if (e == 0u) e = 1u;
        *epoch = e;
        ticket[0] = 0u;
        if (two) ticket[1] = 0u;
    }
}

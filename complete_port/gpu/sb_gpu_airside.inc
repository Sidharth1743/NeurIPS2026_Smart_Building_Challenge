/* airside_k body, included twice by sb_gpu_kernels.cuh (AIR_NAME / AIR_GLOBAL): the k x k air-side
 * matrix lives in shared memory, or in global memory for buildings with too many coupled VAVs. */
__global__ void AIR_NAME(GpuSim* sims, const int* sp, const int* si, const double* sx,
                          const int* ap, const int* ai, const double* ax,
                          const double* sresp0, const int* envw, int k, int ld, double* qout, double* mg) {
    int e = blockIdx.x;
    GpuSim* s = &sims[e];
    const double* sresp = sresp0 + (size_t)envw[e] * k * k;   /* this env's weather */
    extern __shared__ double sbair[];
#if AIR_GLOBAL
    /* more than ~110 coupled VAVs: the k x k matrix does not fit in shared memory */
    double* M = mg + (size_t)e * k * k;
    double* q = sbair;
    double* fr = sbair + k;
#else
    double* M = sbair;             /* k*k */
    double* q = sbair + (size_t)k * k;
    double* fr = sbair + (size_t)k * k + k;   /* per-column multipliers */
#endif
    int tid = threadIdx.x, nthr = blockDim.x;
    double* acc = s->sensed_mean;   /* filled by sensed_y_k */
    for (int i = tid; i < k; i += nthr)
        q[i] = s->air_g[i] * (s->air_T[i] - acc[i]);
    __syncthreads();
    for (int i = tid; i < k * k; i += nthr) {
        int rr = i / k, cc = i % k;
        M[i] = (rr == cc ? 1.0 : 0.0) + s->air_g[rr] * sresp[rr * k + cc];
    }
    __syncthreads();
    /* forward elimination, one column at a time, exactly the arithmetic the
     * CPU dense_solve does: no pivot-row normalization, f = M[r][c]/d, and
     * b[r] -= f*b[c] BEFORE M[r][c] is overwritten. The multipliers are staged
     * in shared memory so no thread reads a row another is updating. */
    /* The pivot search, the row swap and the multipliers are spread over the block; every value is
     * computed by the same expression as the serial version (same pivot: the first row holding the
     * largest |M[r][c]|, the diagonal winning ties; fr = M[r][c] / d), so the result is unchanged. */
    __shared__ double s_bv[8];
    __shared__ int s_bi[8];
    __shared__ int s_piv;
    const int lane = tid & 31, wid = tid >> 5, nw = (nthr + 31) >> 5;
    for (int c = 0; c < k; c++) {
        const double d0 = fabs(M[c * k + c]);
        double bv = -1.0; int bi = k;            /* candidates below the diagonal */
        for (int rr = c + 1 + tid; rr < k; rr += nthr) {
            const double v = fabs(M[rr * k + c]);
            if (v > bv) { bv = v; bi = rr; }     /* NaN never wins */
        }
        for (int o = 16; o > 0; o >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, bv, o);
            const int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_bv[wid] = bv; s_bi[wid] = bi; }
        __syncthreads();
        if (tid == 0) {
            double b = s_bv[0]; int i0 = s_bi[0];
            for (int w = 1; w < nw; w++)
                if (s_bv[w] > b || (s_bv[w] == b && s_bi[w] < i0)) { b = s_bv[w]; i0 = s_bi[w]; }
            /* serial rule: best = |M[c][c]|, replaced only by a strictly larger later entry */
            s_piv = (i0 < k && b > d0) ? i0 : c;
        }
        __syncthreads();
        const int piv = s_piv;
        if (piv != c) {
            for (int j = c + tid; j < k; j += nthr) {
                const double t0 = M[c * k + j]; M[c * k + j] = M[piv * k + j]; M[piv * k + j] = t0;
            }
            if (tid == 0) { const double tb = q[c]; q[c] = q[piv]; q[piv] = tb; }
            __syncthreads();
        }
        const double d = M[c * k + c];
        for (int rr = c + 1 + tid; rr < k; rr += nthr) fr[rr] = M[rr * k + c] / d;
        __syncthreads();
        for (int rr = c + 1 + tid; rr < k; rr += nthr)
            if (fr[rr] != 0.0) q[rr] -= fr[rr] * q[c];
        __syncthreads();
        for (int ix = tid; ix < (k - c - 1) * (k - c); ix += nthr) {
            int rr = c + 1 + ix / (k - c), j = c + ix % (k - c);
            if (fr[rr] != 0.0) M[rr * k + j] -= fr[rr] * M[c * k + j];
        }
        __syncthreads();
    }
    /* back substitution, ascending-order inner sums as in dense_solve */
    for (int r = k - 1; r >= 0; r--) {
        if (tid == 0) {
            double sum = q[r];
            for (int j = r + 1; j < k; j++) sum -= M[r * k + j] * q[j];
            q[r] = sum / M[r * k + r];
        }
        __syncthreads();
    }
    /* The correction A^-1 (applied q) is folded into the backward triangular solve (row i starts
     * from z_i + sum_j Z_ij q_j, Z = L^-1 P_r applied): hand it q in the solver's [j][ld] layout. */
    for (int j = tid; j < k; j += nthr) qout[(size_t)j * ld + e] = q[j];
}

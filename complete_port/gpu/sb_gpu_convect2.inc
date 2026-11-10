/* The convection producer kernel body, included twice by sb_gpu_kernels.cuh: CV2_NAME is the
 * kernel's name and CV2_GLOBAL selects where the swap list / permutation live. */
__global__ void __launch_bounds__(256, 1)
CV2_NAME(const int* room_off, const int* room_cells, const int* cand_off,
           const int* cand_idx, int n_rooms, int nc, int max_room, double p,
           int distance, GpuRng* rng, int* out_dst, int* out_src, int* out_count,
           int list_cap, unsigned int* lr_g, unsigned short* p_g) {
    const int e = blockIdx.x;
    if (p == 0.0 || distance == 0) {
        if (threadIdx.x == 0) out_count[e] = 0;
        return;
    }
    const int tid = threadIdx.x, nthr = blockDim.x;
    extern __shared__ __align__(16) unsigned char cv_smem[];
    unsigned int* ring = (unsigned int*)cv_smem;           /* CV_RING + 4 words  */
    unsigned int* bitsE = ring + CV_RING + 4;              /* CV_BW words/parity */
    unsigned int* bitsO = bitsE + CV_BW;
    unsigned int* A = bitsO + CV_BW;                       /* MT state           */
    unsigned int* B = A + MT_N;                            /* twist scratch      */
#if CV2_GLOBAL
    /* huge rooms: the swap list and the permutation live in global memory (L2), only the
     * candidate counts the chain reads stay in shared */
    unsigned int* LR = lr_g + (size_t)e * max_room;
    unsigned short* cntP = (unsigned short*)(B + MT_N);
    unsigned short* Pm = p_g + (size_t)e * max_room;
#else
    unsigned int* LR = B + MT_N;                           /* max_room words     */
    unsigned short* cntP = (unsigned short*)(LR + max_room);   /* cnt, then P (overlaid) */
    unsigned short* Pm = cntP;
#endif
    __shared__ int s_F, s_gA0, s_gidx, s_status, s_m, s_pos;

    int* odst = out_dst + (size_t)e * list_cap;
    int* osrc = out_src + (size_t)e * list_cap;
    __shared__ int s_off;
    const unsigned long long p_int = (unsigned long long)(p * 9007199254740992.0);

    for (int i = tid; i < MT_N; i += nthr) A[i] = rng[e].mt[i];
    for (int i = tid; i < CV_BW; i += nthr) { bitsE[i] = 0u; bitsO[i] = 0u; }
    if (tid == 0) {
        s_gidx = (int)rng[e].idx;
        s_gA0 = -s_gidx;
        s_F = 0;
        s_pos = 0;
        s_off = 0;
    }
    __syncthreads();

    CvChain c;
    c.j = 0; c.pos = 0; c.mcol = 0; c.phase = 0; c.t = 0;

    for (int rm = 0; rm < n_rooms; rm++) {
        const int base = room_off[rm];
        const int n = room_off[rm + 1] - base;
        if (n <= 0) continue;
        for (int j = tid; j < n; j += nthr) {
            int d = cand_off[base + j + 1] - cand_off[base + j];
            cntP[j] = (unsigned short)(d > 0 ? d : 0);
        }
        if (tid == 0) { c.j = 0; c.mcol = 0; c.phase = 0; c.t = 0; }
        __syncthreads();
        for (;;) {
            if (tid == 0) s_status = cv_chain_run(c, ring, bitsE, bitsO, cntP, LR, n, s_F);
            __syncthreads();
            if (s_status == 0) break;
            /* ---- generate the next chunk of the stream (all threads) ---- */
            {
                int gidx = s_gidx, F = s_F, gA0 = s_gA0;
                if (gidx >= MT_N) {
                    mt_twist_passes(A, B, nthr);
                    gidx = 0;
                    gA0 = F;
                }
                const int cntw = MT_N - gidx;
                for (int i = tid; i < cntw; i += nthr)
                    ring[(F + i) & CV_RING_MASK] = grng_temper(A[gidx + i]);
                __syncthreads();
                if (tid < 4) ring[CV_RING + tid] = ring[tid];     /* mirror */
                const int Fn = F + cntw;
                const int lo = F > 0 ? F - 1 : 0, hi = Fn - 1;   /* pair starts */
                if (hi > lo) {
                    const int lane = tid & 31, wid = tid >> 5, nw = nthr >> 5;
                    for (int par = 0; par < 2; par++) {
                        unsigned int* BW = par ? bitsO : bitsE;
                        /* pair starts g = 2h + par with lo <= g < hi */
                        const int hmin = (lo - par + 1) >> 1;          /* ceil */
                        const int hmax = (hi - 1 - par) >> 1;          /* floor */
                        if (hmax < hmin) continue;
                        for (int hq = (hmin >> 5) + wid; hq <= (hmax >> 5); hq += nw) {
                            const int h = hq * 32 + lane;
                            const int g = 2 * h + par;
                            const bool in = (h >= hmin) && (h <= hmax);
                            bool flag = false;
                            if (in) {
                                const unsigned int a = ring[g & CV_RING_MASK];
                                const unsigned int b = ring[(g + 1) & CV_RING_MASK];
                                const unsigned long long D =
                                    ((unsigned long long)(a >> 5) << 26) |
                                    (unsigned long long)(b >> 6);
                                flag = !(D > p_int);
                            }
                            const unsigned int fb = __ballot_sync(0xffffffffu, flag);
                            const unsigned int mb = __ballot_sync(0xffffffffu, in);
                            if (lane == 0) {
                                const int w = hq & (CV_BW - 1);
                                BW[w] = (BW[w] & ~mb) | fb;
                            }
                        }
                    }
                }
                __syncthreads();
                if (tid == 0) { s_gidx = MT_N; s_F = Fn; s_gA0 = gA0; }
                __syncthreads();
            }
        }
        const int m = c.mcol;     /* only thread 0's copy is meaningful; share it */
        if (tid == 0) { s_m = m; s_pos = c.pos; }
        __syncthreads();
        const int mm = s_m;
        if (mm == 0) continue;
        for (int j = tid; j < n; j += nthr) Pm[j] = (unsigned short)j;   /* P = id */
        for (int t = tid; t < mm; t += nthr) {
            const unsigned int w = LR[t];
            const int a = (int)(w & 0xFFFFu), v = (int)(w >> 16);
            const int b = cand_idx[cand_off[base + a] + v] - base;
            LR[t] = (unsigned int)a | ((unsigned int)b << 16);
        }
        __syncthreads();
        if (tid == 0) {
            for (int t = 0; t < mm; t++) {
                const unsigned int w = LR[t];
                const int a = (int)(w & 0xFFFFu), b = (int)(w >> 16);
                const unsigned short pa = Pm[a], pb = Pm[b];
                Pm[a] = pb;
                Pm[b] = pa;
            }
        }
        __syncthreads();
        /* The room's net permutation as (destination cell, source cell) pairs over
         * the touched cells: field[dst] <- old field[src], applied by conv_apply_*.
         * Nothing here reads the temperatures - the swap decisions are a function
         * of the random stream and the room tables only. */
        {
            const int off = s_off;
            for (int idx = tid; idx < 2 * mm; idx += nthr) {
                const unsigned int w = LR[idx < mm ? idx : idx - mm];
                const int x = (int)(idx < mm ? (w & 0xFFFFu) : (w >> 16));
                const int qx = base + x, qs = base + Pm[x];
                odst[off + idx] = room_cells[2 * qx] * nc + room_cells[2 * qx + 1];
                osrc[off + idx] = room_cells[2 * qs] * nc + room_cells[2 * qs + 1];
            }
            __syncthreads();
            if (tid == 0) s_off = off + 2 * mm;
            __syncthreads();
        }
    }
    if (tid == 0) out_count[e] = s_off;
    /* the generator state, advanced by exactly the words consumed */
    __syncthreads();
    const int ia = s_pos - s_gA0;
    if (ia < 0 || ia > MT_N) {
        if (tid == 0) printf("convect2_k: generator lag (pos=%d gA0=%d)\n", s_pos, s_gA0);
        __trap();
    }
    for (int i = tid; i < MT_N; i += nthr) rng[e].mt[i] = A[i];
    if (tid == 0) rng[e].idx = (unsigned int)ia;
}

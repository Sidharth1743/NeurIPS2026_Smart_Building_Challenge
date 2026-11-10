/* Gate for the device replay of the baked SuperLU solve (sb_gpu_lusolve.cuh):
 * bit-identical to csim's sb_lu_solve on random right-hand sides, plus timing.
 *   usage: sb_gpu_lutest <blob.sbb> <nenv> [reps] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
extern "C" {
#include "sb_baked.h"
#include "sb_lu.h"
}
#include "sb_gpu_lusolve.cuh"
#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)

__global__ void perm_in_k(const double* b, const int* ir, double* work, int n, int nenv, int ld) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * ld) return;
    int row = (int)(i / ld), e = (int)(i % ld);
    work[i] = e < nenv ? b[(size_t)e * n + ir[row]] : 0.0;
}
__global__ void perm_out_k(const double* work, const int* pc, double* x, int n, int nenv, int ld) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * nenv) return;
    int e = (int)(i / n), k = (int)(i % n);
    x[i] = work[(size_t)pc[k] * ld + e];
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const int nenv = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 10;
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 2;
    const int n = (int)(sb_find(&b, "lu_L_indptr")->count) - 1;
    const int *Lp = sb_i32(&b, "lu_L_indptr"), *Li = sb_i32(&b, "lu_L_indices");
    const double* Lx = sb_f64(&b, "lu_L_data");
    const int *Up = sb_i32(&b, "lu_U_indptr"), *Ui = sb_i32(&b, "lu_U_indices");
    const double* Ux = sb_f64(&b, "lu_U_data");
    const int *pr = sb_i32(&b, "lu_perm_r"), *pc = sb_i32(&b, "lu_perm_c");
    SbLu cpu;
    if (sb_lu_build(&cpu, n, Lp, Li, Lx, Up, Ui, Ux, pr, pc) != 0) return 2;
    /* strict-lower CSR (ascending cols), strict-upper CSR (descending cols), diag */
    std::vector<int> lrp(n + 1, 0), urp(n + 1, 0);
    for (int j = 0; j < n; j++) {
        for (int p = Lp[j]; p < Lp[j + 1]; p++) if (Li[p] != j) lrp[Li[p] + 1]++;
        for (int p = Up[j]; p < Up[j + 1]; p++) if (Ui[p] != j) urp[Ui[p] + 1]++;
    }
    for (int i = 0; i < n; i++) { lrp[i + 1] += lrp[i]; urp[i + 1] += urp[i]; }
    std::vector<int> lci(lrp[n]), uci(urp[n]), lf(lrp.begin(), lrp.end() - 1), uf(urp.begin(), urp.end() - 1);
    std::vector<double> lcx(lrp[n]), ucx(urp[n]), ud(n);
    /* columns visited ascending for L; for U, visit columns DEscending so each row's list is descending */
    for (int j = 0; j < n; j++)
        for (int p = Lp[j]; p < Lp[j + 1]; p++)
            if (Li[p] != j) { int i = Li[p]; lci[lf[i]] = j; lcx[lf[i]] = Lx[p]; lf[i]++; }
    for (int j = n - 1; j >= 0; j--)
        for (int p = Up[j]; p < Up[j + 1]; p++) {
            int i = Ui[p];
            if (i == j) { ud[j] = Ux[p]; continue; }
            uci[uf[i]] = j; ucx[uf[i]] = Ux[p]; uf[i]++;
        }
    std::vector<int> pcinv_unused;
    int *d_lrp, *d_lci, *d_urp, *d_uci, *d_ir, *d_pc; double *d_lcx, *d_ucx, *d_ud;
    auto up = [&](void** d, const void* h, size_t bytes) { cudaMalloc(d, bytes); cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice); };
    up((void**)&d_lrp, lrp.data(), 4 * (n + 1)); up((void**)&d_lci, lci.data(), 4 * lci.size()); up((void**)&d_lcx, lcx.data(), 8 * lcx.size());
    up((void**)&d_urp, urp.data(), 4 * (n + 1)); up((void**)&d_uci, uci.data(), 4 * uci.size()); up((void**)&d_ucx, ucx.data(), 8 * ucx.size());
    up((void**)&d_ud, ud.data(), 8 * n);
    up((void**)&d_ir, cpu.ir, 4 * n); up((void**)&d_pc, cpu.pc, 4 * n);
    std::vector<double> uinvh(n); for (int i = 0; i < n; i++) uinvh[i] = 1.0 / ud[i]; double* d_uinv; up((void**)&d_uinv, uinvh.data(), 8 * n);
    /* ticket order = rows sorted by dependency level (stable in index) */
    std::vector<int> ordL(n), ordU(n);
    {
        std::vector<int> lv(n, 0); int nl = 0;
        for (int i = 0; i < n; i++) { int m = 0; for (int p = lrp[i]; p < lrp[i + 1]; p++) m = std::max(m, lv[lci[p]] + 1); lv[i] = m; nl = std::max(nl, m + 1); }
        std::vector<int> cnt(nl + 1, 0); for (int i = 0; i < n; i++) cnt[lv[i] + 1]++; for (int l = 0; l < nl; l++) cnt[l + 1] += cnt[l];
        for (int i = 0; i < n; i++) ordL[cnt[lv[i]]++] = i;
        std::vector<int> lu2(n, 0); int nl2 = 0;
        for (int i = n - 1; i >= 0; i--) { int m = 0; for (int p = urp[i]; p < urp[i + 1]; p++) m = std::max(m, lu2[uci[p]] + 1); lu2[i] = m; nl2 = std::max(nl2, m + 1); }
        std::vector<int> cnt2(nl2 + 1, 0); for (int i = 0; i < n; i++) cnt2[lu2[i] + 1]++; for (int l = 0; l < nl2; l++) cnt2[l + 1] += cnt2[l];
        for (int i = n - 1; i >= 0; i--) ordU[cnt2[lu2[i]]++] = i;
    }
    int *d_ordL, *d_ordU; up((void**)&d_ordL, ordL.data(), 4 * n); up((void**)&d_ordU, ordU.data(), 4 * n);
    LuDev lu = {n, d_lrp, d_lci, d_lcx, d_urp, d_uci, d_ucx, d_ud, d_uinv, d_ordL, d_ordU};
    const int rpl = std::max(1, std::min(16, (nenv + 31) / 32));
    const int ld = 32 * rpl;
    double *d_b, *d_x, *d_w, *d_z; unsigned int *d_fl, *d_fu, *d_tk;
    cudaMalloc(&d_b, 8 * (size_t)nenv * n); cudaMalloc(&d_x, 8 * (size_t)nenv * n);
    cudaMalloc(&d_w, 8 * (size_t)n * ld); cudaMalloc(&d_z, 8 * (size_t)n * ld);
    cudaMalloc(&d_fl, 4 * n); cudaMalloc(&d_fu, 4 * n); cudaMalloc(&d_tk, 4);
    cudaMemset(d_fl, 0, 4 * n); cudaMemset(d_fu, 0, 4 * n);
    std::vector<double> hb((size_t)nenv * n), hx((size_t)nenv * n), hr((size_t)nenv * n), wk(n);
    unsigned long long s = 88172645463325252ull;
    for (auto& v : hb) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v = (double)(s >> 11) / 9007199254740992.0 * 600.0 - 300.0; }
    for (int e = 0; e < nenv; e++) sb_lu_solve(&cpu, &hb[(size_t)e * n], &hr[(size_t)e * n], wk.data());
    std::vector<double> hostw, hostz;
    {   /* the same pull algorithm on the host: separates algorithm errors from GPU races */
        hostz.assign(n, 0.0); std::vector<double>& z = hostz;
        hostw.assign(n, 0.0); std::vector<double>& w = hostw;
        size_t nd = 0;
        for (int i = 0; i < n; i++) w[i] = hb[cpu.ir[i]];
        for (int i = 0; i < n; i++) {
            double a = w[i];
            for (int p = lrp[i]; p < lrp[i + 1]; p++) { volatile double pr_ = lcx[p] * w[lci[p]]; a = a - pr_; }
            w[i] = a;
        }
        for (int i = n - 1; i >= 0; i--) {
            double a = w[i];
            for (int p = urp[i]; p < urp[i + 1]; p++) { volatile double pr_ = ucx[p] * z[uci[p]]; a = a - pr_; }
            z[i] = a / ud[i];
        }
        for (int k = 0; k < n; k++) if (memcmp(&z[cpu.pc[k]], &hr[k], 8) != 0) nd++;
        printf("host pull replay (env 0): %zu of %d differ\n", nd, n);
    }
    cudaMemcpy(d_b, hb.data(), 8 * hb.size(), cudaMemcpyHostToDevice);
    unsigned long long* d_ts; cudaMalloc(&d_ts, 24 * (size_t)n); cudaMemset(d_ts, 0, 24 * (size_t)n);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    unsigned int* d_ep; cudaMalloc(&d_ep, 4); cudaMemset(d_ep, 0, 4);
    int bad = 0; float best = 1e30f, tot = 0;
    for (int rep = 0; rep < reps; rep++) {
        lu_prep_k<<<1, 1>>>(d_ep, d_tk, d_fl, d_fu, n);
        cudaEventRecord(e0);
        perm_in_k<<<(unsigned)(((size_t)n * ld + 255) / 256), 256>>>(d_b, d_ir, d_w, n, nenv, ld);
        const int tpb = getenv("LU_TPB") ? atoi(getenv("LU_TPB")) : 256;
        const int blocks = getenv("LU_BLOCKS") ? atoi(getenv("LU_BLOCKS")) : 82 * 6;                     /* resident warps, 8 warps per block */
        lu_launch_solve(rpl, lu, d_w, d_z, ld, d_fl, d_fu, d_tk, d_ep, d_ts, blocks, tpb, 0);
        perm_out_k<<<(unsigned)(((size_t)n * nenv + 255) / 256), 256>>>(d_z, d_pc, d_x, n, nenv, ld);
        cudaEventRecord(e1);
        if (cudaEventSynchronize(e1) != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (rep > 0) { tot += ms; if (ms < best) best = ms; }
        if (rep == 0) {
            cudaMemcpy(hx.data(), d_x, 8 * hx.size(), cudaMemcpyDeviceToHost);
            size_t nd = 0; double md = 0;
            for (size_t i = 0; i < hx.size(); i++) if (memcmp(&hx[i], &hr[i], 8) != 0) { nd++; md = fmax(md, fabs(hx[i] - hr[i])); }
            printf("nenv=%d rpl=%d: %zu of %zu values differ from sb_lu_solve (max |d| %.3e)\n", nenv, rpl, nd, hx.size(), md);
            bad = nd != 0;
            { std::vector<double> gw((size_t)n * ld); cudaMemcpy(gw.data(), d_w, 8 * gw.size(), cudaMemcpyDeviceToHost);
              size_t fd = 0; int first = -1; for (int i = 0; i < n; i++) if (memcmp(&gw[(size_t)i * ld], &hostw[i], 8) != 0) { fd++; if (first < 0) first = i; }
              printf("  forward result y vs host: %zu of %d rows differ (first %d)\n", fd, n, first); }
            { std::vector<double> gz((size_t)n * ld); cudaMemcpy(gz.data(), d_z, 8 * gz.size(), cudaMemcpyDeviceToHost);
              long cntmod[9] = {0}, tot[9] = {0}; int firstbad = -1, lastbad = -1;
              for (int i = 0; i < n; i++) { int len = urp[i + 1] - urp[i]; tot[len % 8]++; if (memcmp(&gz[(size_t)i * ld], &hostz[i], 8) != 0) { cntmod[len % 8]++; if (firstbad < 0) firstbad = i; lastbad = i; } }
              { int i = n - 1; printf("  row n-1: urp nnz=%d y(gpu)=%.17g y(host)=%.17g z(gpu)=%.17g z(host)=%.17g ud=%.17g\n", urp[i+1]-urp[i], 0.0, hostw[i], gz[(size_t)i*ld], hostz[i], ud[i]); }
              printf("  backward z mismatches by (row nnz mod 8): "); for (int m = 0; m < 8; m++) printf("%ld/%ld ", cntmod[m], tot[m]); printf(" first=%d last=%d\n", firstbad, lastbad); }
            int shown = 0; long cnt_mod[3] = {0,0,0};
            for (int k = 0; k < n; k++) if (memcmp(&hx[k], &hr[k], 8) != 0) { cnt_mod[cpu.pc[k] % 3]++; if (shown++ < 6) printf("  k=%d row=%d gpu=%.17g cpu=%.17g\n", k, cpu.pc[k], hx[k], hr[k]); }
            printf("  mismatching rows by row%%3: %ld %ld %ld\n", cnt_mod[0], cnt_mod[1], cnt_mod[2]);
        }
    }
    {   /* timeline: completion time of each row (forward rows 0..n-1, backward rows n..2n-1) */
        std::vector<unsigned long long> ts(3 * (size_t)n);
        cudaMemcpy(ts.data(), d_ts, 24 * (size_t)n, cudaMemcpyDeviceToHost);
        unsigned long long t0 = ~0ull, tf = 0, tb = 0;
        for (int i = 0; i < n; i++) { t0 = std::min(t0, ts[i]); tf = std::max(tf, ts[i]); tb = std::max(tb, ts[n + i]); }
        /* time at which forward rows with index >= X completed, X = 90%,99% of n */
        auto q = [&](size_t off, double frac) { unsigned long long m = 0; for (int i = (int)(frac * n); i < n; i++) m = std::max(m, ts[off + i]); return (double)(m - t0) * 1e-3; };
        printf("timeline (us since first row done): forward rows<%.0f%% done at %.0f, <99%% at %.0f, all at %.0f; backward all at %.0f\n",
               99.0, 0.0, 0.0, (double)(tf - t0) * 1e-3, (double)(tb - t0) * 1e-3);
        double ft[11]; for (int k = 0; k <= 10; k++) { unsigned long long m = 0; for (int i = 0; i < (int)(n * (k / 10.0)) + 1 && i < n; i++) m = std::max(m, ts[i]); ft[k] = (double)(m - t0) * 1e-3; }
        { double bt[11]; unsigned long long tmin = ~0ull; for (int i = 0; i < n; i++) tmin = std::min(tmin, ts[n + i]);
          for (int k = 0; k <= 10; k++) { unsigned long long m = 0; int lo = n - (int)(n * (k / 10.0)) - 1; if (lo < 0) lo = 0; for (int i = lo; i < n; i++) m = std::max(m, ts[n + i]); bt[k] = (double)(m - t0) * 1e-3; }
          printf("  backward progress (top fraction of rows done -> us since fwd start): "); for (int k = 1; k <= 10; k++) printf("%d%%:%.0f ", k * 10, bt[k]); printf("\n"); }
        {   /* per-level latency on the narrow tail of the forward solve */
            std::vector<int> lev(n, 0); int nl = 0;
            for (int i = 0; i < n; i++) { int m = 0; for (int p = lrp[i]; p < lrp[i + 1]; p++) m = std::max(m, lev[lci[p]] + 1); lev[i] = m; nl = std::max(nl, m + 1); }
            std::vector<unsigned long long> tl(nl, 0); std::vector<int> wd(nl, 0);
            for (int i = 0; i < n; i++) { tl[lev[i]] = std::max(tl[lev[i]], ts[i]); wd[lev[i]]++; }
            { int b[6] = {0, 76, 200, 400, 700, 991}; unsigned long long t0f = ~0ull; for (int i = 0; i < n; i++) t0f = std::min(t0f, ts[i]);
              printf("  forward: completion of level groups (us): "); for (int g = 0; g < 5; g++) { unsigned long long m = 0; for (int l = b[g]; l < std::min(b[g + 1], nl); l++) m = std::max(m, tl[l]); printf("[%d,%d):%.0f ", b[g], b[g + 1], (double)(m - t0f) * 1e-3); } printf("\n"); }
            std::vector<double> d; for (int l = 76; l + 1 < nl; l++) if (tl[l + 1] > tl[l]) d.push_back((double)(tl[l + 1] - tl[l]) * 1e-3);
            std::sort(d.begin(), d.end());
            double sum = 0; for (double x : d) sum += x;
            printf("  narrow forward levels: %zu, per-level latency us: median %.2f mean %.2f p90 %.2f p99 %.2f (levels sum %.0f)\n", d.size(), d[d.size() / 2], sum / d.size(), d[d.size() * 9 / 10], d[d.size() * 99 / 100], sum);
        }
        {   /* backward per-level latency */
            int w_slow = 0;
            std::vector<int> lev(n, 0); int nl = 0;
            for (int i = n - 1; i >= 0; i--) { int m = 0; for (int p = urp[i]; p < urp[i + 1]; p++) m = std::max(m, lev[uci[p]] + 1); lev[i] = m; nl = std::max(nl, m + 1); }
            std::vector<unsigned long long> tl(nl, 0); std::vector<int> wd(nl, 0); std::vector<double> rowlen(nl, 0);
            for (int i = 0; i < n; i++) { tl[lev[i]] = std::max(tl[lev[i]], ts[n + i]); wd[lev[i]]++; rowlen[lev[i]] += urp[i + 1] - urp[i]; }
            int narrow = 0; long nr = 0; double nn = 0; for (int l = 0; l < nl; l++) if (wd[l] < 64) { narrow++; nr += wd[l]; nn += rowlen[l]; }
            printf("  backward: %d levels, %d narrow (<64 rows) with %ld rows and %.0f nnz (avg %.0f per row)\n", nl, narrow, nr, nn, nn / std::max(1L, nr));
            std::vector<double> d; for (int l = 1; l + 1 < nl; l++) if (wd[l] < 64 && tl[l + 1] > tl[l]) d.push_back((double)(tl[l + 1] - tl[l]) * 1e-3);
            std::sort(d.begin(), d.end()); double sum = 0; for (double x : d) sum += x;
            { std::vector<std::pair<double,int>> w; for (int l = 1; l + 1 < nl; l++) if (wd[l] < 64 && tl[l + 1] > tl[l]) w.push_back({(double)(tl[l + 1] - tl[l]) * 1e-3, l});
              std::sort(w.rbegin(), w.rend()); w_slow = w[0].second; printf("  slowest backward narrow levels (us, level, rows, avg nnz/row, max nnz/row):\n");
              for (int q = 0; q < 8 && q < (int)w.size(); q++) { int l = w[q].second; int mx = 0; for (int i = 0; i < n; i++) if (lev[i] == l) mx = std::max(mx, urp[i + 1] - urp[i]); printf("    %.0f us  level %d  rows %d  avg nnz %.0f  max nnz %d\n", w[q].first, l, wd[l], rowlen[l] / wd[l], mx); } }
            { int l = w_slow; printf("  rows of backward level %d: (start, end) us since fwd start:\n", l);
              for (int i = 0; i < n; i++) if (lev[i] == l) printf("    row %d nnz %d: start %.0f end %.0f\n", i, urp[i + 1] - urp[i], (double)(ts[2 * (size_t)n + i] - t0) * 1e-3, (double)(ts[n + i] - t0) * 1e-3); }
            printf("  backward narrow-level latency us: median %.2f mean %.2f p90 %.2f (n=%zu, sum %.0f)\n", d[d.size() / 2], sum / d.size(), d[d.size() * 9 / 10], d.size(), sum);
        }
        printf("  forward progress (row index fraction -> us): "); for (int k = 1; k <= 10; k++) printf("%d%%:%.0f ", k * 10, ft[k]); printf("\n");
    }
    printf("solve incl. perm in/out: best %.3f ms, mean %.3f ms (%d reps)\n", best, tot / (reps > 1 ? reps - 1 : 1), reps - 1);
    printf(bad ? "LU DEVICE SOLVE: FAIL\n" : "LU DEVICE SOLVE: BIT-IDENTICAL\n");
    return bad;
}

/* M5 device solve gate + batching (cuDSS, NVIDIA sparse direct solver).
 *
 * The FDM matrix is time-invariant, so the factorization happens once and every
 * environment step costs solves only. Batching across a scenario group means
 * many right-hand sides against ONE factorization.
 *
 * Timing uses the host clock around a stream sync: cuDSS schedules work on
 * internal streams, so CUDA events recorded on ours do not observe completion.
 *
 * usage: sb_gpu_cudss <blob.sbb> [nrhs] [reps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <cudss.h>

extern "C" {
#include "sb_baked.h"
#include "sb_lu.h"
}

#define CK(x)                                                                 \
    do {                                                                      \
        cudssStatus_t st_ = (x);                                              \
        if (st_ != CUDSS_STATUS_SUCCESS) {                                    \
            printf("cuDSS error %d at line %d (%s)\n", (int)st_, __LINE__, #x);\
            return 1;                                                         \
        }                                                                     \
    } while (0)
#define CU(x)                                                                 \
    do {                                                                      \
        cudaError_t e_ = (x);                                                 \
        if (e_ != cudaSuccess) {                                              \
            printf("cuda error %s at line %d (%s)\n",                         \
                   cudaGetErrorString(e_), __LINE__, #x);                     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + 1e-6 * (double)t.tv_nsec;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        printf("usage: %s <blob.sbb> [nrhs] [reps]\n", argv[0]);
        return 2;
    }
    int nrhs = argc > 2 ? atoi(argv[2]) : 1;
    int reps = argc > 3 ? atoi(argv[3]) : 20;

    SbBlob blob;
    if (sb_open(&blob, argv[1]) != 0)
        return 1;
    const int n = (int)sb_i64(&blob, "A_system_shape")[0];
    const int* Sc_p = sb_i32(&blob, "A_system_indptr");
    const int* Sc_i = sb_i32(&blob, "A_system_indices");
    const double* Sc_x = sb_f64(&blob, "A_system_data");
    if (!Sc_p || !Sc_i || !Sc_x) { printf("blob has no A_system: re-bake\n"); return 1; }
    /* cuDSS takes CSR only; convert, rows sorted by column. */
    std::vector<int> Ap_v(n + 1, 0), Ai_v;
    std::vector<double> Ax_v;
    {
        std::vector<int> cnt(n, 0);
        for (int j = 0; j < n; j++)
            for (int q = Sc_p[j]; q < Sc_p[j + 1]; q++) cnt[Sc_i[q]]++;
        for (int i = 0; i < n; i++) Ap_v[i + 1] = Ap_v[i] + cnt[i];
        std::vector<int> nxt(Ap_v.begin(), Ap_v.end() - 1);
        Ai_v.assign((size_t)Ap_v[n], 0);
        Ax_v.assign((size_t)Ap_v[n], 0.0);
        for (int j = 0; j < n; j++)
            for (int q = Sc_p[j]; q < Sc_p[j + 1]; q++) {
                int i = Sc_i[q];
                Ai_v[nxt[i]] = j;
                Ax_v[nxt[i]] = Sc_x[q];
                nxt[i]++;
            }
    }
    const int* Ap = Ap_v.data();
    const int* Ai = Ai_v.data();
    const double* Ax = Ax_v.data();
    const int nnz = Ap_v[n];
    const double* bref = sb_f64(&blob, "solref_b");
    const double* xref = sb_f64(&blob, "solref_x");
    printf("blob %s: n=%d nnz=%d nrhs=%d reps=%d\n", argv[1], n, nnz, nrhs, reps);

    /* ---- CPU reference on the same system ---- */
    SbLu lu;
    if (sb_lu_build(&lu, n, sb_i32(&blob, "lu_L_indptr"), sb_i32(&blob, "lu_L_indices"),
                    sb_f64(&blob, "lu_L_data"), sb_i32(&blob, "lu_U_indptr"),
                    sb_i32(&blob, "lu_U_indices"), sb_f64(&blob, "lu_U_data"),
                    sb_i32(&blob, "lu_perm_r"), sb_i32(&blob, "lu_perm_c")) != 0)
        return 1;
    std::vector<double> xcpu((size_t)n * nrhs), work(n);
    double mref = 0.0;
    for (int i = 0; i < n; i++) mref = fmax(mref, fabs(xref[i]));

    std::vector<double> B((size_t)n * nrhs), X((size_t)n * nrhs, 0.0);
    for (int c = 0; c < nrhs; c++)
        for (int i = 0; i < n; i++)
            B[(size_t)c * n + i] = bref[i]
                + (c ? (double)c * 1e-3 * cos(0.5 + 1e-4 * i) : 0.0);
    for (int c = 0; c < nrhs; c++)
        sb_lu_solve(&lu, &B[(size_t)c * n], &xcpu[(size_t)c * n], work.data());
    double ecpu = 0.0;
    for (int i = 0; i < n; i++) ecpu = fmax(ecpu, fabs(xcpu[i] - xref[i]));
    printf("cpu replay vs solref_x: max err %.3e\n", ecpu);

    double *d_Ax, *d_b, *d_x;
    int *d_Ap, *d_Ai;
    CU(cudaMalloc(&d_Ap, (size_t)(n + 1) * 4));
    CU(cudaMalloc(&d_Ai, (size_t)nnz * 4));
    CU(cudaMalloc(&d_Ax, (size_t)nnz * 8));
    CU(cudaMalloc(&d_b, (size_t)n * nrhs * 8));
    CU(cudaMalloc(&d_x, (size_t)n * nrhs * 8));
    CU(cudaMemcpy(d_Ap, Ap, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ai, Ai, (size_t)nnz * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ax, Ax, (size_t)nnz * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_b, B.data(), (size_t)n * nrhs * 8, cudaMemcpyHostToDevice));
    CU(cudaMemset(d_x, 0, (size_t)n * nrhs * 8));

    cudssHandle_t handle;
    CK(cudssCreate(&handle));
    cudaStream_t stream;
    CU(cudaStreamCreate(&stream));
    CK(cudssSetStream(handle, stream));
    cudssConfig_t cfg;
    CK(cudssConfigCreate(&cfg));
    cudssData_t data;
    CK(cudssDataCreate(handle, &data));

    cudssMatrix_t A, Bw, Xw, B1, X1;
    CK(cudssMatrixCreateCsr(&A, n, n, nnz, d_Ap, NULL, d_Ai, d_Ax,
                            CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                            CUDSS_BASE_ZERO));
    CK(cudssMatrixCreateDn(&Bw, n, nrhs, n, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&Xw, n, nrhs, n, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    /* Analysis + factorization are RHS-independent: run them on one column. */
    CK(cudssMatrixCreateDn(&B1, n, 1, n, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&X1, n, 1, n, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    double h0 = now_ms();
    CK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, cfg, data, A, X1, B1));
    CK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X1, B1));
    CU(cudaStreamSynchronize(stream));
    printf("analysis+factorization: %.1f ms\n", now_ms() - h0);

    h0 = now_ms();
    CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, Xw, Bw));
    CU(cudaStreamSynchronize(stream));
    printf("first solve (nrhs=%d): %.3f ms\n", nrhs, now_ms() - h0);

    std::vector<double> Xg((size_t)n * nrhs);
    CU(cudaMemcpy(Xg.data(), d_x, (size_t)n * nrhs * 8, cudaMemcpyDeviceToHost));
    double worst = 0.0, worstref = 0.0;
    int wc = -1;
    for (int c = 0; c < nrhs; c++)
        for (int i = 0; i < n; i++) {
            double d = fabs(Xg[(size_t)c * n + i] - xcpu[(size_t)c * n + i]);
            if (d > worst) { worst = d; wc = c; }
            if (c == 0) worstref = fmax(worstref, fabs(Xg[i] - xref[i]));
        }
    printf("cuDSS vs cpu replay: max err %.3e (col %d)\n", worst, wc);
    printf("cuDSS vs solref_x  : max err %.3e -> rel %.2e\n", worstref,
           mref > 0 ? worstref / mref : 0.0);
    {
        std::vector<double> r(n, 0.0);
        double xn = 0.0, an = 0.0;
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int q = Ap[i]; q < Ap[i + 1]; q++) s += Ax[q] * Xg[Ai[q]];
            r[i] = s - B[i];
            xn = fmax(xn, fabs(Xg[i]));
            double rs = 0.0;
            for (int q = Ap[i]; q < Ap[i + 1]; q++) rs += fabs(Ax[q]);
            an = fmax(an, rs);
        }
        double rn = 0.0;
        for (int i = 0; i < n; i++) rn = fmax(rn, fabs(r[i]));
        printf("residual rel = %.3e\n", (an > 0 && xn > 0) ? rn / (an * xn) : 1.0);
    }

    h0 = now_ms();
    for (int r = 0; r < reps; r++) {
        CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, Xw, Bw));
    }
    CU(cudaStreamSynchronize(stream));
    double tcu = (now_ms() - h0) / reps;
    h0 = now_ms();
    for (int r = 0; r < reps; r++)
        for (int c = 0; c < nrhs; c++)
            sb_lu_solve(&lu, &B[(size_t)c * n], &xcpu[(size_t)c * n], work.data());
    double tcp = (now_ms() - h0) / reps;
    printf("solve batch (nrhs=%d): cuDSS %.3f ms | cpu %.3f ms | %.1fx\n", nrhs,
           tcu, tcp, tcu > 0 ? tcp / tcu : 0.0);
    printf("  per RHS: cuDSS %.4f ms | cpu %.4f ms\n", tcu / nrhs, tcp / nrhs);
    int bad = !(worst / (mref > 0 ? mref : 1.0) < 1e-9);
    printf(bad ? "CUDSS SOLVE: FAIL\n" : "CUDSS SOLVE: OK\n");
    return bad;
}

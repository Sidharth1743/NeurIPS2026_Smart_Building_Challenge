/* M5a: device-side SuperLU replay, batched over RHS columns.
 *
 * The FDM matrix is time-invariant (the bake stores one factorization), so the
 * whole per-step cost is SpTRSV with a FIXED matrix and a batch of RHS: three
 * solves per env step (forcing, and two air-side corrections), and one RHS per
 * environment. Same scenario across envs => one multi-RHS solve per group,
 * which is what cuSPARSE SpSM does (a dense B with one column per env).
 *
 * This program gates the device solve against the baked reference solution
 * (solref_b -> solref_x, the same ground truth M1 used) and against the CPU
 * replay in sb_lu.c, then times both.
 *
 * build (inside pufferai/puffertank:5.0):
 *   nvcc -O2 -std=c++17 sb_gpu_solve.cu csim/*.c -o sb_gpu_solve \
 *        -Icsim -lcusparse
 * run:
 *   ./sb_gpu_solve <blob.sbb> [nrhs] [reps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

extern "C" {
#include "sb_baked.h"
#include "sb_lu.h"
}

#define CK(x)                                                                  \
    do {                                                                       \
        cusparseStatus_t st_ = (x);                                             \
        if (st_ != CUSPARSE_STATUS_SUCCESS) {                                   \
            printf("cusparse error %d at %s:%d (%s)\n", (int)st_, __FILE__,     \
                   __LINE__, #x);                                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#define CU(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                   \
        if (e_ != cudaSuccess) {                                                \
            printf("cuda error %s at %s:%d\n", cudaGetErrorString(e_),          \
                   __FILE__, __LINE__);                                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* y[i] = x[idx[i]]  (gather, one thread per element) */
/* Batch matrices are (n, nrhs) column-major: the permutation is per column.
 * y[c*n+j] = x[c*n + idx[j]]   (gather, row permutation applied to the RHS) */
__global__ void gather_k(const double* __restrict__ x,
                         const int* __restrict__ idx,
                         double* __restrict__ y, int n, int nrhs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n * nrhs)
        return;
    long c = i / n, j = i - c * n;
    y[i] = x[c * n + idx[j]];
}

/* y[c*n + idx[j]] = x[c*n + j]   (scatter back to natural ordering) */
__global__ void scatter_k(const double* __restrict__ x,
                          const int* __restrict__ idx,
                          double* __restrict__ y, int n, int nrhs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n * nrhs)
        return;
    long c = i / n, j = i - c * n;
    y[c * n + idx[j]] = x[i];
}

/* CSC -> CSR. cuSPARSE's SpSM only takes CSR/COO/Sliced-ELL. Rows are filled
 * in increasing column order (the outer loop is over columns), which is the
 * order sb_lu_solve accumulates a row's contributions in, so the device solve
 * differs from the CPU replay only in summation form, not in ordering.
 * drop_diag: L is unit-diagonal in the bake, so its stored diagonal is dropped
 * and cuSPARSE is told the diagonal is unit. */
static void csc_to_csr(int n, const int* Ap, const int* Ai, const double* Ax,
                       int drop_diag, std::vector<int>& Bp,
                       std::vector<int>& Bi, std::vector<double>& Bx) {
    std::vector<int> cnt(n, 0);
    for (int j = 0; j < n; j++)
        for (int q = Ap[j]; q < Ap[j + 1]; q++)
            if (!(drop_diag && Ai[q] == j))
                cnt[Ai[q]]++;
    Bp.assign(n + 1, 0);
    for (int i = 0; i < n; i++)
        Bp[i + 1] = Bp[i] + cnt[i];
    std::vector<int> next(Bp.begin(), Bp.end() - 1);
    int nnz = Bp[n];
    Bi.assign(nnz, 0);
    Bx.assign(nnz, 0.0);
    for (int j = 0; j < n; j++)
        for (int q = Ap[j]; q < Ap[j + 1]; q++) {
            int i = Ai[q];
            if (drop_diag && i == j)
                continue;
            Bi[next[i]] = j;
            Bx[next[i]] = Ax[q];
            next[i]++;
        }
}

static double now_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CU(cudaEventElapsedTime(&ms, a, b));
    return (double)ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <blob.sbb> [nrhs] [reps]\n", argv[0]);
        return 2;
    }
    int nrhs = argc > 2 ? atoi(argv[2]) : 8;
    int reps = argc > 3 ? atoi(argv[3]) : 30;

    SbBlob blob;
    if (sb_open(&blob, argv[1]) != 0)
        return 1;
    int64_t* lshape = (int64_t*)sb_i64(&blob, "lu_L_shape");
    int n = (int)lshape[0];
    int nnzL = (int)sb_find(&blob, "lu_L_data")->count;
    int nnzU = (int)sb_find(&blob, "lu_U_data")->count;
    const int* Lp = sb_i32(&blob, "lu_L_indptr");
    const int* Li = sb_i32(&blob, "lu_L_indices");
    const double* Lx = sb_f64(&blob, "lu_L_data");
    const int* Up = sb_i32(&blob, "lu_U_indptr");
    const int* Ui = sb_i32(&blob, "lu_U_indices");
    const double* Ux = sb_f64(&blob, "lu_U_data");
    const int* perm_r = sb_i32(&blob, "lu_perm_r");
    const int* perm_c = sb_i32(&blob, "lu_perm_c");
    const double* bref = sb_f64(&blob, "solref_b");
    const double* xref = sb_f64(&blob, "solref_x");
    printf("blob %s: n=%d nnzL=%d nnzU=%d nrhs=%d reps=%d\n", argv[1], n, nnzL,
           nnzU, nrhs, reps);

    /* ---- host reference: the baked x_ref, and the CPU replay of it ---- */
    SbLu lu;
    if (sb_lu_build(&lu, n, Lp, Li, Lx, Up, Ui, Ux, perm_r, perm_c) != 0)
        return 1;
    std::vector<double> work(n), xcpu(n);
    sb_lu_solve(&lu, bref, xcpu.data(), work.data());
    double e_ref = 0.0;
    for (int i = 0; i < n; i++)
        e_ref = fmax(e_ref, fabs(xcpu[i] - xref[i]));
    printf("cpu replay vs baked solref_x: max err %.3e\n", e_ref);

    /* ---- RHS batch: column c is a distinct perturbation of solref_b ---- */
    std::vector<double> B((size_t)n * nrhs), Xcpu((size_t)n * nrhs);
    for (int c = 0; c < nrhs; c++) {
        for (int i = 0; i < n; i++) {
            double v = bref[i];
            if (c > 0)
                v += (double)c * 1e-3 * cos(0.5 + 1e-4 * (double)i) +
                     (double)c * 1e-6 * (double)(i % 97);
            B[(size_t)c * n + i] = v;
        }
        sb_lu_solve(&lu, &B[(size_t)c * n], &Xcpu[(size_t)c * n], work.data());
    }

    /* ---- device ---- */
    std::vector<int> Lp_c, Li_c, Up_c, Ui_c;
    std::vector<double> Lx_c, Ux_c;
    csc_to_csr(n, Lp, Li, Lx, 0, Lp_c, Li_c, Lx_c);
    csc_to_csr(n, Up, Ui, Ux, 0, Up_c, Ui_c, Ux_c);
    int nnzLc = (int)Lx_c.size(), nnzUc = (int)Ux_c.size();
    printf("CSR: L nnz %d -> %d (unit diag dropped), U nnz %d -> %d\n", nnzL,
           nnzLc, nnzU, nnzUc);

    double *d_Lx, *d_Ux, *dB, *dX, *dx, *dy, *dz;
    int *d_Lp, *d_Li, *d_Up, *d_Ui, *d_ir, *d_pc;
    CU(cudaMalloc(&d_Lp, (size_t)(n + 1) * 4));
    CU(cudaMalloc(&d_Li, (size_t)nnzLc * 4));
    CU(cudaMalloc(&d_Lx, (size_t)nnzLc * 8));
    CU(cudaMalloc(&d_Up, (size_t)(n + 1) * 4));
    CU(cudaMalloc(&d_Ui, (size_t)nnzUc * 4));
    CU(cudaMalloc(&d_Ux, (size_t)nnzUc * 8));
    CU(cudaMalloc(&d_ir, (size_t)n * 4));
    CU(cudaMalloc(&d_pc, (size_t)n * 4));
    CU(cudaMalloc(&dB, (size_t)n * nrhs * 8));
    CU(cudaMalloc(&dX, (size_t)n * nrhs * 8));
    CU(cudaMalloc(&dx, (size_t)n * nrhs * 8));
    CU(cudaMalloc(&dy, (size_t)n * nrhs * 8));
    CU(cudaMalloc(&dz, (size_t)n * nrhs * 8));
    CU(cudaMemcpy(d_Lp, Lp_c.data(), (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Li, Li_c.data(), (size_t)nnzLc * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Lx, Lx_c.data(), (size_t)nnzLc * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Up, Up_c.data(), (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ui, Ui_c.data(), (size_t)nnzUc * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ux, Ux_c.data(), (size_t)nnzUc * 8, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dB, B.data(), (size_t)n * nrhs * 8, cudaMemcpyHostToDevice));
    {
        std::vector<int> ir(n);
        for (int i = 0; i < n; i++)
            ir[i] = lu.ir[i];
        CU(cudaMemcpy(d_ir, ir.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_pc, lu.pc, (size_t)n * 4, cudaMemcpyHostToDevice));
    }

    cusparseHandle_t h;
    CK(cusparseCreate(&h));

    /* L: CSC, unit lower. U: CSC, upper. */
    cusparseSpMatDescr_t matL, matU;
    CK(cusparseCreateCsr(&matL, n, n, nnzLc, d_Lp, d_Li, d_Lx,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CK(cusparseCreateCsr(&matU, n, n, nnzUc, d_Up, d_Ui, d_Ux,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t fillL = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fillU = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diagU = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diagN = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fillL,
                                 sizeof(fillL)));
    CK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagU,
                                 sizeof(diagU)));
    CK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &fillU,
                                 sizeof(fillU)));
    CK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &diagN,
                                 sizeof(diagN)));

    cusparseDnMatDescr_t dXm, dYm, dZm;
    CK(cusparseCreateDnMat(&dXm, n, nrhs, n, dx, CUDA_R_64F,
                           CUSPARSE_ORDER_COL));
    CK(cusparseCreateDnMat(&dYm, n, nrhs, n, dy, CUDA_R_64F,
                           CUSPARSE_ORDER_COL));
    CK(cusparseCreateDnMat(&dZm, n, nrhs, n, dz, CUDA_R_64F,
                           CUSPARSE_ORDER_COL));

    const double alpha = 1.0;
    cusparseSpSMDescr_t smL, smU;
    CK(cusparseSpSM_createDescr(&smL));
    CK(cusparseSpSM_createDescr(&smU));
    size_t bufL = 0, bufU = 0;
    CK(cusparseSpSM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                               dXm, dYm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT,
                               smL, &bufL));
    CK(cusparseSpSM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU,
                               dYm, dZm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT,
                               smU, &bufU));
    /* Each descriptor needs its OWN analysis buffer: cuSPARSE keeps state in
     * it, so sharing one clobbers the other factorization. */
    void* dworkL = NULL;
    void* dworkU = NULL;
    CU(cudaMalloc(&dworkL, bufL));
    CU(cudaMalloc(&dworkU, bufU));
    CK(cusparseSpSM_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL, dXm,
                             dYm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, smL,
                             dworkL));
    CK(cusparseSpSM_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, dYm,
                             dZm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, smU,
                             dworkU));
    printf("SpSM analysis done (buffers %zu / %zu bytes)\n", bufL, bufU);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int bblocks = (n * nrhs + threads - 1) / threads;

    /* One full solve: gather by perm_r, L, U, scatter by perm_c. */
    int stage = 3;
    if (const char* e = getenv("SB_STAGE"))
        stage = atoi(e);
    auto solve_once = [&]() {
        gather_k<<<bblocks, threads>>>(dB, d_ir, dx, n, nrhs);
        if (stage < 1)
            goto done;
        CK(cusparseSpSM_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                              dXm, dYm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT,
                              smL));
        if (stage < 2)
            goto done;
        CK(cusparseSpSM_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU,
                              dYm, dZm, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT,
                              smU));
        if (stage < 3)
            goto done;
    done:
        scatter_k<<<bblocks, threads>>>(dz, d_pc, dX, n, nrhs);
    };

    solve_once();
    CU(cudaDeviceSynchronize());
    std::vector<double> Xg((size_t)n * nrhs);
    CU(cudaMemcpy(Xg.data(), dX, (size_t)n * nrhs * 8, cudaMemcpyDeviceToHost));
    double worst = 0.0;
    int worst_c = -1;
    for (int c = 0; c < nrhs; c++) {
        for (int i = 0; i < n; i++) {
            double e = fabs(Xg[(size_t)c * n + i] - Xcpu[(size_t)c * n + i]);
            if (e > worst) {
                worst = e;
                worst_c = c;
            }
        }
    }
    printf("gpu vs cpu (nrhs=%d): max err %.3e (col %d)\n", nrhs, worst, worst_c);
    {
        double mx = 0.0;
        for (int i = 0; i < n; i++)
            mx = fmax(mx, fabs(Xcpu[i]));
        printf("  (reference solution magnitude %.3e)\n", mx);
    }

    /* ---- timing ---- */
    cudaEvent_t t0, t1;
    CU(cudaEventCreate(&t0));
    CU(cudaEventCreate(&t1));
    CU(cudaEventRecord(t0));
    for (int r = 0; r < reps; r++)
        solve_once();
    CU(cudaEventRecord(t1));
    CU(cudaDeviceSynchronize());
    double mgpu = now_ms(t0, t1) / reps;

    CU(cudaEventRecord(t0));
    for (int r = 0; r < reps; r++) {
        for (int c = 0; c < nrhs; c++)
            sb_lu_solve(&lu, &B[(size_t)c * n], &Xcpu[(size_t)c * n],
                        work.data());
    }
    CU(cudaEventRecord(t1));
    double mcpu = now_ms(t0, t1) / reps;
    printf("per solve-batch: gpu %.3f ms  cpu %.3f ms  speedup %.2fx\n", mgpu,
           mcpu, mcpu / mgpu);

    int bad = (worst > 1e-9) || (e_ref > 1e-9);
    printf(bad ? "GPU SOLVE: FAIL\n" : "GPU SOLVE: OK\n");
    return bad;
}

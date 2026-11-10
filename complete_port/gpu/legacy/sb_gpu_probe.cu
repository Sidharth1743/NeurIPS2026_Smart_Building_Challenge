/* GPU-kernel-design probe: (1) the dependency-level structure of the baked
 * triangular factors (decides level-scheduled vs partitioned SpTRSV), and
 * (2) whether the cuSPARSE SpSM matrix attributes actually land.
 * usage: sb_gpu_probe <blob.sbb> */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

extern "C" {
#include "sb_baked.h"
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;
    int n = (int)sb_i64(&b, "lu_L_shape")[0];
    int nnzL = (int)sb_find(&b, "lu_L_data")->count;
    int nnzU = (int)sb_find(&b, "lu_U_data")->count;
    const int* Lp = sb_i32(&b, "lu_L_indptr");
    const int* Li = sb_i32(&b, "lu_L_indices");
    const int* Up = sb_i32(&b, "lu_U_indptr");
    const int* Ui = sb_i32(&b, "lu_U_indices");
    printf("n=%d nnzL=%d nnzU=%d\n", n, nnzL, nnzU);

    /* CSR with the diagonal dropped (unit diag implied) -- what a triangular
     * kernel would consume. Rows come out sorted by column. */
    auto csc_to_csr = [&](const int* Ap, const int* Ai, int drop_diag,
                          std::vector<int>& Bp, std::vector<int>& Bi) {
        std::vector<int> cnt(n, 0);
        for (int j = 0; j < n; j++)
            for (int q = Ap[j]; q < Ap[j + 1]; q++)
                if (!(drop_diag && Ai[q] == j)) cnt[Ai[q]]++;
        Bp.assign(n + 1, 0);
        for (int i = 0; i < n; i++) Bp[i + 1] = Bp[i] + cnt[i];
        std::vector<int> nxt(Bp.begin(), Bp.end() - 1);
        Bi.assign(Bp[n], 0);
        for (int j = 0; j < n; j++)
            for (int q = Ap[j]; q < Ap[q == 0 ? j + 1 : j + 1]; q++) {
                int i = Ai[q];
                if (drop_diag && i == j) continue;
                Bi[nxt[i]++] = j;
            }
    };
    std::vector<int> Lp_c, Li_c, Up_c, Ui_c;
    csc_to_csr(Lp, Li, 1, Lp_c, Li_c);
    csc_to_csr(Up, Ui, 0, Up_c, Ui_c);
    printf("CSR: nnzL=%d nnzU=%d\n", Lp_c[n], Up_c[n]);

    /* ---- level structure ---- */
    /* forward L: level(i) = 1 + max(level(j)) over the strictly-lower row
     * entries -> a single increasing i pass is a valid topological order. */
    std::vector<int> levL(n, 0), levU(n, 0);
    int maxL = 0;
    for (int i = 0; i < n; i++) {
        int best = -1;
        for (int p = Lp_c[i]; p < Lp_c[i + 1]; p++) {
            int j = Li_c[p];
            if (j < i && levL[j] > best) best = levL[j];
        }
        levL[i] = best + 1;
        if (levL[i] > maxL) maxL = levL[i];
    }
    /* backward U: z(j) depends on z(i), i > j -> single decreasing pass. */
    int maxU = 0;
    for (int i = n - 1; i >= 0; i--) {
        int best = -1;
        for (int p = Up_c[i]; p < Up_c[i + 1]; p++) {
            int j = Ui_c[p];
            if (j > i && levU[j] > best) best = levU[j];
        }
        levU[i] = best + 1;
        if (levU[i] > maxU) maxU = levU[i];
    }
    auto stats = [&](const std::vector<int>& lev, int mx, const char* nm) {
        std::vector<long> hist(mx + 2, 0);
        for (int i = 0; i < n; i++) hist[lev[i]]++;
        long w = 0;
        int widest = 0;
        for (int l = 0; l <= mx; l++)
            if (hist[l] > w) { w = hist[l]; widest = l; }
        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += lev[i];
        printf("%s: levels=%d  widest=%ld (level %d)  mean level=%.1f\n", nm, mx + 1,
               w, widest, mean / n);
        printf("   first 12 level widths:");
        for (int l = 0; l < 12 && l <= mx; l++) printf(" %ld", hist[l]);
        printf("\n");
    };
    stats(levL, maxL, "L (forward)");
    stats(levU, maxU, "U (backward)");

    /* ---- cuSPARSE attribute round-trip ---- */
    double* dLx = NULL; int* dLp = NULL; int* dLi = NULL;
    cudaMalloc(&dLp, (size_t)(n + 1) * 4);
    cudaMalloc(&dLi, (size_t)Lp_c[n] * 4);
    cudaMalloc(&dLx, (size_t)Lp_c[n] * 8);
    cudaMemcpy(dLp, Lp_c.data(), (size_t)(n + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dLi, Li_c.data(), (size_t)Lp_c[n] * 4, cudaMemcpyHostToDevice);
    cudaMemset(dLx, 0, (size_t)Lp_c[n] * 8);
    cusparseHandle_t h;
    cusparseCreate(&h);
    cusparseSpMatDescr_t mat;
    cusparseStatus_t st = cusparseCreateCsr(&mat, n, n, Lp_c[n], dLp, dLi, dLx,
                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    printf("createCsr: %d\n", (int)st);
    cusparseFillMode_t f = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t d = CUSPARSE_DIAG_TYPE_UNIT;
    printf("set fill=%d set diag=%d\n",
           (int)cusparseSpMatSetAttribute(mat, CUSPARSE_SPMAT_FILL_MODE, &f, sizeof(f)),
           (int)cusparseSpMatSetAttribute(mat, CUSPARSE_SPMAT_DIAG_TYPE, &d, sizeof(d)));
    cusparseFillMode_t gf = (cusparseFillMode_t)-1;
    cusparseDiagType_t gd = (cusparseDiagType_t)-1;
    printf("get fill=%d diag=%d\n",
           (int)cusparseSpMatGetAttribute(mat, CUSPARSE_SPMAT_FILL_MODE, &gf, sizeof(gf)),
           (int)cusparseSpMatGetAttribute(mat, CUSPARSE_SPMAT_DIAG_TYPE, &gd, sizeof(gd)));
    printf("  values: fill=%d (expect %d) diag=%d (expect %d)\n", (int)gf,
           (int)CUSPARSE_FILL_MODE_LOWER, (int)gd, (int)CUSPARSE_DIAG_TYPE_UNIT);
    return 0;
}

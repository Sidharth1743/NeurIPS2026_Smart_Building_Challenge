/* Real-matrix Cholesky test: factor A[perm][:,perm] from the baked Au_p and
 * check the solve residual ||A x - b|| / ||b||, plus nnz(L). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sb_baked.h"
#include "sb_chol.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <blob.sbb>\n", argv[0]); return 2; }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;

    int n = (int)(sb_find(&b, "perm")->count);
    const int* Ap = sb_i32(&b, "Au_p_indptr");
    const int* Ai = sb_i32(&b, "Au_p_indices");
    const double* Ax = sb_f64(&b, "Au_p_data");
    const int* perm = sb_i32(&b, "perm");

    double t0 = now();
    SbCsc L;
    if (sb_chol(n, Ap, Ai, Ax, &L) != 0) { printf("CHOL BIG: FAIL (not SPD)\n"); return 1; }
    double tf = now() - t0;

    double* bvec = (double*)malloc(sizeof(double) * n);
    double* bp = (double*)malloc(sizeof(double) * n);
    double* y = (double*)malloc(sizeof(double) * n);
    double* xp = (double*)malloc(sizeof(double) * n);
    double* x = (double*)malloc(sizeof(double) * n);
    for (int j = 0; j < n; j++) bvec[j] = sin(0.1 * j + 1.0);
    for (int i = 0; i < n; i++) bp[i] = bvec[perm[i]];

    t0 = now();
    sb_lsolve(&L, bp, y);
    sb_ltsolve(&L, y, xp);
    double ts = now() - t0;
    for (int i = 0; i < n; i++) x[perm[i]] = xp[i];

    /* residual via the symmetric product of Au_p */
    double* r = (double*)calloc(n, sizeof(double));
    for (int j = 0; j < n; j++)
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            int i = Ai[p];
            r[i] += Ax[p] * xp[j];
            if (i != j) r[j] += Ax[p] * xp[i];
        }
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; j++) {
        num = fmax(num, fabs(r[j] - bp[j]));
        den = fmax(den, fabs(bp[j]));
    }

    printf("n=%d nnzL=%lld factor=%.2fs solve=%.2fs rel_resid=%.3e\n",
           n, (long long)L.nnz, tf, ts, num / den);

    free(bvec); free(bp); free(y); free(xp); free(x); free(r);
    sb_csc_free(&L);
    sb_close(&b);
    if (num / den > 1e-10) { printf("CHOL BIG: FAIL\n"); return 1; }
    printf("CHOL BIG: OK\n");
    return 0;
}

/* Small-matrix Cholesky test against a dense reference.
 * A = tridiag(-1, 4, -1), n=5 (SPD). Checks L L^T == A and A x == b. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sb_chol.h"

int main(void) {
    const int n = 5;
    /* upper-triangular CSC: column j has rows [j-1 (if j>0, -1), j (4)] */
    int Ap[6] = {0, 1, 3, 5, 7, 9};
    int Ai[9] = {0, 0, 1, 1, 2, 2, 3, 3, 4};
    double Ax[9] = {4, -1, 4, -1, 4, -1, 4, -1, 4};

    SbCsc L;
    if (sb_chol(n, Ap, Ai, Ax, &L) != 0) { printf("cholesky failed\n"); return 1; }

    /* dense A */
    double A[5][5]; memset(A, 0, sizeof(A));
    for (int j = 0; j < n; j++)
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            A[Ai[p]][j] = Ax[p];
            A[j][Ai[p]] = Ax[p];
        }
    /* dense L L^T */
    double LL[5][5]; memset(LL, 0, sizeof(LL));
    for (int j = 0; j < n; j++)
        for (int p = L.p[j]; p < L.p[j + 1]; p++)
            for (int q = L.p[j]; q < L.p[j + 1]; q++)
                LL[L.i[p]][L.i[q]] += L.x[p] * L.x[q];
    double err = 0.0;
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++)
        err = fmax(err, fabs(A[a][b] - LL[a][b]));
    printf("nnzL=%lld  max|A - LL^T| = %.3e\n", (long long)L.nnz, err);

    double b[5] = {1, 2, 3, 4, 5};
    double x[5];
    double y[5];
    sb_lsolve(&L, b, y);
    sb_ltsolve(&L, y, x);
    double res = 0.0;
    for (int a = 0; a < n; a++) {
        double Ax_ = 0.0;
        for (int c = 0; c < n; c++) Ax_ += A[a][c] * x[c];
        res = fmax(res, fabs(Ax_ - b[a]));
    }
    printf("x = [%.6f %.6f %.6f %.6f %.6f]  max|Ax-b| = %.3e\n",
           x[0], x[1], x[2], x[3], x[4], res);
    sb_csc_free(&L);
    if (err > 1e-12 || res > 1e-12) { printf("CHOL TEST: FAIL\n"); return 1; }
    printf("CHOL TEST: OK\n");
    return 0;
}

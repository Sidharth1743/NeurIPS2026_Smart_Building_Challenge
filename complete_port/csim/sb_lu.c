#include "sb_lu.h"

#include <stdlib.h>

int sb_lu_build(SbLu* lu, int n,
                const int* Lp, const int* Li, const double* Lx,
                const int* Up, const int* Ui, const double* Ux,
                const int* perm_r, const int* perm_c) {
    lu->n = n;
    lu->Lp = Lp; lu->Li = Li; lu->Lx = Lx;
    lu->Up = Up; lu->Ui = Ui; lu->Ux = Ux;
    lu->Ldiag = (int*)malloc(sizeof(int) * n);
    lu->Udiag = (int*)malloc(sizeof(int) * n);
    lu->ir = (int*)malloc(sizeof(int) * n);
    lu->pc = (int*)malloc(sizeof(int) * n);
    if (!lu->Ldiag || !lu->Udiag || !lu->ir || !lu->pc) return -1;

    /* locate the diagonal entry in every column of L and U */
    for (int j = 0; j < n; j++) {
        lu->Ldiag[j] = -1;
        for (int p = Lp[j]; p < Lp[j + 1]; p++)
            if (Li[p] == j) { lu->Ldiag[j] = p; break; }
        lu->Udiag[j] = -1;
        for (int p = Up[j]; p < Up[j + 1]; p++)
            if (Ui[p] == j) { lu->Udiag[j] = p; break; }
        if (lu->Ldiag[j] < 0 || lu->Udiag[j] < 0) return -1;
    }
    for (int i = 0; i < n; i++) lu->ir[perm_r[i]] = i;   /* inv(perm_r) */
    for (int j = 0; j < n; j++) lu->pc[j] = perm_c[j];
    return 0;
}

void sb_lu_free(SbLu* lu) {
    free(lu->Ldiag); free(lu->Udiag); free(lu->ir); free(lu->pc);
    lu->Ldiag = NULL; lu->Udiag = NULL; lu->ir = NULL; lu->pc = NULL;
}

void sb_lu_solve(const SbLu* lu, const double* b, double* x, double* work) {
    int n = lu->n;
    /* y = b[ir] */
    for (int i = 0; i < n; i++) work[i] = b[lu->ir[i]];
    /* forward: L y = work (unit lower, CSC) */
    for (int j = 0; j < n; j++) {
        double yj = work[j];
        int d = lu->Ldiag[j];
        for (int p = lu->Lp[j]; p < lu->Lp[j + 1]; p++) {
            if (p == d) continue;
            work[lu->Li[p]] -= lu->Lx[p] * yj;
        }
    }
    /* back: U z = y (upper, CSC), accumulate in place */
    for (int j = n - 1; j >= 0; j--) {
        double zj = work[j] / lu->Ux[lu->Udiag[j]];
        work[j] = zj;
        int d = lu->Udiag[j];
        for (int p = lu->Up[j]; p < lu->Up[j + 1]; p++) {
            if (p == d) continue;
            work[lu->Ui[p]] -= lu->Ux[p] * zj;
        }
    }
    /* x = work[perm_c] */
    for (int k = 0; k < n; k++) x[k] = work[lu->pc[k]];
}

void sb_lu_solve_multi(const SbLu* lu, const double* B, double* X,
                       double* work, int G) {
    const int n = lu->n;
    /* Work is (n, G) row-major: a push into row i touches G contiguous doubles,
     * so the batch costs one cache line per row rather than G scattered ones. */
    for (int i = 0; i < n; i++) {
        const int r = lu->ir[i];
        double* wi = &work[(size_t)i * G];
        for (int g = 0; g < G; g++) wi[g] = B[(size_t)g * n + r];
    }
    /* Forward: L (unit lower, CSC). Same push order as sb_lu_solve. */
    for (int j = 0; j < n; j++) {
        const int d = lu->Ldiag[j];
        double* wj = &work[(size_t)j * G];
        for (int p = lu->Lp[j]; p < lu->Lp[j + 1]; p++) {
            if (p == d) continue;
            const int i = lu->Li[p];
            const double lx = lu->Lx[p];
            double* wi = &work[(size_t)i * G];
            for (int g = 0; g < G; g++) wi[g] -= lx * wj[g];
        }
    }
    /* Backward: U (upper, CSC). */
    for (int j = n - 1; j >= 0; j--) {
        double* wj = &work[(size_t)j * G];
        const int d = lu->Udiag[j];
        const double ujj = lu->Ux[d];
        for (int g = 0; g < G; g++) wj[g] /= ujj;
        for (int p = lu->Up[j]; p < lu->Up[j + 1]; p++) {
            if (p == d) continue;
            const int i = lu->Ui[p];
            const double ux = lu->Ux[p];
            double* wi = &work[(size_t)i * G];
            for (int g = 0; g < G; g++) wi[g] -= ux * wj[g];
        }
    }
    for (int k = 0; k < n; k++) {
        const int r = lu->pc[k];
        const double* wr = &work[(size_t)r * G];
        for (int g = 0; g < G; g++) X[(size_t)g * n + k] = wr[g];
    }
}

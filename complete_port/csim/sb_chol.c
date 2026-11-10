#include "sb_chol.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Elimination tree of an upper-triangular matrix (CSC, rows sorted). */
static void etree(int n, const int* Ap, const int* Ai, int* parent, int* ancestor) {
    for (int k = 0; k < n; k++) { parent[k] = -1; ancestor[k] = -1; }
    for (int k = 0; k < n; k++) {
        for (int p = Ap[k]; p < Ap[k + 1]; p++) {
            int i = Ai[p];
            while (i != -1 && i < k) {
                int inext = ancestor[i];
                ancestor[i] = k;
                if (inext == -1) parent[i] = k;
                i = inext;
            }
        }
    }
}

/* Pattern of row k of L: fills s[top..n-1] with rows i<k such that L(k,i)!=0,
 * in a valid topological order. w is a mark workspace (init -1), reused. */
static int ereach(int n, const int* Ap, const int* Ai, int k,
                  const int* parent, int* s, int* w) {
    int top = n;
    w[k] = k;
    for (int p = Ap[k]; p < Ap[k + 1]; p++) {
        int i = Ai[p];
        if (i > k) continue;
        int len = 0;
        while (w[i] != k) {
            s[len++] = i;
            w[i] = k;
            i = parent[i];
        }
        while (len > 0) s[--top] = s[--len];
    }
    return top;
}

int sb_chol(int n, const int* Ap, const int* Ai, const double* Ax, SbCsc* L) {
    if (n <= 0) return -1;
    int* parent = (int*)malloc(sizeof(int) * n);
    int* ancestor = (int*)malloc(sizeof(int) * n);
    int* count = (int*)calloc(n, sizeof(int));   /* column counts of L */
    int* w = (int*)malloc(sizeof(int) * n);
    int* s = (int*)malloc(sizeof(int) * n);
    int* cn = (int*)malloc(sizeof(int) * n);     /* next free slot per column */
    double* x = (double*)calloc(n, sizeof(double));
    if (!parent || !ancestor || !count || !w || !s || !cn || !x) return -1;

    etree(n, Ap, Ai, parent, ancestor);
    for (int i = 0; i < n; i++) w[i] = -1;

    /* Symbolic pass: column counts via ereach of every row. */
    for (int k = 0; k < n; k++) {
        int top = ereach(n, Ap, Ai, k, parent, s, w);
        for (int t = top; t < n; t++) count[s[t]]++;
        count[k]++;   /* diagonal */
    }
    int64_t nnz = 0;
    for (int j = 0; j < n; j++) nnz += count[j];

    L->n = n;
    L->nnz = nnz;
    L->p = (int*)malloc(sizeof(int) * (n + 1));
    L->i = (int*)malloc(sizeof(int) * (nnz > 0 ? nnz : 1));
    L->x = (double*)malloc(sizeof(double) * (nnz > 0 ? nnz : 1));
    if (!L->p || !L->i || !L->x) return -1;
    L->p[0] = 0;
    for (int j = 0; j < n; j++) L->p[j + 1] = L->p[j] + count[j];

    /* Numeric pass (up-looking). */
    for (int i = 0; i < n; i++) w[i] = -1;
    for (int k = 0; k < n; k++) {
        int top = ereach(n, Ap, Ai, k, parent, s, w);
        /* scatter column k of the upper triangle */
        x[k] = 0.0;
        for (int p = Ap[k]; p < Ap[k + 1]; p++) {
            int i = Ai[p];
            if (i <= k) x[i] = Ax[p];
        }
        double d = x[k];
        x[k] = 0.0;
        for (int t = top; t < n; t++) {
            int i = s[t];
            double lki = x[i] / L->x[L->p[i]];   /* L(i,i) at column start */
            x[i] = 0.0;
            for (int p = L->p[i] + 1; p < cn[i]; p++)
                x[L->i[p]] -= L->x[p] * lki;
            d -= lki * lki;                      /* subtract L(k,i)^2 from diag */
            int q = cn[i]++;
            L->i[q] = k;
            L->x[q] = lki;
        }
        if (!(d > 0.0)) {
            free(parent); free(ancestor); free(count); free(w); free(s); free(cn); free(x);
            sb_csc_free(L);
            return -1;
        }
        int q = L->p[k];
        cn[k] = q + 1;
        L->i[q] = k;
        L->x[q] = sqrt(d);
    }
    for (int j = 0; j < n; j++)
        if (cn[j] != L->p[j + 1]) { /* internal sanity: counts must be exact */
            free(parent); free(ancestor); free(count); free(w); free(s); free(cn); free(x);
            sb_csc_free(L);
            return -1;
        }
    free(parent); free(ancestor); free(count); free(w); free(s); free(cn); free(x);
    return 0;
}

void sb_csc_free(SbCsc* L) {
    free(L->p); free(L->i); free(L->x);
    L->p = NULL; L->i = NULL; L->x = NULL; L->nnz = 0; L->n = 0;
}

void sb_lsolve(const SbCsc* L, const double* b, double* x) {
    int n = L->n;
    for (int i = 0; i < n; i++) x[i] = b[i];
    for (int j = 0; j < n; j++) {
        double xj = x[j] / L->x[L->p[j]];
        x[j] = xj;
        for (int p = L->p[j] + 1; p < L->p[j + 1]; p++)
            x[L->i[p]] -= L->x[p] * xj;
    }
}

void sb_ltsolve(const SbCsc* L, const double* b, double* x) {
    int n = L->n;
    for (int i = 0; i < n; i++) x[i] = b[i];
    for (int j = n - 1; j >= 0; j--) {
        double xj = x[j];
        for (int p = L->p[j] + 1; p < L->p[j + 1]; p++)
            xj -= L->x[p] * x[L->i[p]];
        x[j] = xj / L->x[L->p[j]];
    }
}

int sb_chol_solve(int n, const int* Ap, const int* Ai, const double* Ax,
                  const double* b, double* x) {
    SbCsc L;
    if (sb_chol(n, Ap, Ai, Ax, &L) != 0) return -1;
    double* y = (double*)malloc(sizeof(double) * n);
    sb_lsolve(&L, b, y);
    sb_ltsolve(&L, y, x);
    free(y);
    sb_csc_free(&L);
    return 0;
}

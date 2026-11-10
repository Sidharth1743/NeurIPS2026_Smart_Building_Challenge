/* Sparse SPD Cholesky for A[perm][:,perm]: L L^T = A.
 * A is passed as its upper-triangular part in CSC with rows sorted ascending.
 * L is returned as CSC: column j occupies p[j]..p[j+1]-1, diagonal first at
 * p[j] (row j), then off-diagonals with strictly increasing row index.
 * Clean-room up-looking algorithm (elimination tree + ereach pattern). */
#ifndef SB_CHOL_H
#define SB_CHOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    int64_t nnz;
    int* p;      /* n+1 */
    int* i;      /* nnz */
    double* x;   /* nnz */
} SbCsc;

/* Returns 0 on success, -1 if A is not positive definite. */
int  sb_chol(int n, const int* Ap, const int* Ai, const double* Ax, SbCsc* L);
void sb_csc_free(SbCsc* L);
/* L x = b (forward substitution). */
void sb_lsolve(const SbCsc* L, const double* b, double* x);
/* L^T x = b (back substitution). */
void sb_ltsolve(const SbCsc* L, const double* b, double* x);

/* Convenience: factor A and solve A x = b in one place. A is upper CSC. */
int  sb_chol_solve(int n, const int* Ap, const int* Ai, const double* Ax,
                   const double* b, double* x);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_CHOL_H */

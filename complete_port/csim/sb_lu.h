/* Replay of a baked SuperLU factorization (CSC L/U + permutations).
 * Solve A x = b as:  y[i] = b[inv_perm_r[i]];  L z = y;  U w = z;  x = w[perm_c].
 * L is unit lower, U is upper; both in CSC (any row order within a column).
 * No SuperLU is linked: the factors are data. */
#ifndef SB_LU_H
#define SB_LU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    const int* Lp; const int* Li; const double* Lx; int* Ldiag;
    const int* Up; const int* Ui; const double* Ux; int* Udiag;
    int* ir;   /* inv(perm_r) */
    int* pc;   /* perm_c    */
} SbLu;

/* Builds the solver from baked arrays (borrowed pointers). Returns 0 on
 * success. `perm_r`/`perm_c` are length n; Ldiag/Udiag/ir are allocated. */
int  sb_lu_build(SbLu* lu, int n,
                 const int* Lp, const int* Li, const double* Lx,
                 const int* Up, const int* Ui, const double* Ux,
                 const int* perm_r, const int* perm_c);
void sb_lu_free(SbLu* lu);

/* Solves A x = b. `work` is caller-provided scratch of length n. */
void sb_lu_solve(const SbLu* lu, const double* b, double* x, double* work);

/* Solves A X = B for G right-hand sides at once, streaming L/U once.
 * B and X are (G, n) row-major (one row per RHS). `work` is caller-provided
 * scratch of length (size_t)n * G. Each column is solved with exactly the
 * same operation order as sb_lu_solve, so the result is BIT-IDENTICAL to G
 * sequential sb_lu_solve calls; only the memory traffic is shared. */
void sb_lu_solve_multi(const SbLu* lu, const double* B, double* X,
                       double* work, int G);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_LU_H */

/* One backward-Euler FDM timestep, mirroring
 * FiniteDifferenceSolver.solve + _add_air_side, using the baked LU replay. */
#ifndef SB_FDM_H
#define SB_FDM_H

#include "sb_baked.h"
#include "sb_lu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nr, nc, n_cvs, n_solved, k;

    const int64_t* solved;      /* n_solved */
    const double* capacitance_rate;
    const double* dirichlet;
    const double* mass_coupling;
    const double* mass_retained;
    const double* mass_gained;
    const double* horizontal;   /* n_cvs */
    const double* vertical;     /* n_cvs */
    const double* conductivity; /* n_cvs */
    const double* amb_face_0;   /* n_cvs */
    const double* amb_face_1;   /* n_cvs */

    double* mass_temp;          /* n_solved, mutable */
    double* temp;               /* n_cvs, mutable */

    SbLu lu;
    int lu_ready;

    /* air-side coupling: applied is n_cvs x k (CSC), sensed is k x n_cvs (CSR) */
    const int* ap_p; const int* ap_i; const double* ap_x;
    const int* sn_p; const int* sn_i; const double* sn_x;
    double* sensed_response;    /* k x k = sensed @ A^-1 @ applied */

    /* scratch */
    double *forcing, *y, *yr, *r, *exposure, *power, *q, *mat, *work;
} SbFdm;

int  sb_fdm_init(SbFdm* s, const SbBlob* b);
void sb_fdm_free(SbFdm* s);

/* ambient W/K per solved CV for a convection coefficient h. */
void sb_fdm_exposure(const SbFdm* s, double h, double* out);

/* Advance one step. input_q is the full field (n_cvs). air_g/air_T are length
 * k or NULL for none. */
void sb_fdm_step(SbFdm* s, const double* input_q, double ambient, double h,
                 const double* air_g, const double* air_T);

/* Advance G environments that share this factorization in ONE step, with the
 * two triangular solves batched over the group's right-hand sides. Every
 * per-environment operation keeps its exact order, and the solve is
 * bit-identical to sb_lu_solve per column, so the result is BIT-IDENTICAL to G
 * sequential sb_fdm_step calls. f[0] supplies the shared factor / coupling
 * matrices. Scratch: rhs, sol of length n*G each, work of length n*G. */
void sb_fdm_step_batch(SbFdm** f, int G, const double* const* input_q,
                       const double* ambient, const double* h,
                       const double* const* air_g, const double* const* air_T,
                       double* rhs, double* sol, double* work);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_FDM_H */

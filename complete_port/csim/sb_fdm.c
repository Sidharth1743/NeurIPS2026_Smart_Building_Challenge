#include "sb_fdm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Dense k x k solve M q = b (M row-major, modified in place), partial pivot. */
static int dense_solve(int k, double* M, double* b) {
    for (int c = 0; c < k; c++) {
        int piv = c;
        double best = fabs(M[c * k + c]);
        for (int r = c + 1; r < k; r++) {
            double v = fabs(M[r * k + c]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return -1;
        if (piv != c) {
            for (int j = c; j < k; j++) {
                double t = M[c * k + j]; M[c * k + j] = M[piv * k + j]; M[piv * k + j] = t;
            }
            double t = b[c]; b[c] = b[piv]; b[piv] = t;
        }
        double d = M[c * k + c];
        for (int r = c + 1; r < k; r++) {
            double f = M[r * k + c] / d;
            if (f == 0.0) continue;
            for (int j = c; j < k; j++) M[r * k + j] -= f * M[c * k + j];
            b[r] -= f * b[c];
        }
    }
    for (int r = k - 1; r >= 0; r--) {
        double s = b[r];
        for (int j = r + 1; j < k; j++) s -= M[r * k + j] * b[j];
        b[r] = s / M[r * k + r];
    }
    return 0;
}

int sb_fdm_init(SbFdm* s, const SbBlob* b) {
    memset(s, 0, sizeof(*s));
    const int64_t* grid = sb_i64(b, "grid");
    s->nr = (int)grid[0];
    s->nc = (int)grid[1];
    s->n_cvs = s->nr * s->nc;
    s->n_solved = (int)(sb_find(b, "solved")->count);
    const int64_t* sn_shape = sb_i64(b, "sensed_shape");  /* [k, n_solved] */
    s->k = sn_shape ? (int)sn_shape[0] : 0;

    s->solved = sb_i64(b, "solved");
    s->capacitance_rate = sb_f64(b, "capacitance_rate");
    s->dirichlet = sb_f64(b, "dirichlet_conductance");
    s->mass_coupling = sb_f64(b, "mass_coupling");
    s->mass_retained = sb_f64(b, "mass_retained");
    s->mass_gained = sb_f64(b, "mass_gained");
    s->horizontal = sb_f64(b, "horizontal");
    s->vertical = sb_f64(b, "vertical");
    s->conductivity = sb_f64(b, "solver_conductivity");
    s->amb_face_0 = sb_f64(b, "ambient_face_areas_0");
    s->amb_face_1 = sb_f64(b, "ambient_face_areas_1");

    int n = s->n_solved, k = s->k;
    s->mass_temp = (double*)malloc(sizeof(double) * n);
    s->temp = (double*)malloc(sizeof(double) * s->n_cvs);
    s->forcing = (double*)malloc(sizeof(double) * n);
    s->y = (double*)malloc(sizeof(double) * n);
    s->yr = (double*)malloc(sizeof(double) * n);
    s->r = (double*)malloc(sizeof(double) * n);
    s->exposure = (double*)malloc(sizeof(double) * n);
    s->work = (double*)malloc(sizeof(double) * n);
    s->power = (double*)malloc(sizeof(double) * (k ? k : 1));
    s->q = (double*)malloc(sizeof(double) * (k ? k : 1));
    s->mat = (double*)malloc(sizeof(double) * (k ? k * k : 1));
    s->sensed_response = (double*)malloc(sizeof(double) * (k ? k * k : 1));
    if (!s->mass_temp || !s->temp || !s->forcing || !s->y || !s->yr ||
        !s->r || !s->exposure || !s->work || !s->power || !s->q || !s->mat ||
        !s->sensed_response) { fprintf(stderr, "sb_fdm_init: malloc\n"); return -1; }
    memcpy(s->mass_temp, sb_f64(b, "mass_temp_init"), sizeof(double) * n);
    memcpy(s->temp, sb_f64(b, "temp_init"), sizeof(double) * s->n_cvs);

    if (sb_lu_build(&s->lu, n,
                    sb_i32(b, "lu_L_indptr"), sb_i32(b, "lu_L_indices"),
                    sb_f64(b, "lu_L_data"),
                    sb_i32(b, "lu_U_indptr"), sb_i32(b, "lu_U_indices"),
                    sb_f64(b, "lu_U_data"),
                    sb_i32(b, "lu_perm_r"), sb_i32(b, "lu_perm_c")) != 0) {
        fprintf(stderr, "sb_fdm_init: lu_build\n");
        return -1;
    }
    s->lu_ready = 1;

    s->ap_p = sb_i32(b, "applied_indptr");
    s->ap_i = sb_i32(b, "applied_indices");
    s->ap_x = sb_f64(b, "applied_data");
    s->sn_p = sb_i32(b, "sensed_indptr");
    s->sn_i = sb_i32(b, "sensed_indices");
    s->sn_x = sb_f64(b, "sensed_data");
    if (!s->ap_p || !s->ap_i || !s->ap_x || !s->sn_p || !s->sn_i || !s->sn_x) {
        fprintf(stderr, "sb_fdm_init: air-side sections missing\n");
        return -1;
    }

    /* S = sensed @ A^-1 @ applied (k x k), computed once */
    for (int j = 0; j < k; j++) {
        for (int t = 0; t < n; t++) s->r[t] = 0.0;
        for (int p = s->ap_p[j]; p < s->ap_p[j + 1]; p++) s->r[s->ap_i[p]] = s->ap_x[p];
        sb_lu_solve(&s->lu, s->r, s->yr, s->work);
        for (int i = 0; i < k; i++) {
            double acc = 0.0;
            for (int p = s->sn_p[i]; p < s->sn_p[i + 1]; p++)
                acc += s->sn_x[p] * s->yr[s->sn_i[p]];
            s->sensed_response[i * k + j] = acc;
        }
    }
    return 0;
}

void sb_fdm_free(SbFdm* s) {
    if (s->lu_ready) sb_lu_free(&s->lu);
    free(s->mass_temp); free(s->temp); free(s->forcing); free(s->y);
    free(s->yr); free(s->r); free(s->exposure); free(s->work);
    free(s->power); free(s->q); free(s->mat); free(s->sensed_response);
    memset(s, 0, sizeof(*s));
}

void sb_fdm_exposure(const SbFdm* s, double h, double* out) {
    int n = s->n_solved;
    const int64_t* solved = s->solved;
    for (int t = 0; t < n; t++) {
        int64_t p = solved[t];
        double e = 0.0;
        double a0 = s->amb_face_0[p];
        if (a0 > 0.0)
            e += h * a0 / (1.0 + h * 0.5 * s->horizontal[p] / s->conductivity[p]);
        double a1 = s->amb_face_1[p];
        if (a1 > 0.0)
            e += h * a1 / (1.0 + h * 0.5 * s->vertical[p] / s->conductivity[p]);
        out[t] = e;
    }
}

void sb_fdm_step(SbFdm* s, const double* input_q, double ambient, double h,
                 const double* air_g, const double* air_T) {
    int n = s->n_solved, k = s->k;
    sb_fdm_exposure(s, h, s->exposure);
    for (int t = 0; t < n; t++) {
        int64_t p = s->solved[t];
        s->forcing[t] = s->capacitance_rate[t] * s->temp[p]
                      + input_q[p]
                      + ambient * (s->dirichlet[t] + s->exposure[t])
                      + s->mass_coupling[t] * s->mass_temp[t];
    }
    sb_lu_solve(&s->lu, s->forcing, s->y, s->work);

    if (k > 0 && air_g) {
        int active = 0;
        for (int i = 0; i < k; i++) if (air_g[i] != 0.0) { active = 1; break; }
        if (active) {
            for (int i = 0; i < k; i++) {
                double acc = 0.0;
                for (int p = s->sn_p[i]; p < s->sn_p[i + 1]; p++)
                    acc += s->sn_x[p] * s->y[s->sn_i[p]];
                s->power[i] = air_g[i] * (air_T[i] - acc);
            }
            /* M = I + diag(g) S */
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                    s->mat[i * k + j] = (i == j ? 1.0 : 0.0)
                                      + air_g[i] * s->sensed_response[i * k + j];
            dense_solve(k, s->mat, s->power);   /* power now holds q */
            for (int t = 0; t < n; t++) s->r[t] = 0.0;
            for (int j = 0; j < k; j++) {
                double qj = s->power[j];
                if (qj == 0.0) continue;
                for (int p = s->ap_p[j]; p < s->ap_p[j + 1]; p++)
                    s->r[s->ap_i[p]] += s->ap_x[p] * qj;
            }
            sb_lu_solve(&s->lu, s->r, s->yr, s->work);
            for (int t = 0; t < n; t++) s->y[t] += s->yr[t];
        }
    }

    for (int t = 0; t < n; t++)
        s->mass_temp[t] = s->mass_temp[t] * s->mass_retained[t]
                        + s->mass_gained[t] * s->y[t];
    for (int p = 0; p < s->n_cvs; p++) s->temp[p] = ambient;
    for (int t = 0; t < n; t++) s->temp[s->solved[t]] = s->y[t];
}

void sb_fdm_step_batch(SbFdm** f, int G, const double* const* input_q,
                       const double* ambient, const double* h,
                       const double* const* air_g, const double* const* air_T,
                       double* rhs, double* sol, double* work) {
    const int n = f[0]->n_solved, k = f[0]->k;

    /* 1. exposure + forcing, packed one env per RHS column. */
    for (int g = 0; g < G; g++) {
        SbFdm* s = f[g];
        sb_fdm_exposure(s, h[g], s->exposure);
        const double* q = input_q[g];
        double* col = &rhs[(size_t)g * n];
        for (int t = 0; t < n; t++) {
            int64_t p = s->solved[t];
            col[t] = s->capacitance_rate[t] * s->temp[p] + q[p]
                   + ambient[g] * (s->dirichlet[t] + s->exposure[t])
                   + s->mass_coupling[t] * s->mass_temp[t];
        }
    }

    /* 2. one batched solve for the whole group. */
    sb_lu_solve_multi(&f[0]->lu, rhs, sol, work, G);
    for (int g = 0; g < G; g++) {
        SbFdm* s = f[g];
        const double* yc = &sol[(size_t)g * n];
        for (int t = 0; t < n; t++) s->y[t] = yc[t];
    }

    /* 3. air side, per env; inactive envs contribute an all-zero RHS. */
    int any2 = 0;
    for (int g = 0; g < G; g++) {
        SbFdm* s = f[g];
        double* col = &rhs[(size_t)g * n];
        int active = 0;
        if (k > 0 && air_g[g]) {
            for (int i = 0; i < k; i++)
                if (air_g[g][i] != 0.0) { active = 1; break; }
        }
        if (!active) {
            for (int t = 0; t < n; t++) col[t] = 0.0;
            continue;
        }
        any2 = 1;
        for (int i = 0; i < k; i++) {
            double acc = 0.0;
            for (int p = s->sn_p[i]; p < s->sn_p[i + 1]; p++)
                acc += s->sn_x[p] * s->y[s->sn_i[p]];
            s->power[i] = air_g[g][i] * (air_T[g][i] - acc);
        }
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                s->mat[i * k + j] = (i == j ? 1.0 : 0.0)
                                  + air_g[g][i] * s->sensed_response[i * k + j];
        dense_solve(k, s->mat, s->power);
        for (int t = 0; t < n; t++) col[t] = 0.0;
        for (int j = 0; j < k; j++) {
            double qj = s->power[j];
            if (qj == 0.0) continue;
            for (int p = s->ap_p[j]; p < s->ap_p[j + 1]; p++)
                col[s->ap_i[p]] += s->ap_x[p] * qj;
        }
    }
    if (any2) {
        sb_lu_solve_multi(&f[0]->lu, rhs, sol, work, G);
        for (int g = 0; g < G; g++) {
            SbFdm* s = f[g];
            const double* yc = &sol[(size_t)g * n];
            for (int t = 0; t < n; t++) s->y[t] += yc[t];
        }
    }

    /* 4. interior mass + field scatter. */
    for (int g = 0; g < G; g++) {
        SbFdm* s = f[g];
        for (int t = 0; t < n; t++)
            s->mass_temp[t] = s->mass_temp[t] * s->mass_retained[t]
                            + s->mass_gained[t] * s->y[t];
        for (int p = 0; p < s->n_cvs; p++) s->temp[p] = ambient[g];
        for (int t = 0; t < n; t++) s->temp[s->solved[t]] = s->y[t];
    }
}

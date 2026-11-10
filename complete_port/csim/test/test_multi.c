/* M5 gate: the batched multi-RHS solve must be BIT-IDENTICAL to G sequential
 * sb_lu_solve calls, and must be faster (L/U streamed once for the whole batch).
 * usage: test_multi <blob.sbb> [G] [reps] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sb_baked.h"
#include "sb_lu.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <blob.sbb> [G] [reps]\n", argv[0]);
        return 2;
    }
    int G = argc > 2 ? atoi(argv[2]) : 8;
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0)
        return 1;
    int n = (int)sb_i64(&b, "lu_L_shape")[0];
    SbLu lu;
    if (sb_lu_build(&lu, n, sb_i32(&b, "lu_L_indptr"), sb_i32(&b, "lu_L_indices"),
                    sb_f64(&b, "lu_L_data"), sb_i32(&b, "lu_U_indptr"),
                    sb_i32(&b, "lu_U_indices"), sb_f64(&b, "lu_U_data"),
                    sb_i32(&b, "lu_perm_r"), sb_i32(&b, "lu_perm_c")) != 0)
        return 1;
    const double* bref = sb_f64(&b, "solref_b");
    const double* xref = sb_f64(&b, "solref_x");

    double* B = malloc(sizeof(double) * (size_t)n * G);
    double* Xs = malloc(sizeof(double) * (size_t)n * G);
    double* Xm = malloc(sizeof(double) * (size_t)n * G);
    double* wk = malloc(sizeof(double) * (size_t)n);
    double* wm = malloc(sizeof(double) * (size_t)n * G);
    if (!B || !Xs || !Xm || !wk || !wm) return 1;
    for (int g = 0; g < G; g++)
        for (int i = 0; i < n; i++)
            B[(size_t)g * n + i] = bref[i] + (double)g * 1e-3 * cos(0.5 + 1e-4 * i)
                                 + (double)g * 1e-6 * (double)(i % 97);

    /* sequential reference */
    for (int g = 0; g < G; g++)
        sb_lu_solve(&lu, &B[(size_t)g * n], &Xs[(size_t)g * n], wk);
    double e_ref = 0.0;
    for (int i = 0; i < n; i++)
        e_ref = fmax(e_ref, fabs(Xs[i] - xref[i]));

    sb_lu_solve_multi(&lu, B, Xm, wm, G);
    long diff = 0;
    double worst = 0.0;
    for (size_t k = 0; k < (size_t)n * G; k++) {
        if (Xs[k] != Xm[k]) diff++;
        worst = fmax(worst, fabs(Xs[k] - Xm[k]));
    }
    printf("n=%d G=%d: seq vs baked xref %.3e; multi vs seq: %ld/%zu differ, "
           "max |d| = %.3e\n", n, G, e_ref, diff, (size_t)n * G, worst);

    /* timing */
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        for (int g = 0; g < G; g++)
            sb_lu_solve(&lu, &B[(size_t)g * n], &Xs[(size_t)g * n], wk);
    double tseq = (now_s() - t0) / reps;
    t0 = now_s();
    for (int r = 0; r < reps; r++)
        sb_lu_solve_multi(&lu, B, Xm, wm, G);
    double tmulti = (now_s() - t0) / reps;
    printf("  %d solves: sequential %.3f ms  batched %.3f ms  %.2fx\n", G, tseq,
           tmulti, tseq / tmulti);
    printf("  per-RHS: sequential %.3f ms  batched %.3f ms\n", tseq / G,
           tmulti / G);

    int bad = (diff != 0) || (e_ref > 1e-9);
    printf(bad ? "MULTI SOLVE: FAIL\n" : "MULTI SOLVE: OK\n");
    return bad;
}

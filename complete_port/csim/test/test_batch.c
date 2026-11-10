/* M5 gate: sb_fdm_step_batch (one batched multi-RHS solve per group per step)
 * must be BIT-IDENTICAL to G sequential sb_fdm_step calls, and must be faster
 * once the working set no longer fits in cache.
 * usage: test_batch <blob.sbb> [G] [steps] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sb_baked.h"
#include "sb_fdm.h"

#define MAXG 32

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Deterministic per-(step, env, index) input stream, so both passes see exactly
 * the same forcing without materialising the whole history. */
static double hval(unsigned long long key) {
    key = key * 6364136223846793005ULL + 1442695040888963407ULL;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return (double)((key >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
}

static void fill_inputs(int step, int g, int ncv, int k, double* q,
                        double* ambient, double* h, double* ag) {
    for (int i = 0; i < ncv; i++)
        q[i] = 2000.0 * (hval(((unsigned long long)step << 40) ^
                              ((unsigned long long)g << 20) ^ (unsigned)i)
                         - 0.5);
    *ambient = 250.0 + 60.0 * hval(((unsigned long long)step << 8) ^ (unsigned)g);
    *h = 0.5 + 3.0 * hval(((unsigned long long)step << 9) ^ (unsigned)g ^ 0x55);
    /* every third env has no air-side coupling (the inactive branch). */
    int off = (g % 3 == 2);
    for (int i = 0; i < k; i++)
        ag[i] = off ? 0.0
                    : 0.02 * hval(((unsigned long long)step << 24) ^
                                  ((unsigned long long)g << 12) ^ (unsigned)i);
}

static void reset_fdm(SbFdm* s, const SbBlob* b) {
    memcpy(s->temp, sb_f64(b, "temp_init"), sizeof(double) * (size_t)s->n_cvs);
    memcpy(s->mass_temp, sb_f64(b, "mass_temp_init"),
           sizeof(double) * (size_t)s->n_solved);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <blob.sbb> [G] [steps]\n", argv[0]);
        return 2;
    }
    int G = argc > 2 ? atoi(argv[2]) : 8;
    int steps = argc > 3 ? atoi(argv[3]) : 20;
    if (G < 1 || G > MAXG)
        return 2;
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0)
        return 1;
    static SbFdm envs[MAXG];
    for (int g = 0; g < G; g++)
        if (sb_fdm_init(&envs[g], &b) != 0) {
            printf("fdm init FAIL\n");
            return 1;
        }
    const int n = envs[0].n_solved, k = envs[0].k, ncv = envs[0].n_cvs;
    printf("blob %s: n_cvs=%d n_solved=%d k=%d  G=%d steps=%d\n", argv[1], ncv,
           n, k, G, steps);

    double* q = malloc(sizeof(double) * (size_t)ncv);
    double* ag = malloc(sizeof(double) * (size_t)(k ? k : 1));
    double* aT = malloc(sizeof(double) * (size_t)(k ? k : 1));
    SbFdm** p = malloc(sizeof(SbFdm*) * (size_t)G);
    const double** qp = malloc(sizeof(double*) * (size_t)G);
    const double** agp = malloc(sizeof(double*) * (size_t)G);
    const double** aTp = malloc(sizeof(double*) * (size_t)G);
    double* rhs = malloc(sizeof(double) * (size_t)n * G * 2);
    double* sol = rhs + (size_t)n * G;
    double* work = malloc(sizeof(double) * (size_t)n * G);
    double* ref_t = malloc(sizeof(double) * (size_t)ncv * G);
    double* ref_m = malloc(sizeof(double) * (size_t)n * G);
    if (!q || !ag || !aT || !p || !rhs || !work || !ref_t || !ref_m)
        return 1;
    /* One forcing buffer per env: the batched pass fills every env's column
     * before the single solve, so a shared buffer would be overwritten. */
    double** qb = malloc(sizeof(double*) * (size_t)G);
    double** agb = malloc(sizeof(double*) * (size_t)G);
    double** aTb = malloc(sizeof(double*) * (size_t)G);
    for (int g = 0; g < G; g++) {
        qb[g] = malloc(sizeof(double) * (size_t)ncv);
        agb[g] = malloc(sizeof(double) * (size_t)(k ? k : 1));
        aTb[g] = malloc(sizeof(double) * (size_t)(k ? k : 1));
    }
    for (int g = 0; g < G; g++) {
        p[g] = &envs[g];
        qp[g] = qb[g];
        agp[g] = agb[g];
        aTp[g] = aTb[g];
    }

    /* ---- pass 1: sequential, one env at a time ---- */
    for (int g = 0; g < G; g++)
        reset_fdm(&envs[g], &b);
    double t0 = now_s();
    for (int s = 0; s < steps; s++) {
        for (int g = 0; g < G; g++) {
            double amb, h;
            fill_inputs(s, g, ncv, k, qb[g], &amb, &h, ag);
            for (int i = 0; i < k; i++)
                aT[i] = 280.0 + 30.0 * hval(((unsigned long long)s << 32) ^
                                            ((unsigned long long)g << 16) ^
                                            (unsigned)i ^ 0x9e37);
            sb_fdm_step(&envs[g], qb[g], amb, h, k ? ag : NULL,
                        k ? aT : NULL);
        }
    }
    double tseq = now_s() - t0;
    for (int g = 0; g < G; g++) {
        memcpy(&ref_t[(size_t)g * ncv], envs[g].temp,
               sizeof(double) * (size_t)ncv);
        memcpy(&ref_m[(size_t)g * n], envs[g].mass_temp,
               sizeof(double) * (size_t)n);
    }

    /* ---- pass 2: batched ---- */
    for (int g = 0; g < G; g++)
        reset_fdm(&envs[g], &b);
    t0 = now_s();
    for (int s = 0; s < steps; s++) {
        double amb[MAXG], h[MAXG];
        for (int g = 0; g < G; g++) {
            fill_inputs(s, g, ncv, k, qb[g], &amb[g], &h[g], agb[g]);
            for (int i = 0; i < k; i++)
                aTb[g][i] = 280.0 + 30.0 * hval(((unsigned long long)s << 32) ^
                                                ((unsigned long long)g << 16) ^
                                                (unsigned)i ^ 0x9e37);
        }
        sb_fdm_step_batch(p, G, qp, amb, h, agp, aTp, rhs, sol, work);
    }
    double tbatch = now_s() - t0;

    long dt = 0, dm = 0;
    double wt = 0.0, wm = 0.0;
    for (size_t i = 0; i < (size_t)ncv * G; i++) {
        double d = fabs(envs[i / ncv].temp[i % ncv] - ref_t[i]);
        if (d != 0.0) dt++;
        if (d > wt) wt = d;
    }
    for (size_t i = 0; i < (size_t)n * G; i++) {
        double d = fabs(envs[i / n].mass_temp[i % n] - ref_m[i]);
        if (d != 0.0) dm++;
        if (d > wm) wm = d;
    }
    printf("batch vs sequential: temp %ld/%zu differ (max %.3e), mass %ld/%zu "
           "(max %.3e)\n", dt, (size_t)ncv * G, wt, dm, (size_t)n * G, wm);
    printf("  %d steps x %d envs: sequential %.3f s  batched %.3f s  %.2fx\n",
           steps, G, tseq, tbatch, tseq / tbatch);
    printf("  per env-step: sequential %.3f ms  batched %.3f ms\n",
           1000.0 * tseq / (G * steps), 1000.0 * tbatch / (G * steps));

    int bad = (dt != 0) || (dm != 0);
    printf(bad ? "BATCH FDM: FAIL\n" : "BATCH FDM: OK\n");
    return bad;
}

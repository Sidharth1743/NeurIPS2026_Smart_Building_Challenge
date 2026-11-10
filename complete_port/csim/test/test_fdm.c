/* M1b: replay the dumped per-step FDM inputs and compare the temperature field.
 * usage: test_fdm <blob.sbb> <trace.fdm> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_fdm.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <blob.sbb> <trace.fdm>\n", argv[0]); return 2; }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;
    static SbFdm s;
    if (sb_fdm_init(&s, &b) != 0) { printf("FDM init FAIL\n"); return 1; }
    int loop = (argc > 3 && strcmp(argv[3], "loop") == 0);
    static SbConv conv;
    if (loop && sb_conv_init(&conv, &b) != 0) { printf("conv init FAIL\n"); return 1; }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    int hdr[5];
    if (fread(hdr, sizeof(int), 5, f) != 5) return 1;
    int n_cvs = hdr[0], k = hdr[1], n_solved = hdr[2], nsteps = hdr[3];
    printf("trace: n_cvs=%d k=%d n_solved=%d steps=%d (solver %d/%d/%d)\n",
           n_cvs, k, n_solved, nsteps, s.n_cvs, s.k, s.n_solved);
    if (n_cvs != s.n_cvs || k != s.k || n_solved != s.n_solved) {
        printf("shape mismatch\n"); return 1;
    }

    double* g = (double*)malloc(sizeof(double) * (k ? k : 1));
    double* T = (double*)malloc(sizeof(double) * (k ? k : 1));
    double* iq = (double*)malloc(sizeof(double) * n_cvs);
    double* tnew = (double*)malloc(sizeof(double) * n_cvs);
    double* mtemp = (double*)malloc(sizeof(double) * n_solved);

    double* tprev = (double*)malloc(sizeof(double) * n_cvs);
    double* mb = (double*)malloc(sizeof(double) * n_solved);
    double worst = 0.0, worstm = 0.0;
    for (int st = 0; st < nsteps; st++) {
        double amb, cc;
        if (fread(&amb, 8, 1, f) != 1) return 1;
        if (fread(&cc, 8, 1, f) != 1) return 1;
        if (fread(g, 8, k, f) != (size_t)k) return 1;
        if (fread(T, 8, k, f) != (size_t)k) return 1;
        if (fread(tprev, 8, n_cvs, f) != (size_t)n_cvs) return 1;
        if (fread(iq, 8, n_cvs, f) != (size_t)n_cvs) return 1;
        if (fread(mb, 8, n_solved, f) != (size_t)n_solved) return 1;
        if (fread(tnew, 8, n_cvs, f) != (size_t)n_cvs) return 1;
        if (fread(mtemp, 8, n_solved, f) != (size_t)n_solved) return 1;
        /* measure C's carried state vs the oracle's, then reset to isolate */
        double sderr = 0.0;
        if (st > 0)
            for (int p = 0; p < n_cvs; p++)
                sderr = fmax(sderr, fabs(s.temp[p] - tprev[p]));
        if (!loop) {
            memcpy(s.temp, tprev, sizeof(double) * n_cvs);
            memcpy(s.mass_temp, mb, sizeof(double) * n_solved);
        }
        sb_fdm_step(&s, iq, amb, cc, g, T);
        double m = 0.0, mm = 0.0;
        for (int p = 0; p < n_cvs; p++) m = fmax(m, fabs(s.temp[p] - tnew[p]));
        for (int p = 0; p < n_solved; p++)
            mm = fmax(mm, fabs(s.mass_temp[p] - mtemp[p]));
        printf("  step %2d: carried-state err=%.3e  T err=%.3e  mass err=%.3e\n",
               st, sderr, m, mm);
        if (loop) sb_conv_apply(&conv, s.temp);   /* advance for the next step */
        if (m > worst) worst = m;
        if (mm > worstm) worstm = mm;
    }
    printf("worst over %d steps: T %.3e  mass %.3e\n", nsteps, worst, worstm);
    free(g); free(T); free(iq); free(tnew); free(mtemp);
    sb_fdm_free(&s);
    sb_close(&b);
    if (worst > 1e-9) { printf("FDM REPLAY: FAIL\n"); return 1; }
    printf("FDM REPLAY: OK\n");
    return 0;
}

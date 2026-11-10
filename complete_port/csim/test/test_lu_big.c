/* Real-matrix LU replay test: solve with the baked L/U/perm and compare to the
 * baked reference solve (which is scipy's own lu.solve). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sb_baked.h"
#include "sb_lu.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <blob.sbb>\n", argv[0]); return 2; }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;

    int n = (int)(sb_find(&b, "lu_perm_r")->count);
    const int* Lp = sb_i32(&b, "lu_L_indptr");
    const int* Li = sb_i32(&b, "lu_L_indices");
    const double* Lx = sb_f64(&b, "lu_L_data");
    const int* Up = sb_i32(&b, "lu_U_indptr");
    const int* Ui = sb_i32(&b, "lu_U_indices");
    const double* Ux = sb_f64(&b, "lu_U_data");
    const int* pr = sb_i32(&b, "lu_perm_r");
    const int* pc = sb_i32(&b, "lu_perm_c");
    const double* bvec = sb_f64(&b, "solref_b");
    const double* xref = sb_f64(&b, "solref_x");

    SbLu lu;
    if (sb_lu_build(&lu, n, Lp, Li, Lx, Up, Ui, Ux, pr, pc) != 0) {
        printf("LU BIG: FAIL (build)\n");
        return 1;
    }
    double* x = (double*)malloc(sizeof(double) * n);
    double* w = (double*)malloc(sizeof(double) * n);

    double t0 = now();
    sb_lu_solve(&lu, bvec, x, w);
    double ts = now() - t0;

    double err = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        err = fmax(err, fabs(x[i] - xref[i]));
        den = fmax(den, fabs(xref[i]));
    }
    printf("n=%d nnzL=%lld nnzU=%lld solve=%.3fs max|x-xref|=%.3e rel=%.3e\n",
           n, (long long)(sb_find(&b, "lu_L_data")->count),
           (long long)(sb_find(&b, "lu_U_data")->count), ts, err, err / den);

    free(x); free(w);
    sb_lu_free(&lu);
    sb_close(&b);
    if (err / den > 1e-9) { printf("LU BIG: FAIL\n"); return 1; }
    printf("LU BIG: OK\n");
    return 0;
}

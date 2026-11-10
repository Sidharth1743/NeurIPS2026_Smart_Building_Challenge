/* Loader sanity test: open a baked blob, print its shape, touch key sections. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sb_baked.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <blob.sbb>\n", argv[0]); return 2; }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;

    printf("sections: %d\n", b.n_sec);
    const SbSection* g = sb_find(&b, "grid");
    const int64_t* grid = sb_i64(&b, "grid");
    printf("grid = %lld x %lld\n", (long long)grid[0], (long long)grid[1]);
    printf("n_solved = %d\n", (int)(sb_find(&b, "solved")->count));
    printf("perm n = %d\n", (int)(sb_find(&b, "perm")->count));
    printf("conv_p = %g  conv_distance = %g\n",
           sb_f64_1(&b, "conv_p", -1), sb_f64_1(&b, "conv_distance", -1));
    printf("n_field = %d  n_action = %d\n",
           (int)(sb_find(&b, "field_names")->count),
           (int)(sb_find(&b, "action_names")->count));

    /* names of first/last sections */
    printf("first=%s last=%s\n", b.sec[0].name, b.sec[b.n_sec - 1].name);

    /* A_conduction CSR/CSC triple present and consistent */
    const int32_t* ap = sb_i32(&b, "A_conduction_indptr");
    const int32_t* ai = sb_i32(&b, "A_conduction_indices");
    const double*  ad = sb_f64(&b, "A_conduction_data");
    int64_t nnz = sb_find(&b, "A_conduction_data")->count;
    printf("A nnz = %lld  indptr[last] = %d  max idx = %d\n",
           (long long)nnz, ap[(int)(sb_find(&b, "A_conduction_indptr")->count) - 1],
           ai[nnz - 1]);

    sb_close(&b);
    (void)g;
    printf("LOAD OK\n");
    return 0;
}

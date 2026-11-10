#include "sb_convect.h"

#include <stdlib.h>
#include <string.h>

int sb_conv_init(SbConv* c, const SbBlob* b) {
    memset(c, 0, sizeof(*c));
    const int64_t* grid = sb_i64(b, "grid");
    c->nr = (int)grid[0];
    c->nc = (int)grid[1];
    c->n_cvs = c->nr * c->nc;
    c->p = sb_f64_1(b, "conv_p", 0.0);
    c->distance = (int)sb_f64_1(b, "conv_distance", -1.0);
    c->seed = (uint32_t)sb_i64_1(b, "conv_seed", 0);
    c->room_off = sb_i32(b, "conv_room_offsets");
    c->room_cells = sb_i32(b, "conv_room_cells");
    c->cand_off = sb_i32(b, "conv_cand_offsets");
    c->cand_idx = sb_i32(b, "conv_cand_idx");
    if (!c->room_off || !c->room_cells || !c->cand_off || !c->cand_idx) return -1;
    c->n_rooms = (int)(sb_find(b, "conv_room_offsets")->count) - 1;
    int max_room = 0;
    for (int i = 0; i < c->n_rooms; i++) {
        int n = c->room_off[i + 1] - c->room_off[i];
        if (n > max_room) max_room = n;
    }
    c->left = (int*)malloc(sizeof(int) * (max_room ? max_room : 1));
    c->right = (int*)malloc(sizeof(int) * (max_room ? max_room : 1));
    if (!c->left || !c->right) return -1;
    sb_rng_seed(&c->rng, c->seed);
    return 0;
}

void sb_conv_free(SbConv* c) {
    free(c->left);
    free(c->right);
    memset(c, 0, sizeof(*c));
}

void sb_conv_apply(SbConv* c, double* temp) {
    if (c->p == 0.0 || c->distance == 0) return;
    int nc = c->nc;
    for (int rm = 0; rm < c->n_rooms; rm++) {
        int base = c->room_off[rm];
        int n = c->room_off[rm + 1] - base;
        if (n <= 0) continue;
        int m = 0;
        for (int j = 0; j < n; j++) {
            if (sb_rng_double(&c->rng) > c->p) continue;
            int cg = base + j;
            int start = c->cand_off[cg];
            int end = c->cand_off[cg + 1];
            if (end <= start) continue;
            int jj = c->cand_idx[start + (int)sb_rng_interval(&c->rng,
                                                              (uint32_t)(end - start - 1))];
            c->left[m] = cg;
            c->right[m] = jj;
            m++;
        }
        for (int t = m - 1; t > 0; t--) {
            int j2 = (int)sb_rng_interval(&c->rng, (uint32_t)t);
            int tmp = c->left[t]; c->left[t] = c->left[j2]; c->left[j2] = tmp;
            tmp = c->right[t]; c->right[t] = c->right[j2]; c->right[j2] = tmp;
        }
        for (int t = 0; t < m; t++) {
            int a = c->left[t], bb = c->right[t];
            int ia = c->room_cells[2 * a] * nc + c->room_cells[2 * a + 1];
            int ib = c->room_cells[2 * bb] * nc + c->room_cells[2 * bb + 1];
            double held = temp[ia];
            temp[ia] = temp[ib];
            temp[ib] = held;
        }
    }
}

/* Stochastic convection: mirrors StochasticConvectionSimulator._convect_room
 * with numpy's MT19937 stream. Baked room cells + candidate CSR. */
#ifndef SB_CONVECT_H
#define SB_CONVECT_H

#include "sb_baked.h"
#include "sb_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nr, nc, n_cvs, n_rooms;
    double p;
    int distance;
    uint32_t seed;
    const int32_t* room_off;    /* n_rooms+1, positions into room_cells */
    const int32_t* room_cells;  /* 2 * n_cells (row,col) */
    const int32_t* cand_off;    /* indexed by global cell position */
    const int32_t* cand_idx;    /* global cell positions */
    int* left;
    int* right;
    SbRng rng;
} SbConv;

int  sb_conv_init(SbConv* c, const SbBlob* b);
void sb_conv_free(SbConv* c);
void sb_conv_apply(SbConv* c, double* temp);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_CONVECT_H */

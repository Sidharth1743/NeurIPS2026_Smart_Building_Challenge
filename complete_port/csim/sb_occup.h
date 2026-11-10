/* Occupancy: RandomizedArrivalDepartureOccupancy, ported exactly.
 * One advance per simulation step, in room order; the reward interval
 * [ts_{k+1},ts_{k+2}) is reused by the next step's gains (cache hit), so the
 * chains tick once per step. */
#ifndef SB_OCCUP_H
#define SB_OCCUP_H

#include "sb_baked.h"
#include "sb_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_zones;
    int zone_assignment;
    int ea, la, ed, ld;
    double p_arrival, p_departure;
    const int64_t* occ_hour;
    const uint8_t* occ_workday;
    int hour_m1, workday_m1;
    int* state;      /* n_zones * zone_assignment, 0 AWAY / 1 WORK */
    double* V;       /* n_zones: occupancy for the current interval */
    SbRng rng;
} SbOccup;

int  sb_occup_init(SbOccup* o, const SbBlob* b);
void sb_occup_free(SbOccup* o);
/* Advances the chains for interval [ts_step, ts_step+1) and fills V. */
void sb_occup_advance(SbOccup* o, int step_index);
/* The reset observation advances the pre-episode interval [ts_-1,ts_0). */
void sb_occup_prime(SbOccup* o);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_OCCUP_H */

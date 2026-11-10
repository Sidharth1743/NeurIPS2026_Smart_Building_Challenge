#ifndef SB_OBS_H
#define SB_OBS_H
#include "sb_sim.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Writes the (n_dev + n_aux) observation floats for the state at observation
 * step j (which follows simulation step j-1). occV is the occupancy for
 * [ts_{j-1}, ts_j) (used for num_occupants). out must hold n_dev + n_aux. */
void sb_obs(SbSim* s, const double* occV, int j, float* out);
#ifdef __cplusplus
} /* extern "C" */
#endif
#endif

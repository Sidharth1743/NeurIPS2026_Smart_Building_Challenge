#ifndef SB_REWARD_H
#define SB_REWARD_H
#include "sb_sim.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Agent reward for the transition ending at ts_{kv}, from the current
 * (post-step) sim state and the occupancy for [ts_kv, ts_kv+1). */
double sb_reward(SbSim* s, const double* occV, int kv, double* parts); /* parts[3] npr,nec,ncar or NULL */
#ifdef __cplusplus
} /* extern "C" */
#endif
#endif

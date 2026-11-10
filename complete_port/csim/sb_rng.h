/* MT19937 matching numpy's legacy RandomState stream (verified: numba's
 * np.random == numpy RandomState for random()/randint()). */
#ifndef SB_RNG_H
#define SB_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { uint32_t mt[624]; int idx; } SbRng;

void     sb_rng_seed(SbRng* r, uint32_t seed);        /* init_genrand */
uint32_t sb_rng_u32(SbRng* r);                        /* genrand_int32 */
double   sb_rng_double(SbRng* r);                     /* RandomState.random_sample */
uint32_t sb_rng_interval(SbRng* r, uint32_t max);     /* RandomState.randint(0,max+1) */

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_RNG_H */

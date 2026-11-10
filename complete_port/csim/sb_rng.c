#include "sb_rng.h"

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfu
#define MT_UPPER_MASK 0x80000000u
#define MT_LOWER_MASK 0x7fffffffu

void sb_rng_seed(SbRng* r, uint32_t seed) {
    r->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        uint32_t prev = r->mt[i - 1];
        r->mt[i] = 1812433253u * (prev ^ (prev >> 30)) + (uint32_t)i;
    }
    r->idx = MT_N;
}

static void mt_generate(SbRng* r) {
    for (int i = 0; i < MT_N; i++) {
        uint32_t y = (r->mt[i] & MT_UPPER_MASK)
                   | (r->mt[(i + 1) % MT_N] & MT_LOWER_MASK);
        r->mt[i] = r->mt[(i + MT_M) % MT_N]
                 ^ (y >> 1)
                 ^ ((y & 1u) ? MT_MATRIX_A : 0u);
    }
    r->idx = 0;
}

uint32_t sb_rng_u32(SbRng* r) {
    if (r->idx >= MT_N) mt_generate(r);
    uint32_t y = r->mt[r->idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

double sb_rng_double(SbRng* r) {
    uint32_t a = sb_rng_u32(r) >> 5;
    uint32_t b = sb_rng_u32(r) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

uint32_t sb_rng_interval(SbRng* r, uint32_t max) {
    uint32_t mask, value;
    if (max == 0) return 0;
    mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    value = sb_rng_u32(r) & mask;
    while (value > max) value = sb_rng_u32(r) & mask;
    return value;
}

// One learned zone command, decoded into the damper and reheat the gateway receives.
// u is the squashed damper-slot sample in [-1, 1]. The reheat slot is not a second sample.
// Physical fraction of an action a is (a+1)/2 over the canonical [-1, 1] range.
// The box never closes the damper below 0.1, which is action -0.8.
// u = 0 is half reheat at that floor. r61_mod_band: the warm side is one joint
// proportional stroke — coil and damper open together with u (the old 0.5 split and
// its coil-first ordering are gone). The valve never outruns the plant: the hot water
// the coil draws is capped by a demand reset, not by "the zones asked, so boiler max".
// The reset is sb_water_reset_cap: it reads the mean over zones of
// max(0, heating setpoint - zone temp) from the observation slots the kernel already
// resolves, and saturates at SB_WATER_RESET_K — the shipped controller's own constants
// (pid_water: 2.0 K proportional valve band, RESET_FULL_K = 4.0 K water reset; its
// docstring calls a jammed on/off valve a resolution failure, not an authority limit).
// So u = 1 means "full valve and damper at reset water", never "boiler at maximum":
// fuel is duty-cycle x demand, the modulation lambda=1's credit window pays for.
// Open loop: no deficit feedback into the per-zone command — the policy's u stays
// the only knob; the reset is a static obs -> cap map.
// The cool decode is separate: the step uses it only when u < 0, the supply is
// colder than the zone, and the zone is above its cooling setpoint.
// A macro guard, not pragma once: pufferl includes this through both
// ocean/smart_buildings and src/, and those are two files to the compiler.
#ifndef SB_ZONE_COMMAND_CUH
#define SB_ZONE_COMMAND_CUH

constexpr float SB_ZONE_FLOOR_A = -0.8f;
constexpr float SB_WATER_RESET_K = 4.0f;

__host__ __device__ inline void sb_zone_clamp(float* u) {
    if (!(*u <= 1.0f)) *u = 1.0f;
    if (!(*u >= -1.0f)) *u = -1.0f;
}

// Warm stroke: for u in (0, 1] coil and damper open together, both proportional to u;
// u = 0 is the neutral half-heat at the damper floor; u in [-1, 0) closes the coil
// toward off and leaves the damper at the floor (the cool request decodes separately).
__host__ __device__ inline void sb_zone_curve(float u, float* damper, float* reheat) {
    sb_zone_clamp(&u);
    if (u > 0.0f) {
        *damper = SB_ZONE_FLOOR_A + u * (1.0f - SB_ZONE_FLOOR_A);
        *reheat = u;
    } else {
        *damper = SB_ZONE_FLOOR_A;
        *reheat = u;
    }
}

// Coil shut, damper opens as u goes from 0 to -1. The step calls this only for a real cool request.
__host__ __device__ inline void sb_zone_cool(float u, float* damper, float* reheat) {
    sb_zone_clamp(&u);
    float t = -u;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    *damper = SB_ZONE_FLOOR_A + t * (1.0f - SB_ZONE_FLOOR_A);
    *reheat = -1.0f;
}

__host__ __device__ inline void sb_zone_apply(float u, int cool, float* damper, float* reheat) {
    if (cool && u < 0.0f) sb_zone_cool(u, damper, reheat);
    else sb_zone_curve(u, damper, reheat);
}

// Physical reheat fraction in [0, 1]. u = 0 (reheat action 0) is 0.5.
__host__ __device__ inline float sb_heat_fraction(float reheat) {
    float h = (reheat + 1.0f) * 0.5f;
    if (h < 0.0f) h = 0.0f;
    if (h > 1.0f) h = 1.0f;
    return h;
}

// Pump differential pressure and run, from the average heat fraction.
// 0.5 -> action 0, which the run bit treats as on. 0 -> fully off. 1 -> fully on.
__host__ __device__ inline void sb_plant_from_heat(float mean_heat, float* dp, float* run) {
    float a = 2.0f * mean_heat - 1.0f;
    if (a > 1.0f) a = 1.0f;
    if (a < -1.0f) a = -1.0f;
    *dp = a;
    *run = a;
}

// r61_mod_band: the hot-water sample cannot command beyond the building's real demand.
// The cap is the demand reset from mean_deficit_k, the mean over zones of
// max(0, heating setpoint - zone temp) in native K. It reaches neutral action at
// SB_WATER_RESET_K/2 shortfall and full boiler at SB_WATER_RESET_K, and holds the loop
// off when the building is inside its bands — the fuel bound at saturation. Open loop:
// a static map of the observation, no state, no controller on the policy's u.
__host__ __device__ inline float sb_water_reset_cap(float mean_deficit_k) {
    float a = 2.0f * mean_deficit_k / SB_WATER_RESET_K - 1.0f;
    if (a > 1.0f) a = 1.0f;
    if (a < -1.0f) a = -1.0f;
    return a;
}

#endif

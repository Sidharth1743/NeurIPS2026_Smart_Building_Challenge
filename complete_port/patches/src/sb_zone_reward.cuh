// Per-zone step target. The same function the env kernel and the contract test call.
// p_z is the existing per-zone productivity npr_z. It is not npr_z minus the
// building productivity. Summed over the building's zones it equals npr.
// An in-band zone (deviation floored at 0) has p_z = 0.
// r_z = reward_scale * (pw * p_z - ew * s_z * energy - cw * s_z * carbon) / wsum,
// with s_z = occupancy / total occupancy. An empty building does not call this:
// every r_z is 0, and the night fuel stays on the building advantage.
#pragma once

__host__ __device__ inline double sb_zone_productivity(
        double temp, double heat_sp, double cool_sp, double ref,
        double max_pph, double min_pph, double occ_z, double total_occ, double dt_sec) {
    if (!(total_occ > 0.0) || !(occ_z > 0.0)) return 0.0;
    double inv_hour = 1.0 / 3600.0;
    double max_prod = max_pph * total_occ * dt_sec * inv_hour;
    double min_prod = min_pph * total_occ * dt_sec * inv_hour;
    double den = max_prod - min_prod;
    if (den == 0.0) return 0.0;
    double dev = temp - cool_sp;
    if (heat_sp - temp > dev) dev = heat_sp - temp;
    if (dev < 0.0) dev = 0.0;
    double shortfall = (dev <= ref) ? dev / ref
        : 1.0 + ((dev - ref) / ref) * ((dev - ref) / ref);
    double produced = max_pph * (1.0 - shortfall) * occ_z * dt_sec * inv_hour;
    double share = occ_z / total_occ;
    return (produced - min_prod * share) / den - share;
}

__host__ __device__ inline double sb_zone_target(
        double productivity, double share, double energy, double carbon,
        double pw, double ew, double cw, double reward_scale) {
    double wsum = pw + ew + cw;
    if (wsum == 0.0) return 0.0;
    return reward_scale * (pw * productivity - ew * share * energy - cw * share * carbon) / wsum;
}

#!/usr/bin/env python3
"""Actions-only step trace (M2 gate). Per step: hybrid action, decoded native
values, per-zone occupancy for gains, the air-side (g,T), and the temperature."""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    cfg = Path(sys.argv[1]); out = Path(sys.argv[2])
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    from smart_buildings import evaluate
    env = evaluate.load_hybrid_environment(cfg)
    obs0, _ = env.reset()
    sim = env.building.simulator; pb = sim.building
    n_cvs = int(pb.temp.size); n_zones = len(pb.room_dict) - 2
    n_field = len(env.field_names)
    occ = env.building._occupancy
    ts0 = sim.current_timestamp; step = int(sim._time_step_sec)
    kk = len(sim._coupled_vavs)

    nat_box = {}
    orig_mk = env._create_action_request
    def mk(arr):
        req = orig_mk(arr)
        nat_box["v"] = np.array([r.continuous_value for r in
                                 req.single_action_requests], dtype=np.float64)
        return req
    env._create_action_request = mk

    fdm_box = {}
    orig_fdm = sim.finite_differences_timestep
    def fdm(*, ambient_temperature, convection_coefficient,
            air_side_conductance=None, supply_temperature=None):
        fdm_box["g"] = (np.zeros(kk) if air_side_conductance is None
                        else np.asarray(air_side_conductance, dtype=np.float64))
        fdm_box["T"] = (np.zeros(kk) if supply_temperature is None
                        else np.asarray(supply_temperature, dtype=np.float64))
        fdm_box["swt"] = float(sim._hvac.hot_water_system
                              .supply_water_temperature_sensor)
        orig_fdm(ambient_temperature=ambient_temperature,
                 convection_coefficient=convection_coefficient,
                 air_side_conductance=air_side_conductance,
                 supply_temperature=supply_temperature)
    sim.finite_differences_timestep = fdm

    ah_box = {}
    orig_ah = sim._get_air_handler_reward_infos
    def ah():
        m = orig_ah()
        for _, v in m.items():
            ah_box["b"] = v.blower_electrical_energy_rate
            ah_box["c"] = v.air_conditioning_electrical_energy_rate
        return m
    sim._get_air_handler_reward_infos = ah
    hw_box = {}
    orig_hw = sim._get_hws_reward_infos
    def hw():
        m, dtp = orig_hw()
        for _, v in m.items():
            hw_box["p"] = v.pump_electrical_energy_rate
        return m, dtp
    sim._get_hws_reward_infos = hw
    rp_box = {}
    orig_cr = env.reward_function.compute_reward
    def cr(info):
        resp = orig_cr(info)
        rp_box["v"] = np.array([resp.normalized_productivity_regret,
                                resp.normalized_energy_cost,
                                resp.normalized_carbon_emission,
                                resp.electricity_energy_cost,
                                resp.natural_gas_energy_cost,
                                ah_box["b"], ah_box["c"], hw_box["p"],
                                0, 0, 0, 0],
                               dtype=np.float64)
        return resp
    env.reward_function.compute_reward = cr
    occ_log = {}
    orig_azo = occ.average_zone_occupancy
    def azo(zone, start, end):
        v = orig_azo(zone, start, end)
        occ_log[(str(start), zone)] = v
        return v
    occ.average_zone_occupancy = azo

    phys = [z for z in pb.room_dict if z not in ("exterior_space", "interior_wall")]
    asp = env.action_spec()
    nc = int(asp["continuous_action"].shape[0])
    nd = int(asp["discrete_action"].shape[0])
    rng = np.random.default_rng(11)
    rec = []
    for k in range(nsteps):
        cont = rng.uniform(-1, 1, size=nc).astype(np.float32)
        disc = rng.integers(0, 2, size=nd).astype(np.int32)
        res = env.step({"continuous_action": cont, "discrete_action": disc})
        rew = float(res[1])
        obs = np.asarray(res[0], dtype=np.float64)
        ts = ts0 + k * pd.Timedelta(step, unit="s")
        occV = np.array([occ_log.get((str(ts), z), 0.0) for z in phys],
                        dtype=np.float64)
        rec.append((cont.astype(np.float64), disc.astype(np.float64),
                    nat_box["v"].copy(), occV,
                    fdm_box["g"].copy(), fdm_box["T"].copy(),
                    np.array([fdm_box["swt"]], dtype=np.float64),
                    np.array(pb.temp, dtype=np.float64).reshape(-1),
                    np.array([rew], dtype=np.float64),
                    rp_box["v"].copy(), obs))

    with open(out, "wb") as f:
        f.write(struct.pack("<iiiiiii", n_cvs, n_zones, nc, nd, len(rec), kk, n_field))
        f.write(np.asarray(obs0, dtype="<f8").tobytes())
        for cont, disc, nat, occV, g, T, swt, tnew, rew, rp, obs in rec:
            for a in (cont, disc, nat, occV, g, T, swt, tnew, rew, rp, obs):
                f.write(a.astype("<f8").tobytes())
    print(f"wrote {out}: n_cvs={n_cvs} n_zones={n_zones} nc={nc} nd={nd} "
          f"k={kk} steps={len(rec)} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

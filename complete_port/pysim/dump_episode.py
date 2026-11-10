#!/usr/bin/env python3
"""Full-episode trace (8640 steps) driven by a REAL baseline policy.

Dumping the whole 205k-cell field every step would be ~14 GB, so the trace is:
  * every step: hybrid action, decoded native values, per-zone occupancy, the
    air-side (g, T), boiler sensor, reward, the full observation, and the
    temperatures of a fixed set of probe cells (exact comparison);
  * full field at a set of checkpoint steps.

usage: dump_episode.py <config.gin> <out.trace> [steps] [policy_key]
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np
import pandas as pd

NPROBE = 64


def main() -> None:
    cfg = Path(sys.argv[1])
    out = Path(sys.argv[2])
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 8640
    pkey = sys.argv[4] if len(sys.argv) > 4 else "bang_bang"

    from smart_buildings import evaluate
    env = evaluate.load_hybrid_environment(cfg)
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    policy = evaluate.initialize_policy(
        evaluate.POLICY_MAP[pkey], meta, env.action_spec()
    )
    print(f"policy {pkey}: {type(policy).__name__}")

    obs, _ = env.reset()
    sim = env.building.simulator
    pb = sim.building
    nx = int(pb.temp.size)
    n_zones = len(pb.room_dict) - 2
    n_field = len(env.field_names)
    occ = env.building._occupancy
    ts0 = sim.current_timestamp
    step = int(sim._time_step_sec)
    kk = len(sim._coupled_vavs)
    probe = np.linspace(0, nx - 1, NPROBE, dtype=np.int64)

    nat_box = {}
    orig_mk = env._create_action_request

    def mk(arr):
        req = orig_mk(arr)
        nat_box["v"] = np.array(
            [r.continuous_value for r in req.single_action_requests], dtype=np.float64
        )
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
        fdm_box["swt"] = float(sim._hvac.hot_water_system.supply_water_temperature_sensor)
        orig_fdm(ambient_temperature=ambient_temperature,
                 convection_coefficient=convection_coefficient,
                 air_side_conductance=air_side_conductance,
                 supply_temperature=supply_temperature)

    sim.finite_differences_timestep = fdm

    occ_log = {}
    orig_azo = occ.average_zone_occupancy

    def azo(zone, start, end):
        v = orig_azo(zone, start, end)
        occ_log[(str(start), zone)] = v
        return v

    occ.average_zone_occupancy = azo

    phys = [z for z in pb.room_dict if z not in ("exterior_space", "interior_wall")]

    # checkpoints: the first 10 steps, then a geometric + final spread
    ck = [s for s in range(min(10, nsteps))]
    s = 10
    while s < nsteps:
        ck.append(s)
        s = s * 2 if s < 2000 else s + 800
    if (nsteps - 1) not in ck:
        ck.append(nsteps - 1)
    ck = sorted(set(c for c in ck if c < nsteps))
    ckset = set(ck)
    print(f"steps={nsteps} n_cvs={nx} n_field={n_field} nprobe={NPROBE} "
          f"checkpoints={len(ck)}")

    f = open(out, "wb")
    f.write(struct.pack("<8i", nx, n_zones, 0, 0, nsteps, kk, n_field, NPROBE))
    f.write(np.asarray(obs, dtype="<f8").tobytes())
    f.write(probe.astype("<i4").tobytes())

    cont_dim = None
    disc_dim = None
    fields = []
    nbad = 0
    for k in range(nsteps):
        act = evaluate.get_policy_action(policy, obs)
        act = evaluate.validate_action(act, env.action_spec())
        cont = np.asarray(act["continuous_action"], dtype=np.float64)
        disc = np.asarray(act["discrete_action"], dtype=np.float64)
        if cont_dim is None:
            cont_dim, disc_dim = cont.size, disc.size
        res = env.step({"continuous_action": act["continuous_action"],
                        "discrete_action": act["discrete_action"]})
        rew = float(res[1])
        obs_next = np.asarray(res[0], dtype=np.float64)
        ts = ts0 + k * pd.Timedelta(step, unit="s")
        occV = np.array([occ_log.get((str(ts), z), 0.0) for z in phys],
                        dtype=np.float64)
        nat = nat_box["v"].copy()
        tflat = np.asarray(pb.temp).reshape(-1)
        probes = np.asarray(tflat[probe], dtype=np.float64)
        if not np.all(np.isfinite(probes)):
            nbad += 1
        rec = (cont, disc, nat, occV, fdm_box["g"].copy(), fdm_box["T"].copy(),
               np.array([fdm_box["swt"]], dtype=np.float64),
               np.array([rew], dtype=np.float64), obs_next, probes)
        for a in rec:
            f.write(np.asarray(a, dtype="<f8").tobytes())
        if k in ckset:
            fields.append((k, np.array(tflat, dtype="<f8")))
        obs = obs_next
        if (k + 1) % 1000 == 0:
            print(f"  step {k+1}/{nsteps} (non-finite probe steps: {nbad})", flush=True)

    f.write(struct.pack("<i", len(fields)))
    f.write(np.asarray([c[0] for c in fields], dtype="<i4").tobytes())
    for _, arr in fields:
        f.write(arr.tobytes())
    f.close()
    # patch the action dims into the header now that they are known
    with open(out, "r+b") as g:
        g.seek(8)
        g.write(struct.pack("<2i", int(cont_dim), int(disc_dim)))
    print(f"wrote {out}: nc={cont_dim} nd={disc_dim} steps={nsteps} "
          f"checkpoints={len(fields)} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

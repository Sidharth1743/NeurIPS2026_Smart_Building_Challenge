#!/usr/bin/env python3
"""Dump the exact per-step inputs/outputs of the FDM timestep (M1b gate).

Per step records: ambient, convection coeff, air-side conductance/supply temps,
the state BEFORE the solve (temp field, mass_temp), the building.input_q used by
the solve, and the state AFTER (temp field, mass_temp). Recording the state
before each step lets the C replay reset to it, isolating per-step correctness
from cross-step drift.

Run from starter-kit/:
    .venv/bin/python ../complete_port/pysim/dump_fdm.py <config.gin> <out.fdm> [nsteps]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    cfg = Path(sys.argv[1])
    out = Path(sys.argv[2])
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    from smart_buildings import evaluate

    env = evaluate.load_hybrid_environment(cfg)
    env.reset()
    sim = env.building.simulator
    pb = sim.building
    n_cvs = int(pb.temp.size)
    n_solved = int(sim._solver._solved.size)
    k = len(sim._coupled_vavs)

    rec = []
    orig = sim.finite_differences_timestep

    def wrapped(*, ambient_temperature, convection_coefficient,
                air_side_conductance=None, supply_temperature=None):
        tprev = np.array(pb.temp, dtype=np.float64).reshape(-1)
        mb = np.array(sim._solver._mass_temp, dtype=np.float64)
        iq = np.array(pb.input_q, dtype=np.float64).reshape(-1)
        orig(ambient_temperature=ambient_temperature,
             convection_coefficient=convection_coefficient,
             air_side_conductance=air_side_conductance,
             supply_temperature=supply_temperature)
        tnew = np.array(pb.temp, dtype=np.float64).reshape(-1)
        ma = np.array(sim._solver._mass_temp, dtype=np.float64)
        g = (np.zeros(k) if air_side_conductance is None
             else np.asarray(air_side_conductance, dtype=np.float64))
        T = (np.zeros(k) if supply_temperature is None
             else np.asarray(supply_temperature, dtype=np.float64))
        rec.append((float(ambient_temperature), float(convection_coefficient),
                    g, T, tprev, iq, mb, tnew, ma))

    sim.finite_differences_timestep = wrapped

    asp = env.action_spec()
    rng = np.random.default_rng(11)
    for _ in range(nsteps):
        env.step({kk: rng.uniform(-1, 1, size=v.shape).astype(v.dtype)
                  for kk, v in asp.items()})

    with open(out, "wb") as f:
        f.write(struct.pack("<iiiii", n_cvs, k, n_solved, len(rec), 0))
        for amb, cc, g, T, tprev, iq, mb, tnew, ma in rec:
            f.write(struct.pack("<dd", amb, cc))
            for a in (g, T, tprev, iq, mb, tnew, ma):
                f.write(a.astype("<f8").tobytes())
    print(f"wrote {out} : n_cvs={n_cvs} k={k} n_solved={n_solved} "
          f"steps={len(rec)} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

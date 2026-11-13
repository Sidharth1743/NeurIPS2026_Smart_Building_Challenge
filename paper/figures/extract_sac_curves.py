#!/usr/bin/env python3
"""Dump SAC training curves from TensorBoard to JSON (run with the SAC venv)."""
from __future__ import annotations

import glob
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/data/neurips-2026-smart-buildings/SAC"
OUT = "/tmp/opencode/sac_curves.json"
RUNS = [
    "r11_structured_b2048_s75",
    "r13_structured_zm_s75",
    "r06_utd15_async_960_1m_s75",
    "r04_utd4_async_960_1m_s75",
    "r07_utd15_ln_960_1m_s75",
    "r05_utd4_async_960_1m_frozen_s75",
    "r10_structured_960_1m_s75",
    "r14_structured_200m_s75",
    "r15_structured_beta_b2048_s75",
]


def main() -> None:
    out = {}
    for run in RUNS:
        f = glob.glob(f"{ROOT}/results/{run}/tb/events*")[0]
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        ev = ea.Scalars("env/episode_score_scen_mean")
        out[run] = {"steps": [e.step / 1e6 for e in ev], "values": [e.value for e in ev]}
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, {k: len(v["steps"]) for k, v in out.items()})


if __name__ == "__main__":
    main()

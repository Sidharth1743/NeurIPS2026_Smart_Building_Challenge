#!/usr/bin/env python3
"""Bin a finished SAC run's scalars into fixed environment-step windows.

  SAC/.venv/bin/python SAC/tools/analyze_run.py --run r01_cleanrl_sac_s75 [--window 2000000]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SAC = Path(__file__).resolve().parents[1]

KEYS = [
    "losses/alpha", "losses/log_pi", "losses/entropy_per_head",
    "losses/qf1_loss", "losses/qf1_values", "losses/actor_loss",
    "env/score_pos", "env/productivity", "env/energy_cost", "env/carbon",
    "env/comfort_dev_K", "env/worst_dev_K", "env/out_of_band_frac",
    "live/reward", "live/productivity", "live/comfort_dev_K",
    "live/elec_kw", "live/gas_kw", "live/out_of_band_frac",
    "norm/reward_mean", "norm/reward_inv_std",
]


def load(run: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = SAC / "results" / run
    files = sorted(glob.glob(str(d / "tb_fixed/events*"))) or sorted(glob.glob(str(d / "tb/events*")))
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for k in KEYS + sorted(t for t in have if t.startswith("scenario/")):
        if k not in have:
            continue
        ev = ea.Scalars(k)
        if "tb_fixed" not in files[-1]:
            pass
        out[k] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, default=4_000_000)
    ap.add_argument("--agents", type=int, default=256)
    a = ap.parse_args()
    series = load(a.run)

    def fix(s):
        # old writers used the vector-step index; new ones already write env steps
        return s * a.agents if (len(s) and s.max() < 1_000_000) else s
    fixed = (SAC / "results" / a.run / "tb_fixed").exists()
    print(f"run {a.run}  (tb_fixed={fixed})")
    edges = np.arange(0, 50_135_040 + 1, a.window)
    hdr = "  ".join(f"{int(e/1e6):>6}M" for e in edges[:-1])
    print(f"\n{'tag':34s} {'window (env steps)':>0s}")
    print(f"{'':34s} {hdr}")
    for k in KEYS:
        if k not in series:
            continue
        s, v = series[k]
        s = fix(s)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi)
            cells.append(f"{np.mean(v[m]):7.3f}" if m.any() else "      -")
        print(f"{k:34s} " + "  ".join(cells))
    print("\nper-month score (window mean):")
    months = sorted(k for k in series if k.endswith("/episode_score"))
    for k in months:
        s, v = series[k]
        s = fix(s)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi) & (v > 0)
            cells.append(f"{np.mean(v[m]):6.3f}" if m.any() else "     -")
        print(f"  {k.split('/')[1]:32s} " + " ".join(cells))
    print("\nper-month comfort_dev_K (window mean):")
    for k in sorted(k for k in series if k.endswith("/comfort_dev_K")):
        s, v = series[k]
        s = fix(s)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s < hi)
            cells.append(f"{np.mean(v[m]):6.2f}" if m.any() else "    -")
        print(f"  {k.split('/')[1]:32s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

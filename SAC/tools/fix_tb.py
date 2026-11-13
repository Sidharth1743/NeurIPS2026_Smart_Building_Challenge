#!/usr/bin/env python3
"""Rewrite a SAC run's TensorBoard scalars into a correct view.

The run writes scalars with the *vector* step as the global step and writes the
env's raw 0 for a per-scenario score before that scenario's first episode has
finished. Both are fine for the event file and wrong for a reader. This exporter
writes an equivalent log with:

  * global_step = environment steps (vector step x n_agents), so the x-axis
    matches the 50,135,040 budget and train.log;
  * no fake zeros: a score point is dropped until its first real value;
  * `env/score` (mean reward, negative) renamed `env/mean_reward`, and a
    positive `env/score_pos` added next to the per-month `episode_score`.

Values are copied verbatim. Idempotent: rewrites the destination each time.

  SAC/.venv/bin/python SAC/tools/fix_tb.py --run r01_cleanrl_sac_s75 [--watch 60]
"""
from __future__ import annotations

import argparse
import glob
import shutil
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

SAC = Path(__file__).resolve().parents[1]
SCORE_TAGS = ("episode_score",)


def export(run: str, agents: int) -> int:
    src = SAC / "results" / run / "tb"
    dst = SAC / "results" / run / "tb_fixed"
    files = sorted(glob.glob(str(src / "events*")))
    if not files:
        return 0
    tmp = dst.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    w = SummaryWriter(str(tmp))
    n_pts = 0
    seen_score: dict[str, bool] = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            is_score = any(s in tag for s in SCORE_TAGS)
            for e in ea.Scalars(tag):
                if is_score and e.value == 0.0 and not seen_score.get(tag):
                    continue  # nothing has finished yet: not a zero, just absent
                if is_score:
                    seen_score[tag] = True
                step = e.step * agents
                if tag == "env/score":
                    w.add_scalar("env/mean_reward", e.value, step, walltime=e.wall_time)
                    w.add_scalar("env/score_pos", -e.value, step, walltime=e.wall_time)
                else:
                    w.add_scalar(tag, e.value, step, walltime=e.wall_time)
                n_pts += 1
    w.close()
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return n_pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--watch", type=int, default=0, help="seconds between refreshes (0 = once)")
    a = ap.parse_args()
    while True:
        n = export(a.run, a.agents)
        print(f"{time.strftime('%H:%M:%S')} exported {n} points -> results/{a.run}/tb_fixed", flush=True)
        if a.watch <= 0:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score SAC checkpoints on the 10-month fp64 grid used by the 50M runs.

The env writes the score itself (SB_EVAL_OUT): one env per month, desync off,
graphs off, deterministic action tanh(mean). Nothing here fabricates a number;
it only feeds the policy and reads the env's JSON.

  SAC/.venv/bin/python SAC/tools/score_months.py --ckpt SAC/results/<run>/ckpt/000010027008.pt \
      --out SAC/results/<run>/scored/10027008.json --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
ROOT = SAC.parent
sys.path.insert(0, str(SAC / "train"))
from envdriver import EnvDriver  # noqa: E402
from model import Actor, StructuredActor  # noqa: E402

CP = ROOT / "complete_port/artifacts/sets"
MONTHS = [
    "r85_eval_building_train_1__fairbanks_1",
    "r85_eval_building_train_2__phoenix_6",
    "r85_eval_building_train_4__denver_2",
    "r85_eval_building_train_5__atlanta_6",
    "gen6_building_train_1__phoenix_5",
    "gen6_building_train_2__fairbanks_2",
    "gen6_building_train_4__atlanta_5",
    "gen6_building_train_5__denver_3",
    "gen6_building_train_8__denver_6",
    "gen6_building_train_8__phoenix_4",
]


def build_set() -> Path:
    d = SAC / "artifacts/eval10"
    d.mkdir(parents=True, exist_ok=True)
    for m in MONTHS:
        src = (CP / m).glob("*.sbb")
        src = list(src)[0]
        dst = d / src.name
        if not dst.exists():
            dst.symlink_to(os.path.relpath(src.resolve(), d))
    n = len(list(d.glob("*.sbb")))
    assert n == len(MONTHS), f"{d} has {n} blobs"
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    a = ap.parse_args()

    setdir = build_set()
    ck = torch.load(a.ckpt, map_location="cpu")
    at = ck.get("actor_type", "flat")
    actor = (StructuredActor(ck["obs_dim"], ck["act_dim"], ck["hidden"]) if at == "structured"
             else Actor(ck["obs_dim"], ck["act_dim"], ck["hidden"])).cuda()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    out = a.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"preserving existing {out}")
    run_dir = SAC / "run" / f"score-{a.ckpt.stem}-{a.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    c_eval = f"/w/SAC/run/{run_dir.name}/{a.tag}.json"
    h_eval = run_dir / f"{a.tag}.json"
    if h_eval.exists():
        h_eval.unlink()

    env = EnvDriver(str(setdir), len(MONTHS), len(MONTHS), run_dir, name=f"score-{a.tag}",
                    graph=0, desync=0, rew_norm=0, gpu=a.gpu, eval_out=c_eval,
                    log_path=run_dir / "driver.log")
    h = env.h
    assert h.n_agents == len(MONTHS) and h.ep_len == 8640, (h.n_agents, h.ep_len)
    mask = torch.as_tensor(env.mask, device="cuda")
    env.reset()
    for t in range(h.ep_len):
        with torch.no_grad():
            xin = torch.as_tensor(env.obs, device="cuda")
            if ck.get("actor_type", "flat") == "structured":
                mean, _ = actor(actor.featurize(xin))
            else:
                mean, _ = actor(xin)
            act = (torch.tanh(mean) * mask).cpu().numpy()
        env.step(act)
        if (t + 1) % 288 == 0:
            print(f"  {a.tag} step {t + 1}/{h.ep_len}", flush=True)
    env.close()

    d = json.loads(h_eval.read_text())
    assert d["complete"], d
    d["checkpoint"] = str(a.ckpt)
    d["step"] = int(ck["step"])
    out.write_text(json.dumps(d, indent=2) + "\n")
    print(f"SCORE {d['score']:.6f}  months {d['n_scenarios_done']}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score one checkpoint on one building's 18 held-out weathers (one merged group).

Same protocol as the r139 90-set runs: 18 scenarios in one group, desync off,
graphs off, deterministic tanh(mean); the env writes the JSON (SB_EVAL_OUT).

  SAC/.venv/bin/python SAC/tools/score_set.py --ckpt <pt> --set <dir with 18 .sbb> \
      --out <json> --gpu 1
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
sys.path.insert(0, str(SAC / "train"))
from envdriver import EnvDriver  # noqa: E402
from model import Actor, StructuredActor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--record", type=Path, default=None,
                    help="directory for the per-step recording (obs, act, rew, term, mask, parts)")
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"preserving existing {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    setdir = a.set.resolve()
    nblob = len(list(setdir.glob("*.sbb")))
    assert nblob > 0, setdir

    ck = torch.load(a.ckpt, map_location="cpu")
    at = ck.get("actor_type", "flat")
    head = ck.get("head", "gauss")
    hkw = ({"head": head, "beta_floor": ck.get("beta_floor", 0.5),
            "beta_conc_max": ck.get("beta_conc_max", 12.0)} if head == "beta" else {})
    actor = (StructuredActor(ck["obs_dim"], ck["act_dim"], ck["hidden"], **hkw)
             if at == "structured"
             else Actor(ck["obs_dim"], ck["act_dim"], ck["hidden"], **hkw)).cuda()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    run_dir = SAC / "run" / f"score-{a.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    c_eval = f"/w/SAC/run/{run_dir.name}/{a.tag}.json"
    h_eval = run_dir / f"{a.tag}.json"
    if h_eval.exists():
        h_eval.unlink()

    env = EnvDriver(str(setdir), nblob, nblob, run_dir, name=f"score-{a.tag}",
                    graph=0, desync=0, rew_norm=0, gpu=a.gpu, eval_out=c_eval,
                    log_path=run_dir / "driver.log")
    h = env.h
    assert h.n_agents == nblob and h.ep_len == 8640, (h.n_agents, h.ep_len)
    mask = torch.as_tensor(env.mask, device="cuda")
    # ---- per-step recording, written in the same pass that scores the month
    rec = a.record
    files = {}
    if rec is not None:
        rec.mkdir(parents=True, exist_ok=True)
        files = {name: open(rec / f"{name}.bin", "wb") for name in
                 ("obs", "loc", "rew", "term", "parts")}
        np.asarray(env.mask, np.float32).tofile(rec / "mask.bin")
        mask_np = np.asarray(env.mask, np.float32)
    env.reset()
    for t in range(h.ep_len):
        with torch.no_grad():
            xin = torch.as_tensor(env.obs, device="cuda")
            feat = actor.featurize(xin) if at == "structured" else xin
            act_np = actor.det_action(feat, mask).cpu().numpy()
        if rec is not None:
            obs_np = np.asarray(env.obs, np.float32)
            _, rew_np, term_np = env.step(act_np)
            parts_np = env.parts()
            files["obs"].write(obs_np.tobytes())
            files["loc"].write(act_np.astype(np.float32).tobytes())
            files["rew"].write(np.asarray(rew_np, np.float32).tobytes())
            files["term"].write(np.asarray(term_np, np.float32).tobytes())
            files["parts"].write(parts_np.tobytes())
        else:
            env.step(act_np)
    env.close()
    if rec is not None:
        for f in files.values():
            f.close()
        (rec / "record.json").write_text(json.dumps({
            "steps": h.ep_len, "agents": h.n_agents, "obs_dim": h.obs_dim,
            "act_dim": h.act_dim, "mask_width": h.mask_width,
            "scen": np.asarray(env.h.scen).tolist() if hasattr(env.h, "scen") else None,
            "parts": ["productivity", "energy_cost", "carbon", "comfort_dev_K",
                      "worst_dev_K", "out_of_band_frac", "elec_kwh", "gas_kwh"],
            "dtypes": {"obs": "f32", "loc": "f32", "rew": "f32", "term": "f32",
                       "mask": "f32", "parts": "f64"},
            "note": "one row per agent per step; loc is the executed mean the scorer ran"
        }, indent=2) + "\n")
        got = {p.name: p.stat().st_size for p in sorted(rec.iterdir())}
        print(f"RECORD {" ".join(f"{k}={v}" for k, v in got.items())}", flush=True)

    d = json.loads(h_eval.read_text())
    assert d["complete"] and d["n_scenarios_done"] == nblob, d
    d["checkpoint"] = str(a.ckpt)
    d["step"] = int(ck["step"])
    d["set"] = str(a.set)
    out.write_text(json.dumps(d, indent=2) + "\n")
    print(f"SCORE {d['score']:.6f}  scenarios {d['n_scenarios_done']}  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

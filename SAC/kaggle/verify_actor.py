#!/usr/bin/env python3
"""Prove the Kaggle submission emits the same action as the trained torch actor.

Observations come from a real recorded episode (the 983-wide canonical vectors the scorer
ran), so no simulator and no GPU lock are involved. Left is the trainer
(features.t_bundle + model.StructuredActor, torch, `det_action`); right is the submission
(numpy feature blocks + numpy actor, `_actor_forward`) with the head read from the
weights sidecar. The head matters: the Gaussian decodes tanh(mean), the Beta decodes
2a/(a+b)-1 with the canonical floor/cap.

    SAC/.venv/bin/python SAC/kaggle/verify_actor.py \
        --ckpt results/r15_structured_beta_b2048_s75/ckpt/000050135040.pt \
        --weights SAC/kaggle/r15_50m/model/weights.bin \
        --record results/r14_structured_200m_s75/record90/49972800/building_train_1 \
        --steps 400 --gpu 0

Exit 0 only if every active head matches within 1e-4 over every compared row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
sys.path.insert(0, str(SAC / "train"))
sys.path.insert(0, str(HERE))

from features import t_bundle            # noqa: E402
from model import StructuredActor        # noqa: E402
import submission as S                   # noqa: E402

TOL = 1e-4
OBS_DIM = 983


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True, help="dir with obs.bin and mask.bin")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    w = S._parse_actor(a.weights)
    meta = S._sidecar(a.weights)
    head = meta.get("head", "gauss")
    floor = float(meta.get("beta_floor", 0.5))
    cap = float(meta.get("beta_conc_max", 12.0))
    print(f"numpy actor: {sum(v.size for v in w.values())} floats, head {head} "
          f"floor {floor} cap {cap}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck_head = ck.get("head", "gauss")
    if ck_head != head:
        raise SystemExit(f"head mismatch: checkpoint {ck_head} vs weights {head}")
    hkw = ({"head": ck_head, "beta_floor": ck.get("beta_floor", 0.5),
            "beta_conc_max": ck.get("beta_conc_max", 12.0)} if ck_head == "beta" else {})
    dev = torch.device(f"cuda:{a.gpu}")
    actor = StructuredActor(int(ck["obs_dim"]), int(ck["act_dim"]), int(ck["hidden"]), **hkw)
    actor.load_state_dict(ck["actor"])
    actor.to(dev).eval()
    print(f"torch actor: step {int(ck['step']):,} head {ck_head}", flush=True)

    mask = np.fromfile(a.record / "mask.bin", dtype=np.float32).reshape(-1, 219)
    obs = np.fromfile(a.record / "obs.bin", dtype=np.float32)
    rows_all = obs.size // OBS_DIM
    n = min(a.steps * mask.shape[0], rows_all)
    obs = obs[: n * OBS_DIM].reshape(n, OBS_DIM)
    # the recording is step-major over agents; the building's 18 scenarios share one mask row
    m = mask[np.arange(n) % mask.shape[0]]
    print(f"{n} rows from {a.record.name}  ({rows_all} rows in the file), "
          f"active heads/row {int(m[0].sum())}", flush=True)

    mt = torch.as_tensor(m, device=dev)
    worst, worst_row = 0.0, -1
    for i in range(0, n, 512):
        o = obs[i:i + 512]
        with torch.no_grad():
            torch_act = actor.det_action(actor.featurize(torch.as_tensor(o, device=dev)),
                                         mt[i:i + 512]).cpu().numpy()
        np_act = np.stack([S._actor_forward(w, o[j], head, floor, cap)
                           for j in range(o.shape[0])]) * m[i:i + 512]
        d = np.abs(np_act - torch_act)
        if d.max() > worst:
            worst, worst_row = float(d.max()), i + int(np.unravel_index(d.argmax(), d.shape)[0])
        print(f"  rows {i:>5}-{i + o.shape[0] - 1:<5} max |diff| {d.max():.3e}  "
              f"mean {d.mean():.3e}", flush=True)

    with torch.no_grad():
        sample = actor.det_action(actor.featurize(torch.as_tensor(obs[:64], device=dev)), mt[:64])
    print(f"\nworst row {worst_row}; max abs difference {worst:.3e} (tol {TOL:.0e}); "
          f"action range [{float(sample.min()):+.3f}, {float(sample.max()):+.3f}]; "
          f"share |a|>0.95 {float((sample.abs() > 0.95).float().mean()):.3f}")
    ok = worst < TOL
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a SAC checkpoint's actor to the flat float32 `weights.bin` the Kaggle kernel reads.

    SAC/.venv/bin/python SAC/kaggle/export_weights.py <ckpt.pt> <out_dir>

Writes <out_dir>/weights.bin in exactly the order `_parse_actor` in submission.py reads:

    trunk.w (256,80) trunk.b  r0.w (4,324) r0.b  r1.w (8,288) r1.b
    r2.w (6,272) r2.b  zm1.w (32,324) zm1.b  zm2.w (4,32) zm2.b

36,518 floats = 146,072 bytes. The action-scale/bias buffers are not weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ORDER = [("trunk", "weight"), ("trunk", "bias"),
         ("r0", "weight"), ("r0", "bias"),
         ("r1", "weight"), ("r1", "bias"),
         ("r2", "weight"), ("r2", "bias"),
         ("zm1", "weight"), ("zm1", "bias"),
         ("zm2", "weight"), ("zm2", "bias")]
SHAPES = {"trunk": (256, 80), "r0": (4, 324), "r1": (8, 288), "r2": (6, 272),
          "zm1": (32, 324), "zm2": (4, 32)}


def main() -> int:
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["actor"]
    flat = []
    for name, part in ORDER:
        key = f"{name}.{part}"
        if key not in sd:
            raise KeyError(f"{key} missing; actor keys are {sorted(sd)}")
        t = sd[key].detach().cpu().numpy().astype(np.float32)
        if part == "weight" and t.shape != SHAPES[name]:
            raise ValueError(f"{key} has shape {t.shape}, expected {SHAPES[name]}")
        if not np.isfinite(t).all():
            raise ValueError(f"{key} has non-finite values")
        flat.append(t.reshape(-1))
    w = np.concatenate(flat).astype(np.float32)
    (out / "weights.bin").write_bytes(w.tobytes())
    meta = {"checkpoint": str(ckpt), "step": int(ck.get("step") or 0),
            "actor_type": ck.get("actor_type"), "floats": int(w.size),
            "bytes": int(w.nbytes), "order": [f"{a}.{b}" for a, b in ORDER],
            "head": ck.get("head", "gauss"),
            "beta_floor": float(ck.get("beta_floor", 0.5)),
            "beta_conc_max": float(ck.get("beta_conc_max", 12.0))}
    (out / "weights.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out/'weights.bin'}  {w.size} floats  {w.nbytes} bytes  step {meta['step']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

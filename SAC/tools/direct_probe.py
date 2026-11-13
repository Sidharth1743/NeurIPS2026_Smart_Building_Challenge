#!/usr/bin/env python3
"""Cheap startup probe: torch, the driver (direct launch), 3 steps, and the blobs.

Runs in ~30 s on any GPU. Used by the Modal launcher's `probe` mode, and locally as

    SAC/.venv/bin/python SAC/tools/direct_probe.py --blob_dir <dir> --gpu 1 --binary bin/sb_env
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

SAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAC / "train"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob_dir", required=True)
    ap.add_argument("--binary", default="bin/sb_env")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--groups", type=int, default=18)
    ap.add_argument("--agents", type=int, default=18)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    print("python", sys.version.split()[0], flush=True)
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
    blobs = sorted(Path(a.blob_dir).glob("*.sbb"))
    print(f"blobs {len(blobs)} in {a.blob_dir}", flush=True)
    assert len(blobs) > 0, "no blobs"

    from envdriver import EnvDriver
    with EnvDriver(a.blob_dir, a.groups, a.agents, Path(a.blob_dir).parent / "probe_run",
                   name="probe", graph=1, desync=1, rew_norm=1, gpu=a.gpu,
                   launch="direct", binary=a.binary,
                   env_vars={"CUDA_VISIBLE_DEVICES": str(a.gpu)},
                   log_path=Path(a.blob_dir).parent / "probe_driver.log") as env:
        print(f"env: agents {env.h.n_agents} obs {env.h.obs_dim} act {env.h.act_dim} "
              f"ep {env.h.ep_len} zm {env.h.zm_dim}", flush=True)
        env.reset()
        r = []
        for _ in range(a.steps):
            o, rw, t = env.step(np.zeros((env.h.n_agents, env.h.act_dim), np.float32))
            r.append(float(rw.mean()))
        print(f"{a.steps} steps ok: reward mean {np.mean(r):.4f}  obs finite {bool(np.isfinite(o).all())}",
              flush=True)
    print("PROBE OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)

#!/usr/bin/env python3
"""Export one SAC checkpoint and publish it to Kaggle: dataset, kernel, then the submission.

    SAC/.venv/bin/python SAC/kaggle/push.py <ckpt.pt> <slug> "<message>" [--no-submit]

<slug> names both objects: dataset `sidharth1743/submission-<slug>-model`,
kernel `sidharth1743/submission-<slug>`. Everything is staged under
SAC/kaggle/stage_<slug>/ so a rerun is self-contained, and every step is checked:
the dataset must be `ready`, the kernel must reach COMPLETE, and its log must contain the
loaded-weights line before the submission is sent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
ROOT = SAC.parent
KAGGLE = ROOT / "starter-kit/.venv/bin/kaggle"
OWNER = "sidharth1743"
COMP = "neurips-2026-smart-buildings"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kaggle(*args: str) -> str:
    r = run([str(KAGGLE), *args])
    out = (r.stdout or "") + (r.stderr or "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("slug")
    ap.add_argument("message")
    ap.add_argument("--no-submit", action="store_true")
    a = ap.parse_args()

    stage = HERE / f"stage_{a.slug}"
    for sub in ("model", "dataset", "kernel"):
        (stage / sub).mkdir(parents=True, exist_ok=True)

    # 1. weights
    r = run([str(SAC / ".venv/bin/python"), str(HERE / "export_weights.py"),
             str(a.ckpt.resolve()), str(stage / "model")])
    print(r.stdout.strip() or r.stderr.strip(), flush=True)
    if r.returncode:
        return 1
    shutil.copyfile(stage / "model/weights.bin", stage / "dataset/weights.bin")
    shutil.copyfile(HERE / "submission.py", stage / "kernel/submission.py")

    ds_id = f"{OWNER}/submission-{a.slug}-model"
    kn_id = f"{OWNER}/submission-{a.slug}"
    (stage / "dataset/dataset-metadata.json").write_text(json.dumps(
        {"title": f"submission-{a.slug}-model", "id": ds_id,
         "licenses": [{"name": "CC0-1.0"}]}, indent=2) + "\n")
    (stage / "kernel/kernel-metadata.json").write_text(json.dumps({
        "id": kn_id, "title": f"submission_{a.slug}", "code_file": "submission.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "false",
        "dataset_sources": [ds_id], "competition_sources": [COMP],
        "kernel_sources": [], "model_sources": []}, indent=2) + "\n")

    # 2. dataset
    print(kaggle("datasets", "create", "-p", str(stage / "dataset")).strip()[-200:], flush=True)
    for _ in range(30):
        st = kaggle("datasets", "status", ds_id).strip().splitlines()[-1]
        if st.strip() == "ready":
            print(f"dataset {ds_id} ready", flush=True)
            break
        time.sleep(20)
    else:
        print(f"dataset {ds_id} not ready: {st}", flush=True)
        return 1

    # 3. kernel
    print(kaggle("kernels", "push", "-p", str(stage / "kernel")).strip()[-200:], flush=True)
    for _ in range(60):
        st = kaggle("kernels", "status", kn_id).strip().splitlines()[-1]
        if "COMPLETE" in st or "ERROR" in st or "FAILED" in st:
            print(st, flush=True)
            break
        time.sleep(30)
    if "COMPLETE" not in st:
        return 1

    # 4. the commit log must show the weights and a real reward
    outdir = HERE / f"out_{a.slug}"
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    print(kaggle("kernels", "output", kn_id, "-p", str(outdir)).strip()[-160:], flush=True)
    logs = list(outdir.glob("*.log"))
    text = logs[0].read_text(errors="replace") if logs else ""
    loaded = "loaded weights" in text
    print(f"log says loaded weights: {loaded}", flush=True)
    for key in ("loaded weights", "mean reward"):
        for line in text.splitlines():
            if key in line:
                print("  ", line[-160:], flush=True)
    if not loaded:
        print("refusing to submit: the weights line is missing", flush=True)
        return 1
    if a.no_submit:
        return 0

    # 5. submit
    print(kaggle("competitions", "submit", "-c", COMP, "-f", "submission.parquet",
                 "-k", kn_id, "-v", "1", "-m", a.message).strip()[-300:], flush=True)
    print(kaggle("competitions", "submissions", "-c", COMP).strip()[:600], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

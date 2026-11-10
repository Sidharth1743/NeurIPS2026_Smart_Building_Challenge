#!/usr/bin/env python3
"""Preserve r22 provenance and evaluate a fixed, historical fp64 binary.

This driver uses the shipped GPU queue/evaluator. It never removes checkpoints,
never trains on evaluation sets, and skips only complete evaluation outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import re
from datetime import datetime, timezone
PL_ROOT = os.environ.get("SB_PL_ROOT", os.path.expanduser("~/.cache/sb/pufferlib"))

CP = Path(__file__).resolve().parents[1]
ROOT = CP.parent
OUT = CP / "results" / "campaign_r23"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> None:
    runtime = OUT / "historical_runtime"
    if (OUT / "provenance.json").exists():
        old = Path(PL_ROOT + "/plverify")
        saved = OUT / "sources_before"
        for src, dst in [(old / "puffer", runtime / "puffer_training_r22"),
                         (old / "build.sh", saved / "build.sh"),
                         (old / "src/pufferenv.h", saved / "pufferenv.h"),
                         (old / "src/tblog.h", saved / "tblog.h")]:
            if not dst.exists(): shutil.copy2(src, dst)
        diff = saved / "deployment.diff"
        if not diff.exists():
            diff.write_bytes(subprocess.check_output(["git", "-C", str(old), "diff"]))
        deployed = saved / "deployed_env"
        if not deployed.exists(): shutil.copytree(old / "ocean/smart_buildings", deployed)
        print("Historical snapshot already exists; preserving it.", flush=True)
        return
    old = Path(PL_ROOT + "/plverify")
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copytree(old / "config", runtime / "config")
    shutil.copy2(old / "puffer_f64", runtime / "puffer_f64")
    saved = OUT / "sources_before"
    saved.mkdir()
    files = {
        "pufferlib_core.diff": CP / "patches/pufferlib_core.diff",
        "smart_buildings.ini": CP / "config/smart_buildings.ini",
        "algo.cu": old / "src/algo.cu",
        "pufferl.cu": old / "src/pufferl.cu",
        "Experiments.md": ROOT / "Experiments.md",
        "HISTORY.md": CP / "docs/HISTORY.md",
    }
    for name, path in files.items():
        shutil.copy2(path, saved / name)
    checkpoints = [
        CP / "results/runs/r21/ckpt/0000000091750400.bin",
        CP / "results/runs/r22/ckpt/0000000009994240.bin",
        CP / "results/runs/r22/ckpt/0000000024985600.bin",
        CP / "results/runs/r22/ckpt/0000000039976960.bin",
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": 0.00598,
        "binary": {"path": str(runtime / "puffer_f64"), "sha256": sha256(runtime / "puffer_f64")},
        "sources": {name: sha256(saved / name) for name in files},
        "checkpoints": {str(p.relative_to(CP)): sha256(p) for p in checkpoints},
        "note": "Historical binary preserved for unchanged forward-policy comparisons; no solver approximation.",
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def summarize() -> None:
    rows = []
    for path in sorted((CP / "results/eval").glob("r22_*_fairbanks.json")):
        step = path.name.split("_")[1]
        scores, parts = {}, {}
        for weather in ["fairbanks", "atlanta", "denver"]:
            other = path.with_name(f"r22_{step}_{weather}.json")
            if not other.exists():
                continue
            doc = json.loads(other.read_text())
            if not doc.get("complete"):
                continue
            scores[weather] = doc["score"]
            s = next(iter(doc["scenarios"].values()))
            parts[weather] = {"comfort": -0.6 * s["productivity"],
                              "energy": 0.2 * s["energy_cost"], "carbon": 0.2 * s["carbon"]}
        if len(scores) == 3:
            rows.append({"step": int(step), "scores": scores, "mean": sum(scores.values()) / 3,
                         "pair_mean": (scores["fairbanks"] + scores["atlanta"]) / 2,
                         "parts": parts})
    baselines = {}
    with (ROOT / "reference_baselines.csv").open() as f:
        for row in csv.DictReader(f):
            if row["building"] == "building_train_1" and row["weather"] in {"fairbanks_1", "atlanta_2", "denver_7"}:
                baselines.setdefault(row["policy"], []).append(abs(float(row["total_reward"]) / 8640))
    report = {"r22": rows, "baselines_three": {k: sum(v) / len(v) for k, v in baselines.items()},
              "best_three": min(rows, key=lambda x: x["mean"]),
              "limitations": ["Three training weathers only; not the leaderboard metric over its actual test set.",
                              "Gradient floor and weather diversity changed together; causal effects are confounded.",
                              "logit_outside measures |logit|>3, not a zero derivative."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r22_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def prepare_train30() -> None:
    """Use the declared shipped set, never the historically contaminated train/ directory."""
    pairs, building = [], None
    for line in (ROOT / "starter-kit/smart_buildings/training_set.yaml").read_text().splitlines():
        if line.startswith("- building: "): building = line.split(": ", 1)[1]
        elif line.startswith("  weather: "):
            assert building
            pairs.append((building, line.split(": ", 1)[1]))
    assert len(pairs) == len(set(pairs)) == 30
    assert len({b for b, _ in pairs}) == 6
    sources = []
    for b, w in pairs:
        name = f"{b}__{w}.sbb"
        candidates = [CP / "artifacts" / folder / name for folder in ["blobs", "nodes", "train"]]
        source = next((x for x in candidates if x.is_file() and x.resolve().name == name), None)
        assert source is not None, name
        sources.append(source)
    dataset = CP / "artifacts/sets/tr23_shipped30"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (b, w), source in zip(pairs, sources):
        target = dataset / source.name
        if target.exists(): assert os.path.samefile(target, source)
        else: os.link(source.resolve(), target)
        manifest.append({"scenario": f"{b}__{w}", "source": str(source.resolve()),
                         "bytes": source.stat().st_size})
    (OUT / "train30_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["python3", str(CP / "tools/check_split.py"), "--deep", "tr23_shipped30", "eval9"], check=True)
    print("Prepared and leak-checked 30 shipped scenarios, six buildings.", flush=True)


def pareto() -> None:
    fronts = {}
    for path in sorted((OUT / "eval").glob("*.json")):
        doc = json.loads(path.read_text())
        if not doc.get("complete"): continue
        scenes = doc["scenarios"]
        key = "|".join(sorted(scenes))
        n = len(scenes)
        row = {"file": str(path.relative_to(CP)), "n_scenarios": n, "score": doc["score"],
               "comfort": sum(-0.6*s["productivity"] for s in scenes.values())/n,
               "energy": sum(0.2*s["energy_cost"] for s in scenes.values())/n,
               "carbon": sum(0.2*s["carbon"] for s in scenes.values())/n}
        fronts.setdefault(key, []).append(row)
    result = []
    for key, rows in fronts.items():
        for row in rows:
            row["pareto"] = not any(all(other[k] <= row[k] for k in ["comfort", "energy", "carbon"])
                                     and any(other[k] < row[k] for k in ["comfort", "energy", "carbon"])
                                     for other in rows)
        result.append({"scenarios": key.split("|"), "rows": sorted(rows,key=lambda r:r["score"])})
    (OUT / "pareto.json").write_text(json.dumps(result, indent=2) + "\n")
    for front in result:
        print("Matched set:",len(front["scenarios"]),"scenarios")
        for row in front["rows"]: print(row)


def stage_fixed(gate: Path) -> None:
    assert gate.read_text().rstrip().endswith("ALL CHECKS PASSED"), "The full verification gate has not passed"
    runtime = OUT / "fixed_runtime"
    runtime.mkdir(exist_ok=False)
    deployed = Path(PL_ROOT + "/plverify")
    binary = deployed / "puffer"
    assert re.search(rb"lu_solve_t[^\x00\n]*EEv5LuDevPf", binary.read_bytes()) is None, "Refuse fp32 physics"
    shutil.copy2(binary, runtime / "puffer")
    shutil.copytree(deployed / "config", runtime / "config")
    manifest = {"gate": str(gate), "gate_sha256": sha256(gate),
                "binary_sha256": sha256(runtime / "puffer"),
                "patch_sha256": sha256(CP / "patches/pufferlib_core.diff"),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    (runtime / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def evaluate(gpu: int, baselines_only: bool = False) -> None:
    runtime = OUT / "historical_runtime"
    assert (runtime / "puffer_f64").exists(), "Run snapshot first"
    jobs = [
        ("r21_918m", "r21/ckpt/0000000091750400.bin", "r22_five"),
        ("r22_25m", "r22/ckpt/0000000024985600.bin", "r22_five"),
        ("r22_final", "r22/ckpt/0000000039976960.bin", "r22_five"),
        ("r21_918m", "r21/ckpt/0000000091750400.bin", "evalcore3"),
        ("r22_25m", "r22/ckpt/0000000024985600.bin", "evalcore3"),
        ("r22_final", "r22/ckpt/0000000039976960.bin", "evalcore3"),
        ("pid_water_matched", None, "r22_five"),
        ("pid_water_matched", None, "evalcore3"),
    ]
    env = dict(os.environ, PUFFERLIB=str(runtime), SB_BIN="puffer_f64")
    for name, checkpoint, dataset in jobs:
        if baselines_only and checkpoint is not None:
            continue
        output = f"campaign_r23/eval/{name}_{dataset}.json"
        target = CP / "results" / output
        expected = len(list((CP / "artifacts/sets" / dataset).glob("*.sbb")))
        if target.exists():
            d = json.loads(target.read_text())
            if d.get("complete") and d.get("n_scenarios_done") == expected:
                print("Already complete:", output, d["score"], flush=True)
                continue
            raise RuntimeError(f"Incomplete output exists; preserve and inspect it before retrying: {target}")
        print(datetime.now(timezone.utc).isoformat(), "EVALUATE", name, dataset, flush=True)
        model = f"/res/runs/{checkpoint}" if checkpoint else "-"
        flags = ["--env.residual=0", "--policy.tanh_squash=1"] if checkpoint else [
            "--env.residual=1", "--policy.zero_init_actions=1", "--policy.tanh_squash=0"]
        subprocess.run([str(CP / "tools/with_gpu.sh"), str(gpu), "--", str(CP / "tools/eval_policy.sh"),
                        f"sets/{dataset}", output, model,
                        "--env.canonical=1", *flags],
                       env=env, check=True)
        d = json.loads(target.read_text())
        assert d["complete"] and d["n_scenarios_done"] == expected, d
        print("RESULT", name, dataset, d["score"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "summarize", "evaluate", "prepare-train30", "pareto", "stage-fixed"])
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot": snapshot()
    elif args.command == "summarize": summarize()
    elif args.command == "prepare-train30": prepare_train30()
    elif args.command == "pareto": pareto()
    elif args.command == "stage-fixed":
        assert args.gate is not None
        stage_fixed(args.gate)
    else: evaluate(args.gpu, args.baselines_only)

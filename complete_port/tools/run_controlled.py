#!/usr/bin/env python3
"""Run an isolated, named experiment on one GPU without clearing any old run.

Example:
  python3 tools/run_controlled.py r23a --gpu 0 --runtime results/campaign_r23/fixed_runtime -- --train.total_timesteps=5000000 ...

The immutable runtime is shared read-only; checkpoint/log paths are unique per
run. The shipped per-GPU flock prevents two jobs from using the same device.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

CP = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("Separate puffer arguments with --")
    cut = sys.argv.index("--")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--binary", default="puffer")
    args = p.parse_args(sys.argv[1:cut])
    options = sys.argv[cut + 1:]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise SystemExit("Run name must contain only letters, numbers, _ and -")
    runtime = args.runtime.resolve()
    binary = runtime / args.binary
    assert binary.is_file(), binary
    # These are user requirements for every learned policy in this campaign.
    prohibited = {"env.residual", "env.comfort_shape", "env.guard_d1", "policy.zero_init_actions"}
    for item in options:
        if item.startswith("--") and "=" in item:
            key, value = item[2:].split("=", 1)
            if key in prohibited and float(value) != 0:
                raise SystemExit(f"Campaign requires {key}=0")
            if key in {"base.checkpoint_dir", "base.log_dir", "base.run_id"}:
                raise SystemExit(f"The driver owns {key} for isolation")
    run = CP / "results/runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    (run / "smart_buildings/ckpt").mkdir(parents=True)
    (run / "smart_buildings/tb").mkdir(parents=True)
    (run / "ckpt").symlink_to("smart_buildings/ckpt", target_is_directory=True)
    (run / "tb").symlink_to("smart_buildings/tb", target_is_directory=True)
    flags = ["--env.residual=0", "--env.comfort_shape=0", "--env.guard_d1=0",
             "--policy.zero_init_actions=0", "--base.eval_episodes=0",
             "--base.checkpoint_optim=0", "--tb.enabled=1",
             f"--base.checkpoint_dir=/res/runs/{args.name}",
             f"--base.log_dir=/res/runs/{args.name}", "--base.run_id=ckpt", *options]
    docker_env = []
    # diagnostics switches the trainer reads from its environment; SB_CREDIT is the number of
    # agents in one group, and update_diagnostics=1 refuses to start without it
    for var in ("SB_TRAJ", "SB_BLEND_PROBE", "SB_CREDIT", "SB_VALVE_DIAG", "SB_DETAIL_METRICS",
                "SB_ROLESTEP", "SB_ZONE_STATS", "SB_ZONE_EMPTY_OBS", "SB_ZONE_CREDIT_AUDIT",
                "SB_ZONE_MLP", "SB_F32"):
        value = os.environ.get(var)
        if value:
            docker_env += ["-e", f"{var}={value}"]
    command = [str(CP / "tools/with_gpu.sh"), str(args.gpu), "--",
               "docker", "run", "--rm", "--name", f"sb-{args.name}",
               "--gpus", f"device={args.gpu}",
               "-v", f"{runtime}:/work:ro", "-v", f"{CP / 'artifacts'}:/art:ro",
               "-v", f"{CP / 'results'}:/res", "-w", "/work",
               *docker_env,
               "pufferai/puffertank:5.0", "stdbuf", "-oL", "-eL",
               f"./{args.binary}", "train", *flags]
    manifest = {"name": args.name, "gpu": args.gpu, "started_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": str(runtime), "binary_sha256": digest(binary),
                "physics": "training-only fp32 solve" if b"EEv5LuDevPf" in binary.read_bytes() else "default fp64 solve",
                "patch_sha256": digest(CP / "patches/pufferlib_core.diff"),
                "flags": flags, "command": command}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "args.txt").write_text(shlex.join(flags) + "\n")
    (run / ".start").touch()
    print("START", args.name, "GPU", args.gpu, "logs", run, flush=True)
    started = time.monotonic()
    with (run / "train.log").open("wb") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout
        for data in iter(lambda: proc.stdout.read(8192), b""):
            log.write(data.replace(b"\r", b"\n")); log.flush()
        code = proc.wait()
    status = {"exit_code": code, "wall_seconds": time.monotonic() - started,
              "ended_utc": datetime.now(timezone.utc).isoformat()}
    (run / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("END", args.name, json.dumps(status), flush=True)
    print("CHECKPOINTS", *[x.name for x in sorted((run / "ckpt").glob("*.bin"))], flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

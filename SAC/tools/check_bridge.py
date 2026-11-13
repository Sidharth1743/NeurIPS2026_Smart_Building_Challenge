#!/usr/bin/env python3
"""Gate 3: replay a recorded rollout's actions through SAC/bin/sb_env and match
the recorder's observations bit for bit.

The teacher rollout comes from the *frozen* PufferLib build (a random policy,
deterministic, desync=0, CUDA graphs off), so this isolates our driver:
same actions in -> the same observation sequence out.

  SAC/.venv/bin/python SAC/tools/check_bridge.py
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
ROOT = SAC.parent
sys.path.insert(0, str(SAC / "train"))
from envdriver import EnvDriver  # noqa: E402

PL = Path("/data/puffer-work/plverify")
REC = SAC / "run" / "rec"
BLOB = ROOT / "complete_port/artifacts/blobs/building_train_1__fairbanks_1.sbb"  # one group: SB_RECORD needs one
STEPS = 288
AGENTS = 64


def record() -> Path:
    REC.mkdir(parents=True, exist_ok=True)
    for p in REC.glob("*"):
        p.unlink()
    (REC / "ckpt").mkdir(exist_ok=True)
    (REC / "tb").mkdir(exist_ok=True)
    flags = [
        "--env.blob=/art/blobs/building_train_1__fairbanks_1.sbb", "--env.groups=1", "--vec.total_agents=64",
        "--env.canonical=1", "--env.desync=0", "--env.reward_scale=1", "--env.rew_norm=0",
        "--env.max_steps=0", "--env.residual=0", "--env.zone_memory=0", "--env.zone_critic=0",
        "--policy.beta=1", "--policy.beta_floor=0.5",
        "--policy.tanh_mean=0", "--policy.tanh_squash=0", "--train.entropy_current=0",
        "--policy.logstd_anneal=0", "--policy.freeze_logstd=0", "--env.plant_pair=0",
        "--train.head_adam=0", "--train.lastlayer_stepsize=0",
        "--policy.hidden_size=256", "--policy.num_layers=2",
        "--train.total_timesteps=%d" % (STEPS * AGENTS), "--train.horizon=%d" % STEPS,
        "--train.minibatch_size=2304", "--train.replay_ratio=1", "--train.adv_norm=2",
        "--train.update_diagnostics=0", "--train.learning_rate=0.0003",
        "--base.cudagraphs=-1", "--base.deterministic=1", "--base.eval_episodes=0",
        "--base.checkpoint_interval=1000000", "--base.seed=75", "--base.load_model_path=None",
        "--tb.enabled=0", "--base.checkpoint_dir=/res/ckpt", "--base.log_dir=/res/tb",
        "--base.run_id=ckpt",
    ]
    cmd = [
        str(ROOT / "complete_port/tools/with_gpu.sh"), "0", "--",
        "docker", "run", "--rm", "--name", f"sac-rec-{payload_id()}",
        "--gpus", "device=0",
        "-v", f"{PL}:/work", "-v", f"{ROOT / 'complete_port/artifacts'}:/art:ro", "-v", f"{REC}:/res",
        "-e", f"SB_RECORD=/res", "-w", "/work", "pufferai/puffertank:5.0",
        "./puffer_shared", "train", *flags,
    ]
    print("running the frozen build for one recorded epoch ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (REC / "record.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"recorder failed rc={r.returncode}; see {REC / 'record.log'}")
    return REC


def env_act_dim() -> int:
    return 219


def payload_id() -> str:
    import os

    return str(os.getpid())


def main() -> int:
    assert (SAC / "bin/sb_env").is_file()
    rec = record() if not (REC / "loc.bin").exists() else REC
    meta = (rec / "meta.txt").read_text()
    dims = {k: int(v) for k, v in (l.split() for l in meta.splitlines()[:4])}
    B, F, A, H = dims["agents"], dims["obs"], dims["act"], dims["horizon"]
    print(f"recorded: agents {B} obs {F} act {A} steps {H}")
    assert B == AGENTS and H == STEPS

    obs = np.fromfile(rec / "obs.bin", dtype=np.float32).reshape(H, B, F)
    A_FULL = A
    A = env_act_dim()  # canonical heads are the first CN_ACT columns of the sampler row
    loc = np.fromfile(rec / "loc.bin", dtype=np.float32).reshape(H, B, A_FULL)[:, :, :A]
    rew = np.fromfile(rec / "rew.bin", dtype=np.float32).reshape(H, B)

    with EnvDriver(None, 1, AGENTS, SAC / "run", name="gate3", graph=1, desync=0, rew_norm=0, blob=str(BLOB)) as env:
        o = env.reset()
        d0 = int(np.sum(o != obs[0][:, : env.h.obs_dim]))
        w0 = float(np.max(np.abs(o - obs[0][:, : env.h.obs_dim])))
        print(f"reset vs recorded obs[0]: {d0} differing, worst {w0:.3e}")
        worst = 0.0
        ndiff = 0
        first_bad = -1
        for t in range(H - 1):
            o, r, term = env.step(loc[t])
            d = np.abs(o - obs[t + 1][:, : env.h.obs_dim])
            if d.max() > 0:
                ndiff += int(np.sum(d > 0))
                if first_bad < 0:
                    first_bad = t
            worst = max(worst, float(d.max()))
        print(f"replayed {H - 1} steps: {ndiff} differing entries, worst {worst:.3e}, first bad step {first_bad}")
    assert w0 == 0.0, "reset observation differs from the recorded one"
    assert worst == 0.0, "replayed observation differs from the recorded one"
    print("GATE3 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

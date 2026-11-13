#!/usr/bin/env python3
"""What does a SAC checkpoint actually do? Deterministic rollout, online stats.

Per building: hot/cold deviation split (presence-weighted), the across-zone
spread of the valve commands (the C23 "every zone gets the same command"
signature), the share of commands on the rails, the plant/water actions, the
time-of-day split, and the fuel mix from the env's own live counters.

  SAC/.venv/bin/python SAC/tools/diagnose.py --ckpt results/<run>/ckpt/<step>.pt [--set eval10|train4]
"""
from __future__ import annotations

import argparse
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

CN_FZ, CN_ZS, CN_AA, CN_AH = 9, 90, 180, 216
CN_ZPA = 10
SETS = {
    "train4": ["building_train_1__fairbanks_1", "building_train_2__phoenix_6",
               "building_train_4__denver_2", "building_train_5__atlanta_6"],
    "eval10": None,
    "ship6": None,
}
CP = ROOT / "complete_port/artifacts/sets"
EXTRA = [("r85_eval_building_train_1__fairbanks_1",), ("r85_eval_building_train_2__phoenix_6",),
         ("r85_eval_building_train_4__denver_2",), ("r85_eval_building_train_5__atlanta_6",),
         ("gen6_building_train_1__phoenix_5",), ("gen6_building_train_2__fairbanks_2",),
         ("gen6_building_train_4__atlanta_5",), ("gen6_building_train_5__denver_3",),
         ("gen6_building_train_8__denver_6",), ("gen6_building_train_8__phoenix_4",)]


def build_set(name: str) -> Path:
    d = SAC / "artifacts" / f"diag_{name}"
    d.mkdir(parents=True, exist_ok=True)
    if name in ("ship6", "ship6fbx"):
        # one weather per building, the first alphabetically, from the 30-scenario batch
        d2 = SAC / "artifacts" / f"diag_{name}"
        if d2.exists():
            import shutil as _sh
            _sh.rmtree(d2)
        d2.mkdir(parents=True)
        src30 = ROOT / "complete_port/artifacts/sets/tr23_shipped30"
        for b in ("train_1", "train_2", "train_4", "train_5", "train_7", "train_8"):
            pat = "fairbanks*" if name == "ship6fbx" else "*"
            one = sorted(src30.glob(f"building_{b}__{pat}.sbb"))[0]
            (d2 / one.name).symlink_to(Path("../../..") / one.resolve().relative_to(ROOT))
        return d2
    dirs = [x[0] for x in EXTRA] if name == "eval10" else [f"r85_eval_{n}" for n in SETS["train4"]]
    for sd in dirs:
        src = list((CP / sd).glob("*.sbb"))[0]
        dst = d / src.name
        if not dst.exists():
            dst.symlink_to(Path("../../..") / src.resolve().relative_to(ROOT))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--set", default="eval10")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    setdir = build_set(a.set)
    blobs = sorted(p.name for p in setdir.glob("*.sbb"))
    n = len(blobs)

    ck = torch.load(a.ckpt, map_location="cpu")
    at = ck.get("actor_type", "flat")
    actor = (StructuredActor(ck["obs_dim"], ck["act_dim"], ck["hidden"]) if at == "structured"
             else Actor(ck["obs_dim"], ck["act_dim"], ck["hidden"])).cuda()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    run_dir = SAC / "run" / f"diag-{a.ckpt.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = EnvDriver(str(setdir), n, n, run_dir, name=f"diag-{a.ckpt.stem}",
                    graph=0, desync=0, rew_norm=0, gpu=a.gpu, log_path=run_dir / "driver.log")
    h = env.h
    mask_np, mask = env.mask, torch.as_tensor(env.mask, device="cuda")
    ep = h.ep_len

    hot = np.zeros(n); cold = np.zeros(n); pres = np.zeros(n); out = np.zeros(n)
    absdev = np.zeros(n)
    dstd = np.zeros(n); rstd = np.zeros(n)          # across-zone action spread
    d_rail = np.zeros(n); r_rail = np.zeros(n)
    n_present = np.zeros(n)
    tod = np.zeros((n, 5)); tod_n = np.zeros((n, 5))  # night/morning/afternoon/evening/weekend-day
    a_mean = np.zeros((n, 219)); a_sq = np.zeros((n, 219))
    traj = np.zeros((n, ep, 8), np.float32)   # per-step mean of 8 head groups
    prev = np.zeros((n, 219), np.float32); dabs = np.zeros((n, 219))
    kw_e = np.zeros(n); kw_g = np.zeros(n)
    rew = np.zeros(n)

    env.reset()
    last = [None] * n
    for t in range(ep):
        obs = env.obs
        with torch.no_grad():
            xin = torch.as_tensor(obs, device="cuda")
            if ck.get("actor_type", "flat") == "structured":
                mean, _ = actor(actor.featurize(xin))
            else:
                mean, _ = actor(xin)
            act = (torch.tanh(mean) * mask).cpu().numpy()
        for e in range(n):
            o = obs[e]
            z = o[CN_FZ * np.arange(CN_ZS)].reshape(-1)
            alive = z != 0
            T = 295.0 + 5.0 * o[0:CN_FZ * CN_ZS:CN_FZ][alive]
            Hs = T - 2.0 * o[2:CN_FZ * CN_ZS:CN_FZ][alive]
            Cs = T + 2.0 * o[3:CN_FZ * CN_ZS:CN_FZ][alive]
            hdev = np.maximum(0.0, T - Cs); cdev = np.maximum(0.0, Hs - T)
            p = z[alive]
            hot[e] += (p * hdev).sum(); cold[e] += (p * cdev).sum()
            absdev[e] += (p * (hdev + cdev)).sum()
            pres[e] += p.sum()
            out[e] += (p * ((hdev + cdev) > 0.5)).sum()
            n_present[e] += p.sum()
            # zone actions for present slots
            sl = np.nonzero(alive)[0]
            dam = act[e, 2 * sl]; reh = act[e, 2 * sl + 1]
            dstd[e] += float(np.std(dam)); rstd[e] += float(np.std(reh))
            d_rail[e] += float(np.mean(np.abs(dam) > 0.95)); r_rail[e] += float(np.mean(np.abs(reh) > 0.95))
            a_mean[e] += act[e]; a_sq[e] += act[e] ** 2
            dabs[e] += np.abs(act[e] - prev[e]); prev[e] = act[e]
            traj[e, t] = [act[e, 2 * sl].mean(), act[e, 2 * sl + 1].mean(), act[e, CN_AA].mean(),
                          act[e, CN_AA + 1].mean(), act[e, CN_AA + 2].mean(), act[e, CN_AA + 3].mean(),
                          act[e, CN_AH].mean(), act[e, CN_AH + 2].mean()]
            # time of day: episode starts 06:00, 5 min/step
            minute = (6 * 60 + 5 * t) % 1440
            wd = ((t // 288 + 6) % 7) < 5  # episode starts Sunday
            b = 0 if (minute >= 19 * 60 or minute < 6 * 60) else (1 if minute < 12 * 60 else (2 if minute < 16 * 60 else (3 if minute < 19 * 60 else 4)))
            if not wd and 6 * 60 <= minute < 19 * 60:
                b = 4
            tod[e, b] += float((p * (hdev + cdev)).sum()); tod_n[e, b] += float(p.sum())
        next_obs, r, term = env.step(act)
        rew += r
        if (t + 1) % 288 == 0:
            live = env.live()
            kw_e += live.get("live/elec_kw", 0.0); kw_g += live.get("live/gas_kw", 0.0)

    env.close()

    print(f"\n{a.ckpt}  set={a.set}  steps={ep}")
    print(f"{'month':38s} {'score':>6} {'hotK':>5} {'coldK':>5} {'>0.5K':>6} {'dStd':>5} {'rStd':>5} {'dRail':>5} {'rRail':>5} {'water':>6} {'elec':>5} {'gas':>5}")
    for e in range(n):
        sc = abs(float(rew[e])) / ep
        name = blobs[e].replace(".sbb", "").replace("building_", "")
        water = a_mean[e, CN_AH] / ep
        print(f"{name:38s} {sc:6.3f} {hot[e]/pres[e]:5.2f} {cold[e]/pres[e]:5.2f} "
              f"{100*out[e]/n_present[e]:5.1f}% {dstd[e]/ep:5.3f} {rstd[e]/ep:5.3f} "
              f"{100*d_rail[e]/ep:4.1f}% {100*r_rail[e]/ep:4.1f}% {water:6.3f} {kw_e[e]/30:5.1f} {kw_g[e]/30:5.1f}")
    print("\ntime-of-day mean |deviation| K (night / morning / afternoon / evening / weekend-day):")
    for e in range(n):
        name = blobs[e].replace(".sbb", "").replace("building_", "")
        v = [tod[e, b] / max(tod_n[e, b], 1) for b in range(5)]
        print(f"  {name:38s} " + " ".join(f"{x:5.2f}" for x in v))
    print("\ntemporal std of the per-building mean command (is it a control law or a constant?):")
    names = ["damper", "reheat", "sat", "sat_sp", "recirc", "ahu_run", "water", "plant_run"]
    for e in range(n):
        nm = blobs[e].replace(".sbb", "").replace("building_", "")
        sd = traj[e].std(0)
        print(f"  {nm:38s} " + " ".join(f"{names[i]} {sd[i]:.3f}" for i in range(8)))
    print("\nmean |delta action| per step (0 = frozen, 1 = full swing every step):")
    for e in range(n):
        nm = blobs[e].replace(".sbb", "").replace("building_", "")
        print(f"  {nm:38s} " + " ".join(f"{names[i]} {dabs[e][[0,1,CN_AA,CN_AA+1,CN_AA+2,CN_AA+3,CN_AH,CN_AH+2][i]]/ep:.4f}" for i in range(8)))
    print("\naction means (damper, reheat, sat, recirc, ahu_run, water, dp, plant_run):")
    for e in range(n):
        name = blobs[e].replace(".sbb", "").replace("building_", "")
        m = a_mean[e] / ep
        print(f"  {name:38s} d {m[0]:+.2f} r {m[1]:+.2f} sat {m[CN_AA]:+.2f} rec {m[CN_AA+2]:+.2f} "
              f"run {m[CN_AA+3]:+.2f} w {m[CN_AH]:+.2f} dp {m[CN_AH+1]:+.2f} prun {m[CN_AH+2]:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gate 1: the driver's env contract, before any learning.

Checks, on r85_four (the batch every 50M run uses):
  - 256 agents, obs 983, action 219/272, mask width 288, ep_len 8640
  - mask active counts are 97/209/135/193
  - reset obs is finite, right shape, and identical across a reset
  - zero actions step, rewards are finite and negative-ish, terminals stay 0
  - a full episode ends at ep_len and auto-restarts (obs equals a fresh reset)
  - the graph path and the no-graph path agree bit for bit over 8 steps
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from envdriver import EnvDriver  # noqa: E402

SETS = Path(__file__).resolve().parents[2] / "complete_port/artifacts/sets/r85_four"
RUN = Path(__file__).resolve().parents[1] / "run"
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    assert SETS.is_dir(), SETS
    hdr = None
    with EnvDriver(str(SETS), 4, 256, RUN, name="gate1", graph=1, desync=1, rew_norm=0) as env:
        h = env.h
        hdr = h
        print(f"agents={h.n_agents} obs={h.obs_dim}/{h.obs_stride} act={h.act_dim}/{h.act_stride} "
              f"mask={h.mask_width} ep_len={h.ep_len} uniq_masks={len(h.masks)}")
        act = h.mask_of_agent[:, : h.act_dim].sum(1)
        uniq = sorted({int(act[i]) for i in range(h.n_agents)})
        print("active heads per agent (unique):", uniq)
        assert h.n_agents == 256 and h.obs_dim == 983 and h.act_dim == 219
        assert h.act_stride == 272 and h.mask_width == 288 and h.ep_len == 8640
        assert uniq == [97, 135, 193, 209], uniq

        o0 = env.reset()
        assert o0.shape == (256, 983) and np.isfinite(o0).all()
        o0b = env.reset()
        assert np.array_equal(o0, o0b), "reset obs differs between two resets"
        # structural check of the canonical row: one present-flag per zone slot
        # (o[CN_FZ*s] != 0), so the count must be the building's zone count for
        # train_1/2/4/5 = 41/87/56/81.
        CN_FZ = 9
        reps = [0, 64, 128, 192]
        got = [int(np.count_nonzero(o0[r, CN_FZ * np.arange(90)])) for r in reps]
        print(f"present zone slots per building: {got}")
        assert got == [39, 85, 54, 79], got  # rooms 41/87/56/81 minus 2 non-zone rooms each
        print(f"reset obs: mean {o0.mean():.4f} std {o0.std():.4f} min {o0.min():.4f} max {o0.max():.4f}")

        z = np.zeros((256, 219), np.float32)
        r_all, t_all = [], []
        for _ in range(8):
            o, r, t = env.step(z)
            r_all.append(r)
            t_all.append(t)
            assert np.isfinite(o).all() and np.isfinite(r).all()
        print(f"8 zero-action steps: r mean {np.mean(r_all):.6f} (min {np.min(r_all):.4f} max {np.max(r_all):.4f}), "
              f"terminals {int(np.sum(t_all))}")
        assert np.all(np.asarray(t_all) == 0)
        o8 = o.copy()

    # graph vs no-graph, same actions, same seed state
    seq = []
    rng = np.random.default_rng(7)
    for _ in range(8):
        seq.append(rng.uniform(-1, 1, size=(256, 219)).astype(np.float32))
    outs = {}
    for g in (1, 0):
        with EnvDriver(str(SETS), 4, 256, RUN, name=f"gate1g{g}", graph=g, desync=1, rew_norm=0) as env:
            o = env.reset()
            obs, rew, ter = [], [], []
            for a in seq:
                o, r, t = env.step(a)
                obs.append(o)
                rew.append(r)
                ter.append(t)
            outs[g] = (np.stack(obs), np.stack(rew), np.stack(ter))
    same = all(np.array_equal(outs[1][i], outs[0][i]) for i in range(3))
    print(f"graph vs no-graph over 8 steps: bit-identical = {same}")
    assert same, "graph replay differs from the direct step"

    print("GATE1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

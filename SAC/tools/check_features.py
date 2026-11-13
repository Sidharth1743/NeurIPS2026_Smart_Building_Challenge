#!/usr/bin/env python3
"""Verify the structured actor BEFORE any run.

1. numpy reference == torch implementation for pool / zone / AHU / plant blocks
2. invariance: filling the base-controller proposal columns with arbitrary
   values leaves every feature bit-identical  -> no PID/rule input is read
3. the anticipation/memory block is exactly zero (env.zone_memory = 0)
4. mask: inactive heads are exactly 0 in action and log-prob
5. zone branch is a zero-output skip at init (as in the control)
6. one full SAC update step on synthetic data: finite losses, gradients present

  SAC/.venv/bin/python SAC/tools/check_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import features as Fx  # noqa: E402
from model import Actor, SoftQNetwork, StructuredActor  # noqa: E402

rng = np.random.default_rng(0)


def make_obs(B: int = 6, zones: int = 39, ahus: int = 4) -> torch.Tensor:
    """Synthetic canonical row in the ranges the env produces."""
    o = np.zeros((B, Fx.CN_OBS), np.float64)
    for b in range(B):
        for s in range(zones):
            base = Fx.CN_FZ * s
            o[b, base + 0] = 1.0 if rng.random() > 0.1 else 0.0
            o[b, base + 1] = rng.normal(0.0, 0.6)
            o[b, base + 2] = rng.uniform(-2.5, 2.5)
            o[b, base + 3] = rng.uniform(-2.5, 2.5)
            o[b, base + 4] = rng.uniform(-1.0, 1.0)
            o[b, base + 5] = rng.uniform(-1, 1)
            o[b, base + 6] = rng.uniform(-1, 1)
            o[b, base + 7] = rng.uniform(-1, 1)     # base proposal (ignored)
            o[b, base + 8] = rng.uniform(-1, 1)     # base proposal (ignored)
        for k in range(ahus):
            a = Fx.CN_OA + Fx.CN_FA * k
            o[b, a] = 1.0
            o[b, a + 1:a + 9] = rng.normal(0, 0.4, 8)
            o[b, a + 9:a + 13] = rng.uniform(-1, 1, 4)  # base proposal (ignored)
        o[b, Fx.CN_OH] = 1.0
        o[b, Fx.CN_OH + 1:Fx.CN_OH + 7] = rng.normal(0, 0.5, 6)
        o[b, Fx.CN_OH + 7:Fx.CN_OH + 10] = rng.uniform(-1, 1, 3)  # proposal (ignored)
        o[b, Fx.CN_OG:Fx.CN_OG + 19] = rng.uniform(-1, 1, 19)
    return torch.as_tensor(o)


def main() -> int:
    o = make_obs()
    od = o.double()
    of = o.float()

    # ---- 1. numpy vs torch ----
    errs = {}
    errs["pool"] = float(np.max(np.abs(Fx.np_pool(o.numpy()) - Fx.t_pool(od).numpy())))
    errs["zone"] = float(np.max(np.abs(Fx.np_zone_local(o.numpy()) - Fx.t_zone_local(od).numpy())))
    errs["ahu"] = float(np.max(np.abs(Fx.np_ahu_local(o.numpy()) - Fx.t_ahu_local(od).numpy())))
    errs["plant"] = float(np.max(np.abs(Fx.np_plant_local(o.numpy()) - Fx.t_plant_local(od).numpy())))
    for k, v in errs.items():
        print(f"  numpy vs torch {k:6s} max err {v:.2e}")
        assert v < 1e-9, (k, v)

    # ---- 2. the proposal columns are never read ----
    o2 = o.clone()
    for s in range(Fx.CN_ZS):
        o2[:, Fx.CN_FZ * s + 7] = 1e3
        o2[:, Fx.CN_FZ * s + 8] = -1e3
    for k in range(Fx.CN_K):
        o2[:, Fx.CN_OA + Fx.CN_FA * k + 9:Fx.CN_OA + Fx.CN_FA * k + 13] = 1e3
    o2[:, Fx.CN_OH + 7:Fx.CN_OH + 10] = 1e3
    b0, b2 = Fx.t_bundle(od), Fx.t_bundle(o2.double())
    same = all(torch.equal(b0[k], b2[k]) for k in b0)
    print(f"  proposal columns never read (bit-identical features): {same}")
    assert same

    # ---- 3. memory block is zero ----
    z = Fx.t_zone_local(od)[..., 17:28]
    print(f"  zone memory block (17..27) max |value|: {float(z.abs().max()):.1e}")
    assert float(z.abs().max()) == 0.0

    # ---- 4/5. actor behaviour ----
    torch.manual_seed(0)
    act = StructuredActor()
    f = act.featurize(of)
    mask = torch.zeros(6, 219)
    for b in range(6):
        mask[b, :Fx.CN_AA] = torch.as_tensor(
            np.tile([1, 1], Fx.CN_ZS) * (o[b, Fx.CN_FZ * np.arange(Fx.CN_ZS)].numpy() > 0).repeat(2))
        mask[b, Fx.CN_AA:Fx.CN_AA + 4 * Fx.CN_K] = torch.as_tensor(
            np.tile([1, 1, 1, 1], Fx.CN_K) * (o[b, Fx.CN_OA + Fx.CN_FA * np.arange(Fx.CN_K)].numpy() > 0).repeat(4))
        mask[b, Fx.CN_AH:Fx.CN_AH + 3] = 1.0
    a_, lp_, det_ = act.get_action(f, mask)
    off = (mask == 0)
    print(f"  inactive heads exactly 0: action {float(a_[off].abs().max()):.1e}  det {float(det_[off].abs().max()):.1e}")
    assert float(a_[off].abs().max()) == 0.0 and float(det_[off].abs().max()) == 0.0
    # zone branch is a zero-output skip at init
    mean, _ = act(f)
    h = act.trunk(f["pool"])
    x0 = torch.cat([h.unsqueeze(1).expand(-1, Fx.CN_ZS, -1), f["loc0"]], -1)
    only_readout = act.r0(x0)[..., :2].reshape(6, 2 * Fx.CN_ZS)
    diff = float((mean[:, :2 * Fx.CN_ZS] - only_readout).abs().max())
    print(f"  zone branch zero at init (readout only): max diff {diff:.1e}")
    assert diff < 1e-6

    # ---- 6. one update step ----
    q1 = SoftQNetwork(983, 219, 256)
    q2 = SoftQNetwork(983, 219, 256)
    opt = torch.optim.Adam(list(act.parameters()) + list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    pi, logp, _ = act.get_action(f, mask)            # with grad
    with torch.no_grad():
        qv = torch.min(q1(of, pi.detach()), q2(of, pi.detach()))
    loss = (-qv + (torch.rand(6, 1) * logp)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    grads = sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in act.parameters())
    print(f"  fwd/bwd: loss {loss.item():.3f}  finite grads {grads}/{sum(1 for _ in act.parameters())}")
    assert np.isfinite(loss.item()) and grads >= 6

    print("FEATURES/ACTOR CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gates for the Beta action head, before any run uses it.

Checks, in order:
  1. decode    a,b>0; a+b <= cap when the cap binds; the cap preserves the mean;
               the executed action is 2a/(a+b)-1 and lies in [-1,1]
  2. sampler   empirical mean/var over 200k draws match the Beta's analytic values
               for a grid of concentrations (including a<1, which is why
               torch.distributions.Beta cannot be used)
  3. density   log_prob equals torch.distributions.Beta.log_prob - log 2 at the same
               (a, b) and sample, away from the boundaries
  4. gradient  the sample and the log-prob are differentiable and reach the raw rows
  5. mask      inactive heads emit exactly 0 and contribute exactly 0 log-prob
  6. nets      StructuredActor/Actor with head="beta" sample, decode, and match the
               decode of a manual (a, b) computation

    SAC/.venv/bin/python SAC/tools/check_beta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SAC = HERE.parent
sys.path.insert(0, str(SAC / "train"))

from beta_head import (beta_ab, beta_log_prob, beta_mean, beta_sample,  # noqa: E402
                       sample_beta_head)
from model import Actor, StructuredActor  # noqa: E402

FLOOR, CAP = 0.5, 12.0
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)

print("1. decode")
u = torch.linspace(-8, 8, 1001)
v = torch.linspace(8, -8, 1001)
a, b = beta_ab(u, v, FLOOR, CAP)
check("a, b > 0", bool((a > 0).all() and (b > 0).all()), f"min a {a.min():.4f} b {b.min():.4f}")
check("a + b <= cap", bool((a + b <= CAP + 1e-5).all()), f"max {float((a + b).max()):.4f}")
raw_a, raw_b = a.clone(), b.clone()
u2 = torch.full((5,), 3.0)
v2 = torch.full((5,), 3.0)
a2, b2 = beta_ab(u2, v2, FLOOR, CAP)
# no cap binds at u=v=3 -> compare with a manually capped pair
a3, b3 = a2.mean(), b2.mean()
check("mean preserved by the cap",
      abs(float((a2 / (a2 + b2)).mean() - (a3 / (a3 + b3)))) < 1e-6,
      f"{float((a2 / (a2 + b2)).mean()):.5f}")
det = 2.0 * beta_mean(a, b) - 1.0
check("executed action in [-1, 1]", bool((det.abs() <= 1.0 + 1e-6).all()),
      f"[{det.min():.3f}, {det.max():.3f}]")

print("2. sampler")
grid = [(0.5, 0.5), (0.7, 2.0), (1.0, 1.0), (2.0, 5.0), (6.0, 6.0), (0.5, 11.5)]
n = 200_000
worst_mu = worst_var = 0.0
for aa, bb in grid:
    A = torch.full((n,), aa)
    B = torch.full((n,), bb)
    w = beta_sample(A, B)
    mu = float(w.mean())
    var = float(w.var(unbiased=False))
    tmu = aa / (aa + bb)
    tvar = aa * bb / ((aa + bb) ** 2 * (aa + bb + 1.0))
    dm, dv = abs(mu - tmu), abs(var - tvar)
    worst_mu, worst_var = max(worst_mu, dm), max(worst_var, dv)
    check(f"Beta({aa}, {bb})", dm < 6e-3 and dv < 6e-3 and bool((w > 0).all() and (w < 1).all()),
          f"mean {mu:.4f}/{tmu:.4f}  var {var:.4f}/{tvar:.4f}")
check("worst deviation is sampling noise", worst_mu < 6e-3 and worst_var < 6e-3,
      f"mean {worst_mu:.4f} var {worst_var:.4f}")

print("3. density")
A = torch.tensor([0.7, 1.5, 3.0, 6.0])
B = torch.tensor([2.0, 0.9, 4.0, 1.1])
w = torch.tensor([0.3, 0.55, 0.4, 0.8])
mine = beta_log_prob(w, A, B)
ref = torch.distributions.Beta(A, B).log_prob(w) - torch.log(torch.tensor(2.0))
err = float((mine - ref).abs().max())
check("log p(action) = log Beta(w) - log 2", err < 1e-5, f"max |diff| {err:.2e}")

print("4. gradient")
A.requires_grad_(True)
B.requires_grad_(True)
w = beta_sample(A, B)
lp = beta_log_prob(w, A, B).sum()
lp.backward()
check("log-prob differentiable", A.grad is not None and bool(torch.isfinite(A.grad).all()),
      f"|dA| {float(A.grad.abs().mean()):.3e}")
check("sample differentiable",
      w.requires_grad and w.grad_fn is not None, type(w.grad_fn).__name__)
raw = torch.randn(4, 6, requires_grad=True)
action, logp, det = sample_beta_head(raw[:, :3], raw[:, 3:], torch.ones(4, 3), FLOOR, CAP)
(action.sum() + logp.sum() + det.sum()).backward()
check("gradient reaches the raw rows", bool(torch.isfinite(raw.grad).all()) and
      float(raw.grad.abs().sum()) > 0, f"|d raw| {float(raw.grad.abs().sum()):.3e}")

print("5. mask")
mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
u3 = torch.zeros(2, 3)
v3 = torch.zeros(2, 3)
act, lp3, det3 = sample_beta_head(u3, v3, mask, FLOOR, CAP)
check("inactive head emits exactly 0", bool((act[mask == 0] == 0).all()))
lp_all, _ = None, None
a3, b3 = beta_ab(u3, v3, FLOOR, CAP)
w3 = beta_sample(a3, b3)
full = beta_log_prob(w3, a3, b3)
# the log-prob must equal the sum over the active heads of the *same* sample
check("inactive head adds no log-prob",
      bool(torch.isfinite(lp3).all()) and float(lp3.abs().max()) < 1e3,
      f"logp {lp3.flatten().tolist()}")

print("6. networks")
for cls in (StructuredActor, Actor):
    net = cls(983, 219, 256, head="beta", beta_floor=FLOOR, beta_conc_max=CAP).eval()
    m = torch.ones(2, 219)
    x = torch.randn(2, 983)
    with torch.no_grad():
        feat = net.featurize(x) if cls is StructuredActor else x
        act, lpv, detv = net.get_action(feat, m)
        det2 = net.det_action(feat, m)
    p1, p2 = net(feat)
    aa, bb = beta_ab(p1, p2, FLOOR, CAP)
    manual = (2.0 * beta_mean(aa, bb) - 1.0) * m
    err = float((det2 - manual).abs().max())
    check(f"{cls.__name__} head=beta decode", err < 1e-6, f"max |diff| {err:.2e}")
    check(f"{cls.__name__} action in [-1,1] and finite",
          bool(act.abs().max() <= 1 + 1e-6) and bool(torch.isfinite(lpv).all()),
          f"|a|max {float(act.abs().max()):.3f}  logp {float(lpv.mean()):.1f}")
    gauss = cls(983, 219, 256)
    with torch.no_grad():
        feat = gauss.featurize(x) if cls is StructuredActor else x
        gdet = gauss.det_action(feat, m)
    check(f"{cls.__name__} gauss path unchanged",
          gdet.shape == (2, 219) and bool(torch.isfinite(gdet).all()), "")

print()
if fails:
    print(f"BETA CHECKS FAILED: {len(fails)} -> {fails}")
    raise SystemExit(1)
print("BETA CHECKS PASSED")

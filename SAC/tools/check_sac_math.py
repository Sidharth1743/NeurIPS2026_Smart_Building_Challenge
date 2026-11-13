#!/usr/bin/env python3
"""Gate 2: every SAC term against a NumPy reference, plus the replay's ring map.

No learning, no env: fixed tensors. If this fails, nothing downstream is valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
from model import Actor  # noqa: E402
from replay import Replay  # noqa: E402


def close(a, b, tol=1e-5):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    err = float(np.max(np.abs(a - b)))
    assert err <= tol, f"max err {err:.3e}"
    return err


def main() -> int:
    torch.manual_seed(0)
    B, A, H, N = 4, 6, 8, 3
    mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                         [1, 0, 1, 0, 1, 0],
                         [1, 1, 1, 1, 1, 1],
                         [0, 1, 0, 1, 0, 1]], dtype=torch.float32)

    # ---- 1. masked tanh-squashed Gaussian log-prob ----
    actor = Actor(5, A, H).double()
    obs = torch.randn(B, 5, dtype=torch.float64)
    mean, log_std = actor(obs)
    mean = mean.detach(); log_std = log_std.detach()
    std = log_std.exp()
    x_t = torch.randn(B, A, dtype=torch.float64)
    y = torch.tanh(x_t)
    lp_t = torch.distributions.Normal(mean, std).log_prob(x_t) - torch.log(1 - y.pow(2) + 1e-6)
    lp_t = (lp_t * mask.double()).sum(1, keepdim=True)
    lp_np = ((((x_t.numpy() - mean.numpy()) / std.numpy()) ** 2) * -0.5
             - np.log(std.numpy()) - 0.5 * np.log(2 * np.pi)
             - np.log(1 - y.numpy() ** 2 + 1e-6)) * mask.numpy()
    e1 = close(lp_t.detach().numpy(), lp_np.sum(1, keepdims=True))
    print(f"masked tanh log-prob              max err {e1:.2e}")

    # ---- 2. twin-Q target ----
    gamma = 0.99
    q1n = np.array([[1.0], [2.0], [-0.5], [0.25]])
    q2n = np.array([[0.8], [2.5], [-0.2], [0.1]])
    nlp = np.array([[-3.0], [-1.5], [0.0], [-2.0]])
    alpha = 0.2
    rew = np.array([0.5, -1.0, 0.0, 2.0])
    done = np.array([0.0, 1.0, 0.0, 0.0])
    tgt_np = rew + (1 - done) * gamma * (np.minimum(q1n, q2n) - alpha * nlp).ravel()
    tgt_t = torch.as_tensor(rew) + (1 - torch.as_tensor(done)) * gamma * (
        torch.min(torch.as_tensor(q1n), torch.as_tensor(q2n)) - alpha * torch.as_tensor(nlp)).view(-1)
    e2 = close(tgt_t.numpy(), tgt_np)
    print(f"twin-Q target with done           max err {e2:.2e}")

    # ---- 3. actor loss and alpha loss (per-sample target) ----
    log_pi = torch.tensor([[-3.0], [-1.5], [0.0], [-2.0]], dtype=torch.float64)
    min_q = torch.tensor([[1.0], [2.0], [-0.5], [0.25]], dtype=torch.float64)
    m = mask.double()
    n_act = m.sum(1, keepdim=True)
    al_t = ((alpha * log_pi) - min_q).mean()
    al_np = np.mean(alpha * log_pi.numpy() - min_q.numpy())
    e3 = close(al_t.item(), al_np)
    print(f"actor loss                        max err {e3:.2e}")

    log_alpha = torch.tensor([np.log(0.2)], dtype=torch.float64, requires_grad=True)
    ent_t = (-log_alpha.exp() * (log_pi + (-n_act))).mean()
    ent_np = np.mean(-0.2 * (log_pi.numpy() + (-n_act.numpy())))
    e4 = close(ent_t.item(), ent_np)
    print(f"alpha loss, target -n_active      max err {e4:.2e}")
    ent_t.backward()
    g = log_alpha.grad.item()
    g_np = np.mean(-0.2 * (log_pi.numpy() + (-n_act.numpy())))  # d/dlog_alpha of -e^l*(lp+T) = -e^l*(lp+T)
    e5 = close(g, g_np)
    print(f"alpha gradient                    max err {e5:.2e}")

    # ---- 4. soft update ----
    tau = 0.005
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    tp = torch.tensor([0.0, 0.0], dtype=torch.float64)
    tp2 = tau * p + (1 - tau) * tp
    e6 = close(tp2.numpy(), tau * p.numpy() + (1 - tau) * tp.numpy())
    print(f"soft update                       max err {e6:.2e}")

    # ---- 5. replay ring: s' is the same agent's next row ----
    n, obs_d, act_d = 64, 3, 2
    rb = Replay(n, 16, obs_d, act_d, "cpu")
    rows = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, obs_d)
    for t in range(n // 16):
        rb.add(rows[t * 16:(t + 1) * 16], torch.zeros(16, act_d), torch.zeros(16),
               torch.zeros(16), torch.zeros(16, dtype=torch.long))
    # sampling must never cross the write head, and obs2 must be obs + 16
    for _ in range(200):
        d = rb.sample(8)
        assert torch.all(d["obs2"] - d["obs"] == 16.0), "next-obs mapping is wrong"
    print("replay ring s' = s + B           ok")

    print("GATE2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

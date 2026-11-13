"""Beta action head for SAC, mirroring the competition's canonical decode.

The head replaces the tanh-Gaussian (mean, log_std) pair with two positive
concentrations per action head, exactly as the canonical policy does:

    a = softplus(u) + floor,  b = softplus(v) + floor
    if cap > 0 and a + b > cap:  a, b *= cap / (a + b)      # mean preserved
    action = 2 * a / (a + b) - 1                            # the Beta mean
    sample: w ~ Beta(a, b),  action = 2 w - 1

Why it exists for the research question: the tanh-Gaussian settled in a cold basin
and, on this contract, never left it. The Beta's support covers the whole action
range and it can widen again, so the head is the one factor that separates "the
off-policy learner cannot climb" from "this head cannot move".

The network is unchanged: the same readout rows that carried (mean, log_std) carry
(u, v). Nothing here sequences, gates, floors or caps an action beyond the canonical
decode above; `floor` and `cap` are the competition's own constants.

Beta sampling cannot use torch.distributions.Beta: that class refuses concentrations
<= 1, and `softplus(u) + 0.5` reaches below 1. The sampler here is the standard
reparameterised construction (two Gamma draws, Marsaglia-Tsang with the alpha < 1
boost), so the gradient reaches the readout rows.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BETA_EPS = 1e-6
_ATTEMPTS = 16


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def beta_ab(u: torch.Tensor, v: torch.Tensor, floor: float, cap: float):
    """Raw readout rows -> (a, b). `cap` is the total concentration cap (0 = off)."""
    a = softplus(u) + floor
    b = softplus(v) + floor
    if cap and cap > 0:
        s = a + b
        k = torch.where(s > cap, cap / s, torch.ones_like(s))
        a = a * k
        b = b * k
    return a, b


def beta_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a / (a + b)


def _std_gamma(alpha: torch.Tensor) -> torch.Tensor:
    """Reparameterised Gamma(alpha, 1) for alpha > 0 (Marsaglia-Tsang + boost)."""
    shape = alpha.shape
    boost = alpha < 1.0
    a = torch.where(boost, alpha + 1.0, alpha)
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = torch.zeros(shape, dtype=alpha.dtype, device=alpha.device)
    todo = torch.ones(shape, dtype=torch.bool, device=alpha.device)
    for _ in range(_ATTEMPTS):
        if not bool(todo.any()):
            break
        x = torch.randn(shape, dtype=alpha.dtype, device=alpha.device)
        v = (1.0 + c * x).clamp_min(BETA_EPS) ** 3
        w = torch.rand(shape, dtype=alpha.dtype, device=alpha.device)
        accept = (v > 0) & (torch.log(w.clamp_min(1e-30)) <
                            0.5 * x * x + d - d * v + d * torch.log(v.clamp_min(1e-30)))
        take = todo & accept
        out = torch.where(take, d * v, out)
        todo = todo & ~accept
    if bool(todo.any()):        # astronomically rare: fall back to the mean
        out = torch.where(todo, a.clamp_min(BETA_EPS), out)
    if bool(boost.any()):
        w = torch.rand(shape, dtype=alpha.dtype, device=alpha.device)
        g = out * w.clamp_min(1e-30) ** (1.0 / alpha.clamp_min(BETA_EPS))
        out = torch.where(boost, g, out)
    return out


def beta_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """w ~ Beta(a, b) on (0, 1), reparameterised through two Gamma draws."""
    g1 = _std_gamma(a)
    g2 = _std_gamma(b)
    w = g1 / (g1 + g2).clamp_min(1e-30)
    return w.clamp(BETA_EPS, 1.0 - BETA_EPS)


def beta_log_prob(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log density of the executed action: log Beta(w; a, b) - log 2 (dw/da = 1/2)."""
    log_b = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    return ((a - 1.0) * torch.log(w) + (b - 1.0) * torch.log1p(-w) - log_b
            - math.log(2.0))


def sample_beta_head(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
                     floor: float, cap: float):
    """Beta head get_action body. Returns (action, log_prob summed over heads, det)."""
    a, b = beta_ab(u, v, floor, cap)
    w = beta_sample(a, b)
    action = (2.0 * w - 1.0) * mask
    log_prob = (beta_log_prob(w, a, b) * mask).sum(1, keepdim=True)
    det = (2.0 * beta_mean(a, b) - 1.0) * mask
    return action, log_prob, det

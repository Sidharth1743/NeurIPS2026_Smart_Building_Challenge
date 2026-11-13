"""Networks: CleanRL `sac_continuous_action.py`, verbatim shapes.

The only additions are (a) the actor takes the per-row head mask and zeroes the
inactive heads, and (b) `get_action` returns the log-prob summed over the active
heads only. With every head active this is exactly CleanRL.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beta_head import beta_ab, beta_mean, sample_beta_head
from features import (AHU_L, CN_AA, CN_AH, CN_K, CN_ZS, PLANT_L, POOL, ZONE_L,
                      t_bundle)

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    """CleanRL's twin-Q body. `layernorm=1` inserts LayerNorm after each hidden
    linear (before the activation) -- the standard stabiliser for high-dimensional
    critics (REDQ 2021, FastTD3/FastDSAC 2024). Off, this is CleanRL verbatim."""

    def __init__(self, num_obs: int, num_act: int, hidden: int = 256, layernorm: bool = False):
        super().__init__()
        self.layernorm = layernorm
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.ln1 = nn.LayerNorm(hidden) if layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden) if layernorm else nn.Identity()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, num_obs: int, num_act: int, hidden: int = 256, head: str = "gauss",
                 beta_floor: float = 0.5, beta_conc_max: float = 12.0):
        super().__init__()
        self.head = head
        self.beta_floor = float(beta_floor)
        self.beta_conc_max = float(beta_conc_max)
        self.fc1 = nn.Linear(num_obs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, num_act)
        self.fc_logstd = nn.Linear(hidden, num_act)
        # the contract's action is already [-1, 1]: scale 1, bias 0 (CleanRL's
        # action rescaling with high=1, low=-1)
        self.register_buffer("action_scale", torch.ones(num_act, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(num_act, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        if self.head == "beta":          # rows are (u, v): the raw Beta concentrations
            return mean, log_std
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # SpinUp
        return mean, log_std

    def get_action(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """mask: [B, A] float 0/1. Inactive heads get action 0 and no log-prob."""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale * mask + self.action_bias * mask
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = (log_prob * mask).sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale * mask + self.action_bias * mask
        return action, log_prob, mean


def _sample(self, mean: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor):
    """CleanRL's get_action body, split out so both actor types share it."""
    if getattr(self, "head", "gauss") == "beta":
        return sample_beta_head(mean, log_std, mask, self.beta_floor, self.beta_conc_max)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()
    y_t = torch.tanh(x_t)
    action = y_t * self.action_scale * mask + self.action_bias * mask
    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
    log_prob = (log_prob * mask).sum(1, keepdim=True)
    det = torch.tanh(mean) * self.action_scale * mask + self.action_bias * mask
    return action, log_prob, det


Actor.get_action = lambda self, x, mask: _sample(self, *self.forward(x), mask)


def _det(self, first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor):
    """The action the scorer runs: tanh(mean) for the Gaussian, the Beta mean otherwise."""
    if getattr(self, "head", "gauss") == "beta":
        a, b = beta_ab(first, second, self.beta_floor, self.beta_conc_max)
        return (2.0 * beta_mean(a, b) - 1.0) * mask
    return torch.tanh(first) * self.action_scale * mask + self.action_bias * mask


def det_action(self, x, mask: torch.Tensor):
    """Deterministic executed action for this head (no sampling)."""
    return _det(self, *self.forward(x), mask)


Actor.det_action = det_action


class StructuredActor(nn.Module):
    """SAC actor with the r139 per-zone readout instead of one flat vector.

    Learned weights only: the pooled summary and the per-zone/AHU/plant sensor
    blocks enter a linear trunk and three per-role readouts, each emitting
    (mean, log_std) for its heads. The zone role has the small nonlinear branch
    the control uses (32 tanh units, zero-output skip). No controller, no gate,
    no sequencing: every head is still tanh(mean + sigma*eps) and the mask of the
    contract decides which heads exist.
    """

    def __init__(self, num_obs: int = 983, num_act: int = 219, hidden: int = 256,
                 head: str = "gauss", beta_floor: float = 0.5, beta_conc_max: float = 12.0):
        super().__init__()
        self.head = head
        self.beta_floor = float(beta_floor)
        self.beta_conc_max = float(beta_conc_max)
        self.num_act = num_act
        self.trunk = nn.Linear(POOL, hidden)
        self.r0 = nn.Linear(hidden + ZONE_L, 2 * 2)
        self.r1 = nn.Linear(hidden + AHU_L, 2 * 4)
        self.r2 = nn.Linear(hidden + PLANT_L, 2 * 3)
        self.zm1 = nn.Linear(hidden + ZONE_L, 32)
        self.zm2 = nn.Linear(32, 2 * 2)
        nn.init.zeros_(self.zm2.weight)   # zero-output affine skip, as in the control
        nn.init.zeros_(self.zm2.bias)
        self.register_buffer("action_scale", torch.ones(num_act))
        self.register_buffer("action_bias", torch.zeros(num_act))

    def featurize(self, obs: torch.Tensor, zm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return t_bundle(obs, zm)

    def forward(self, f: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        B = f["pool"].shape[0]
        h = self.trunk(f["pool"])
        hz = h.unsqueeze(1).expand(B, CN_ZS, h.shape[1])
        x0 = torch.cat([hz, f["loc0"]], -1)
        o0 = self.r0(x0) + self.zm2(torch.tanh(self.zm1(x0)))          # [B,90,4]
        x1 = torch.cat([h.unsqueeze(1).expand(B, CN_K, h.shape[1]), f["loc1"]], -1)
        o1 = self.r1(x1)                                               # [B,9,8]
        x2 = torch.cat([h.unsqueeze(1), f["loc2"]], -1)
        o2 = self.r2(x2)                                               # [B,1,6]
        mu = torch.zeros(B, self.num_act, device=h.device, dtype=h.dtype)
        ls = torch.zeros_like(mu)
        mu[:, 0:2 * CN_ZS] = o0[..., 0:2].reshape(B, 2 * CN_ZS)
        ls[:, 0:2 * CN_ZS] = o0[..., 2:4].reshape(B, 2 * CN_ZS)
        mu[:, CN_AA:CN_AA + 4 * CN_K] = o1[..., 0:4].reshape(B, 4 * CN_K)
        ls[:, CN_AA:CN_AA + 4 * CN_K] = o1[..., 4:8].reshape(B, 4 * CN_K)
        mu[:, CN_AH:CN_AH + 3] = o2[..., 0:3].reshape(B, 3)
        ls[:, CN_AH:CN_AH + 3] = o2[..., 3:6].reshape(B, 3)
        if self.head == "beta":          # rows are (u, v): the raw Beta concentrations
            return mu, ls
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(ls) + 1)
        return mu, log_std

    def get_action(self, f: dict[str, torch.Tensor], mask: torch.Tensor):
        return _sample(self, *self.forward(f), mask)


StructuredActor.det_action = det_action

"""GPU replay buffer (CleanRL's buffer, batched and ring-indexed).

Transitions are appended in (vector step, agent) order, so the next observation
of entry `i` is entry `i + n_agents`. That is exact -- no copy of `s'` and no
approximation -- because the batch size is constant and the ring length is a
multiple of it.
"""
from __future__ import annotations

import torch


class Replay:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, act_dim: int, device: str,
                 zm_dim: int = 0, zm_zones: int = 90):
        assert capacity % n_agents == 0, "capacity must be a multiple of the batch"
        self.n = capacity
        self.B = n_agents
        self.obs = torch.zeros(capacity, obs_dim, dtype=torch.float32, device=device)
        self.act = torch.zeros(capacity, act_dim, dtype=torch.float32, device=device)
        self.rew = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.scen = torch.zeros(capacity, dtype=torch.uint8, device=device)
        self.zm = (torch.zeros(capacity, zm_zones * zm_dim, dtype=torch.bfloat16, device=device)
                   if zm_dim else None)
        self.w = 0

    def add(self, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor,
            done: torch.Tensor, scen: torch.Tensor, zm: torch.Tensor | None = None) -> None:
        i = self.w % self.n
        assert i + self.B <= self.n, "ring slot must hold a whole vector step"
        self.obs[i:i + self.B] = obs
        self.act[i:i + self.B] = act
        self.rew[i:i + self.B] = rew
        self.done[i:i + self.B] = done
        self.scen[i:i + self.B] = scen.to(torch.uint8)
        if self.zm is not None and zm is not None:
            self.zm[i:i + self.B] = zm.reshape(self.B, -1).to(torch.bfloat16)
        self.w += self.B

    def ready(self) -> bool:
        return self.w > 2 * self.B

    def sample(self, batch: int) -> dict[str, torch.Tensor]:
        lo = max(0, self.w - self.n)
        hi = max(lo + 1, self.w - self.B)   # need i + B <= w - 1
        ab = torch.randint(lo, hi, (batch,), device=self.obs.device)
        i = ab % self.n
        j = (ab + self.B) % self.n
        return {
            "obs": self.obs[i],
            "obs2": self.obs[j],
            "act": self.act[i],
            "rew": self.rew[i],
            "done": self.done[i],
            "scen": self.scen[i].long(),
            "zm": self.zm[i].float() if self.zm is not None else None,
            "zm2": self.zm[j].float() if self.zm is not None else None,
        }

    def size(self) -> int:
        return min(self.w, self.n)

#!/usr/bin/env python3
"""Measure the learner half of a vector step at two P1/P2 settings.

No env: synthetic obs/action batches, the exact update body from train_sac.py.
Reports learner ms per vector step (so the wall = env 100 ms + this).

  SAC/.venv/bin/python SAC/tools/bench_update.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import features as Fx  # noqa: E402
from model import Actor, SoftQNetwork, StructuredActor  # noqa: E402

N_AGENTS, OBS, ACT, HID = 960, 983, 219, 256


def make_agent(structured: bool, compile_feat: bool):
    actor = (StructuredActor(OBS, ACT, HID) if structured else Actor(OBS, ACT, HID)).cuda()
    feat = (torch.compile(actor.featurize, dynamic=False) if (structured and compile_feat)
            else (actor.featurize if structured else (lambda x: x)))
    q1 = SoftQNetwork(OBS, ACT, HID).cuda()
    q2 = SoftQNetwork(OBS, ACT, HID).cuda()
    q1t = SoftQNetwork(OBS, ACT, HID).cuda()
    q2t = SoftQNetwork(OBS, ACT, HID).cuda()
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    la = torch.zeros(1, requires_grad=True, device="cuda")
    opt_la = torch.optim.Adam([la], lr=1e-3)
    return actor, feat, q1, q2, q1t, q2t, opt_q, opt_a, la, opt_la


def one_step(ctx, batch: int, utd: int, mask: torch.Tensor) -> None:
    actor, feat, q1, q2, q1t, q2t, opt_q, opt_a, la, opt_la = ctx
    alpha = la.exp().item()
    for u in range(utd):
        o = torch.rand(batch, OBS, device="cuda")
        o2 = torch.rand(batch, OBS, device="cuda")
        a = torch.rand(batch, ACT, device="cuda") * 2 - 1
        r = torch.randn(batch, device="cuda")
        d = torch.zeros(batch, device="cuda")
        m = mask[:batch] if mask.shape[0] >= batch else mask[:1].expand(batch, -1)
        both = feat(torch.cat([o, o2], 0)) if isinstance(feat, torch.nn.Module) or callable(feat) else None
        if isinstance(both, dict):
            fb = {k: v[:batch] for k, v in both.items()}
            fb2 = {k: v[batch:] for k, v in both.items()}
        else:
            fb, fb2 = o, o2
        with torch.no_grad():
            na, nlp, _ = actor.get_action(fb2, m)
            qn = torch.min(q1t(o2, na), q2t(o2, na)) - alpha * nlp
            tgt = r + 0.99 * (1 - d) * qn.view(-1)
        qf = F.mse_loss(q1(o, a).view(-1), tgt) + F.mse_loss(q2(o, a).view(-1), tgt)
        opt_q.zero_grad(set_to_none=True)
        qf.backward()
        opt_q.step()
        if u % 2 == 0:
            for _ in range(2):
                pi, lp, _ = actor.get_action(fb, m)
                al = (alpha * lp - torch.min(q1(o, pi), q2(o, pi))).mean()
                opt_a.zero_grad(set_to_none=True)
                al.backward()
                opt_a.step()
                with torch.no_grad():
                    _, lpc, _ = actor.get_action(fb, m)
                lal = (-la.exp() * (lpc - m.sum(1, keepdim=True))).mean()
                opt_la.zero_grad(set_to_none=True)
                lal.backward()
                opt_la.step()
                alpha = la.exp().item()


def bench(label: str, structured: bool, compile_feat: bool, batch: int, utd: int, reps: int = 5):
    ctx = make_agent(structured, compile_feat)
    mask = torch.ones(batch, ACT, device="cuda")
    one_step(ctx, batch, utd, mask)          # warm + compile
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        one_step(ctx, batch, utd, mask)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / reps * 1000
    print(f"{label:34s} batch {batch:5d} utd {utd:2d} -> learner {ms:6.1f} ms/vector step", flush=True)
    del ctx
    torch.cuda.empty_cache()
    return ms


if __name__ == "__main__":
    a = bench("r06 flat, 256 x 15", False, False, 256, 15)
    b = bench("r10 structured, 256 x 15 (eager)", True, False, 256, 15)
    c = bench("P1+P2 structured, 2048 x 2", True, True, 2048, 2)
    print(f"\nenv alone (measured): 100 ms")
    for name, ms in (("r06", a), ("r10", b), ("P1+P2", c)):
        print(f"  {name:6s} step ~{100 + ms:6.1f} ms  ->  {960 / (100 + ms) * 1000:6.0f} steps/s")

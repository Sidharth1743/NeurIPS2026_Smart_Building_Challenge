"""Per-zone feature layout for the structured SAC actor (torch + numpy reference).

Everything here is a deterministic function of the gateway observation. It reads
sensors, setpoints, the previous command echo and presence flags ONLY. It never
reads the base-controller proposal columns (zone c7/c8, AHU c9..12, plant
c7..c9) -- in this build those are identically 0 anyway -- and it does not run
any controller: the returned tensors are inputs, the actions come from learned
weights.

Layout follows the r139 shared policy exactly so the two policies see the same
thing:
  pool        80 values   (zone/AHU/plant statistics, presence, demand aggregates)
  zone local  68 per zone slot  [own sensors 0..5 | its AHU 6..13 | cold deficit
                                 14,15 | 1 | memory 11 (zeros) | siblings 40]
  AHU local   32 per AHU        [own sensors 0..7 | its zones' stats 8..25 | zone
                                 presence 26 | zone excess 27,28 | 1 | plant 30,31]
  plant local 16                [plant sensors 0..5 | demand 6..11 | occupancy 12 | 13,14 | 1]
Local blocks are asinh-compressed, the pooled f<51 and f>=72 entries likewise, as
in sbs_role_input_k / sbs_pool_k.
"""
from __future__ import annotations

import numpy as np
import torch

CN_FZ, CN_ZS, CN_ZPA, CN_K, CN_FA = 9, 90, 10, 9, 16
CN_OA, CN_OH, CN_OG, CN_OBS = 810, 954, 964, 983
CN_AA, CN_AH = 180, 216
ZONE_L, AHU_L, PLANT_L = 68, 32, 16
POOL = 80


def _zone_idx(feature: int) -> np.ndarray:
    return CN_FZ * np.arange(CN_ZS) + feature


def _ahu_idx(feature: int) -> np.ndarray:
    return CN_OA + CN_FA * np.arange(CN_K) + feature


# --------------------------------------------------------------- numpy ref --
def np_stats(x: np.ndarray, present: np.ndarray) -> np.ndarray:
    """x, present: [..., n] -> [..., 3] = (mean, min, max) over present entries (0 if none)."""
    p = present > 0
    n = p.sum(-1)
    xm = np.where(p, x, 0.0)
    s = xm.sum(-1)
    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    lo = np.where(p, x, np.inf).min(-1)
    hi = np.where(p, x, -np.inf).max(-1)
    lo = np.where(n > 0, lo, 0.0)
    hi = np.where(n > 0, hi, 0.0)
    return np.stack([mean, lo, hi], -1)


def np_demand(o: np.ndarray, feature: int, kind: int) -> np.ndarray:
    """sbs_demand: x = max(0, -o[c]); 0 mean, 1 max, 2 count, 3 mean of x^2."""
    x = np.maximum(0.0, -o[:, _zone_idx(feature)])
    p = o[:, _zone_idx(0)] > 0
    n = p.sum(-1)
    if kind == 1:
        return np.where(n > 0, np.where(p, x, 0.0).max(-1), 0.0)
    xm = np.where(p, x, 0.0)
    if kind == 2:
        return (xm > 0).sum(-1).astype(np.float64)
    if kind == 3:
        return np.where(n > 0, (xm ** 2).sum(-1) / np.maximum(n, 1), 0.0)
    return np.where(n > 0, xm.sum(-1) / np.maximum(n, 1), 0.0)


def np_pool(o: np.ndarray) -> np.ndarray:
    B = o.shape[0]
    out = np.zeros((B, POOL))
    zfeat = np.stack([o[:, _zone_idx(f)] for f in range(1, 7)], -1)   # [B,90,6] = c1..c6
    zp = o[:, _zone_idx(0)]
    zs = np_stats(zfeat, np.broadcast_to(zp[:, :, None], zfeat.shape))    # [B,90,6,3]? no
    # np_stats over the zone axis: transpose so the zone axis is last
    zs = np_stats(np.transpose(zfeat, (0, 2, 1)), zp[:, None, :].repeat(6, 1))  # [B,6,3]
    out[:, 0:18] = zs.reshape(B, 18)
    afeat = np.stack([o[:, _ahu_idx(1 + q)] for q in range(8)] +
                     [np.stack([o[:, CN_FZ * (CN_ZPA * np.arange(CN_K) + j) + 1]
                                for j in range(CN_ZPA)], -1).mean(-1)], -1)   # [B,9,9]
    ap = o[:, _ahu_idx(0)]
    as_ = np_stats(np.transpose(afeat, (0, 2, 1)), ap[:, None, :].repeat(9, 1))  # [B,9,3]
    out[:, 18:45] = as_.reshape(B, 27)
    out[:, 45:51] = o[:, CN_OH + 1:CN_OH + 7]
    out[:, 51:70] = o[:, CN_OG:CN_OG + 19]
    out[:, 70] = (o[:, _zone_idx(0)] > 0).sum(-1) / CN_ZS
    out[:, 71] = (o[:, _ahu_idx(0)] > 0).sum(-1) / CN_K
    for i, (f, k) in enumerate([(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2), (2, 3), (3, 3)]):
        out[:, 72 + i] = np_demand(o, f, k)
    comp = np.zeros(POOL, bool)
    comp[:51] = True
    comp[72:] = True
    return np.where(comp, np.arcsinh(out), out)


def np_zone_local(o: np.ndarray) -> np.ndarray:
    B = o.shape[0]
    L = np.zeros((B, CN_ZS, ZONE_L))
    for f in range(6):
        L[:, :, f] = o[:, _zone_idx(1 + f)]
    for f in range(6, 14):
        L[:, :, f] = o[:, CN_OA + CN_FA * (np.arange(CN_ZS) // CN_ZPA) + 1 + (f - 6)]
    L[:, :, 14] = np.maximum(0.0, -o[:, _zone_idx(2)])
    L[:, :, 15] = np.maximum(0.0, -o[:, _zone_idx(3)])
    L[:, :, 16] = 1.0
    # 17..27 memory features: env.zone_memory = 0 -> zeros
    for s in range(CN_ZS):
        base = CN_ZPA * (s // CN_ZPA)
        for j in range(CN_ZPA):
            f0 = 28 + j * 4
            if j == s % CN_ZPA:
                continue
            sib = base + j
            pres = o[:, CN_FZ * sib] > 0
            L[:, s, f0 + 0] = np.where(pres, o[:, CN_FZ * sib + 1], 0.0)
            L[:, s, f0 + 1] = np.where(pres, o[:, CN_FZ * sib + 5], 0.0)
            L[:, s, f0 + 2] = np.where(pres, o[:, CN_FZ * sib + 6], 0.0)
            L[:, s, f0 + 3] = np.where(pres, 1.0, 0.0)
    return np.arcsinh(L)


def np_ahu_local(o: np.ndarray) -> np.ndarray:
    B = o.shape[0]
    L = np.zeros((B, CN_K, AHU_L))
    for f in range(8):
        L[:, :, f] = o[:, _ahu_idx(1 + f)]
    zones = (CN_ZPA * np.arange(CN_K)[:, None] + np.arange(CN_ZPA)[None, :]).ravel()  # [90]
    for q in range(6):
        for st in range(3):
            f = 8 + q * 3 + st
            x = o[:, CN_FZ * zones + 1 + q].reshape(B, CN_K, CN_ZPA)
            p = (o[:, CN_FZ * zones].reshape(B, CN_K, CN_ZPA) > 0)
            L[:, :, f] = np_stats(x, p)[..., st]
    zp = (o[:, CN_FZ * zones].reshape(B, CN_K, CN_ZPA) > 0)
    L[:, :, 26] = zp.sum(-1) / CN_ZPA
    L[:, :, 27] = np.where(zp, np.maximum(0.0, -o[:, CN_FZ * zones + 2].reshape(B, CN_K, CN_ZPA)), 0.0).max(-1)
    L[:, :, 28] = np.where(zp, np.maximum(0.0, -o[:, CN_FZ * zones + 3].reshape(B, CN_K, CN_ZPA)), 0.0).max(-1)
    L[:, :, 29] = 1.0
    L[:, :, 30] = o[:, CN_OH + 1][:, None]
    L[:, :, 31] = o[:, CN_OH + 2][:, None]
    return np.arcsinh(L)


def np_plant_local(o: np.ndarray) -> np.ndarray:
    B = o.shape[0]
    L = np.zeros((B, 1, PLANT_L))
    for f in range(6):
        L[:, 0, f] = o[:, CN_OH + 1 + f]
    for i, (f, k) in enumerate([(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2)]):
        L[:, 0, 6 + i] = np_demand(o, f, k)
    L[:, 0, 12] = o[:, CN_OG + 18]
    L[:, 0, 13] = o[:, CN_OG + 16]
    L[:, 0, 14] = o[:, CN_OG + 17]
    L[:, 0, 15] = 1.0
    return np.arcsinh(L)


# ------------------------------------------------------------------- torch --
# Kernel-launch bound: every per-column gather is a launch, so the blocks are
# built with ONE index_select each and reshaped, not 40 small assignments.
_IDX: dict[tuple[str, str], torch.Tensor] = {}


def _t(dev, key: str, arr) -> torch.Tensor:
    k = (str(dev), key)
    t = _IDX.get(k)
    if t is None:
        t = torch.as_tensor(np.ascontiguousarray(arr), device=dev)
        _IDX[k] = t
    return t


def _gather(o: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return o.index_select(1, idx)


def t_stats(x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    p = present > 0
    pf = p.to(x.dtype)
    n = pf.sum(-1, keepdim=True)
    s = (x * pf).sum(-1, keepdim=True)
    mean = torch.where(n > 0, s / n.clamp_min(1.0), torch.zeros_like(s))
    lo = torch.where(p, x, torch.full_like(x, float("inf"))).min(-1, keepdim=True).values
    hi = torch.where(p, x, torch.full_like(x, float("-inf"))).max(-1, keepdim=True).values
    lo = torch.where(n > 0, lo, torch.zeros_like(lo))
    hi = torch.where(n > 0, hi, torch.zeros_like(hi))
    return torch.cat([mean, lo, hi], -1)


def t_demand(o: torch.Tensor, feature: int, kind: int) -> torch.Tensor:
    Z = _t(o.device, f"Zd{feature}", CN_FZ * np.arange(CN_ZS) + feature)
    zp = _t(o.device, "Z0", CN_FZ * np.arange(CN_ZS))
    x = torch.clamp(-_gather(o, Z), min=0.0)
    p = _gather(o, zp) > 0
    pf = p.to(x.dtype)
    n = pf.sum(-1)
    xm = x * pf
    if kind == 1:
        v = torch.where(p, x, torch.zeros_like(x)).max(-1).values
        return torch.where(n > 0, v, torch.zeros_like(v))
    if kind == 2:
        return (xm > 0).sum(-1).to(x.dtype)
    num = (xm ** 2).sum(-1) if kind == 3 else xm.sum(-1)
    return torch.where(n > 0, num / n.clamp_min(1.0), torch.zeros_like(num))


def t_pool(o: torch.Tensor) -> torch.Tensor:
    B, dev = o.shape[0], o.device
    zc = _t(dev, "zfeat", np.stack([CN_FZ * np.arange(CN_ZS) + f for f in range(1, 7)], 1).ravel())
    zfeat = _gather(o, zc).reshape(B, CN_ZS, 6).transpose(1, 2)          # [B,6,90]
    zp = _gather(o, _t(dev, "Z0", CN_FZ * np.arange(CN_ZS)))
    zs = t_stats(zfeat, zp[:, None, :].expand(-1, 6, -1))                # [B,6,3]
    ac = _t(dev, "afeat", np.stack([CN_OA + CN_FA * np.arange(CN_K) + f for f in range(1, 9)], 1).ravel())
    afeat = _gather(o, ac).reshape(B, CN_K, 8).transpose(1, 2)           # [B,8,9]
    mzt = torch.stack([_gather(o, _t(dev, f"zmean{j}", CN_FZ * (CN_ZPA * np.arange(CN_K) + j) + 1))
                       for j in range(CN_ZPA)], -1).mean(-1)             # [B,9]
    afeat = torch.cat([afeat, mzt[:, None, :]], 1)                       # [B,9,9]
    ap = _gather(o, _t(dev, "A0", CN_OA + CN_FA * np.arange(CN_K)))
    as_ = t_stats(afeat, ap[:, None, :].expand(-1, 9, -1))               # [B,9,3]
    f70 = (zp > 0).sum(-1, keepdim=True).to(o.dtype) / CN_ZS
    f71 = (ap > 0).sum(-1, keepdim=True).to(o.dtype) / CN_K
    dem = torch.stack([t_demand(o, f, k) for f, k in
                       [(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2), (2, 3), (3, 3)]], -1)
    out = torch.cat([zs.reshape(B, 18), as_.reshape(B, 27), o[:, CN_OH + 1:CN_OH + 7],
                     o[:, CN_OG:CN_OG + 19], f70, f71, dem], 1)
    comp = _t(dev, "pcomp", np.concatenate([np.ones(51, bool), np.zeros(21, bool), np.ones(8, bool)]))
    return torch.where(comp, torch.asinh(out), out)


def t_zone_local(o: torch.Tensor, zm: torch.Tensor | None = None) -> torch.Tensor:
    B, dev = o.shape[0], o.device
    Z = CN_FZ * np.arange(CN_ZS)
    own = np.stack([Z + f for f in range(1, 7)], 1).ravel()              # [90*6]
    L = torch.zeros((B, CN_ZS, ZONE_L), device=dev, dtype=o.dtype)
    L[:, :, 0:6] = _gather(o, _t(dev, "zown", own)).reshape(B, CN_ZS, 6)
    aoff = CN_OA + CN_FA * (np.arange(CN_ZS) // CN_ZPA)
    at = np.stack([aoff + 1 + f for f in range(8)], 1).ravel()
    L[:, :, 6:14] = _gather(o, _t(dev, "zahu", at)).reshape(B, CN_ZS, 8)
    L[:, :, 14] = torch.clamp(-_gather(o, _t(dev, "z2", Z + 2)), min=0.0)
    L[:, :, 15] = torch.clamp(-_gather(o, _t(dev, "z3", Z + 3)), min=0.0)
    L[:, :, 16] = 1.0
    if zm is not None:                      # env.zone_memory = 1: 11 per-zone features
        L[:, :, 17:28] = zm
    # sibling block: one gather of [90*10] slots x 4 fields
    j = np.arange(CN_ZPA)
    sib = CN_ZPA * (np.arange(CN_ZS)[:, None] // CN_ZPA) + j[None, :]      # [90,10]
    same = (np.arange(CN_ZS)[:, None] % CN_ZPA) == j[None, :]
    cols = np.stack([CN_FZ * sib + 1, CN_FZ * sib + 5, CN_FZ * sib + 6], -1)  # [90,10,3]
    sidx = cols.reshape(-1)
    vals = _gather(o, _t(dev, "zsib", sidx)).reshape(B, CN_ZS, CN_ZPA, 3)
    pres = (_gather(o, _t(dev, "zsib0", (CN_FZ * sib).reshape(-1)))
            .reshape(B, CN_ZS, CN_ZPA) > 0).to(o.dtype)
    keep = (pres * _t(dev, "zkeep", (~same).astype(np.float32)))          # [B,90,10]
    blk = torch.cat([vals, pres[:, :, :, None]], -1) * keep[..., None]   # [B,90,10,4]
    L[:, :, 28:68] = blk.reshape(B, CN_ZS, CN_ZPA * 4)
    return torch.asinh(L)


def t_ahu_local(o: torch.Tensor) -> torch.Tensor:
    B, dev = o.shape[0], o.device
    L = torch.zeros((B, CN_K, AHU_L), device=dev, dtype=o.dtype)
    ac = np.stack([CN_OA + CN_FA * np.arange(CN_K) + f for f in range(1, 9)], 1).ravel()
    L[:, :, 0:8] = _gather(o, _t(dev, "aown", ac)).reshape(B, CN_K, 8)
    zones = (CN_ZPA * np.arange(CN_K)[:, None] + np.arange(CN_ZPA)[None, :]).ravel()
    zi = _t(dev, "ahuzone", CN_FZ * zones)
    p = (_gather(o, zi) > 0).reshape(B, CN_K, CN_ZPA)
    for q in range(6):
        x = _gather(o, _t(dev, f"ahuz{q}", CN_FZ * zones + 1 + q)).reshape(B, CN_K, CN_ZPA)
        st = t_stats(x, p)
        L[:, :, 8 + q * 3:11 + q * 3] = st
    L[:, :, 26] = p.to(o.dtype).sum(-1) / CN_ZPA
    c2 = torch.clamp(-_gather(o, _t(dev, "ahum2", CN_FZ * zones + 2)), min=0.0).reshape(B, CN_K, CN_ZPA)
    c3 = torch.clamp(-_gather(o, _t(dev, "ahum3", CN_FZ * zones + 3)), min=0.0).reshape(B, CN_K, CN_ZPA)
    L[:, :, 27] = torch.where(p, c2, torch.zeros((), device=dev, dtype=o.dtype)).max(-1).values
    L[:, :, 28] = torch.where(p, c3, torch.zeros((), device=dev, dtype=o.dtype)).max(-1).values
    L[:, :, 29] = 1.0
    L[:, :, 30] = o[:, CN_OH + 1][:, None]
    L[:, :, 31] = o[:, CN_OH + 2][:, None]
    return torch.asinh(L)


def t_plant_local(o: torch.Tensor) -> torch.Tensor:
    B, dev = o.shape[0], o.device
    L = torch.zeros((B, 1, PLANT_L), device=dev, dtype=o.dtype)
    L[:, 0, 0:6] = o[:, CN_OH + 1:CN_OH + 7]
    for i, (f, k) in enumerate([(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2)]):
        L[:, 0, 6 + i] = t_demand(o, f, k)
    L[:, 0, 12] = o[:, CN_OG + 18]
    L[:, 0, 13] = o[:, CN_OG + 16]
    L[:, 0, 14] = o[:, CN_OG + 17]
    L[:, 0, 15] = 1.0
    return torch.asinh(L)


def t_bundle(o: torch.Tensor, zm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {"pool": t_pool(o), "loc0": t_zone_local(o, zm),
            "loc1": t_ahu_local(o), "loc2": t_plant_local(o)}

"""Kaggle policy: the structured SAC actor trained on the 30-scenario batch.

Deterministic decode: every head is tanh(mu) of a learned affine readout, masked by the
contract. There is no controller, no sequencer, no guard -- the same forward pass the
training eval (`SAC/tools/score_set.py`, tanh(mean)) ran.

Inputs are built exactly as the trainer builds them, from the gateway observation only:

  pool        80 values   (zone/AHU/plant statistics, presence, demand aggregates)
  zone local  68 per zone slot  [own sensors 0..5 | its AHU 6..13 | cold deficit 14,15 |
                                 1 | memory 11 (env.zone_memory=0 -> zeros) | siblings 40]
  AHU local   32 per AHU
  plant local 16

The canonical 983-value observation and the 219-head -> gateway action mapping are the
proven r139 submission code; the feature blocks and the actor are this repository's SAC
implementation. Weights are a flat float32 dump of the 12 actor tensors.

Assembly: SAC/kaggle/build_submission.py. Do not hand-edit.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

try:
    from kaggle_evaluation.smart_buildings_inference_server import Policy as _Base
except Exception:  # noqa: BLE001
    class _Base:  # type: ignore[no-redef]
        def __init__(self, building_meta_data, action_spec):
            self.building_meta_data = building_meta_data
            self.action_spec = action_spec

f32 = np.float32

# Canonical layout (sb_gpu_base.inc / sb_shared_policy.cuh).
CN_K, CN_ZPA = 9, 10
CN_ZS = CN_K * CN_ZPA
CN_FZ, CN_FA, CN_FH, CN_FG = 9, 16, 10, 19
CN_OA = CN_ZS * CN_FZ
CN_OH = CN_OA + CN_K * CN_FA
CN_OG = CN_OH + CN_FH
CN_OBS = CN_OG + CN_FG
CN_AA = 2 * CN_ZS
CN_AH = CN_AA + 4 * CN_K
CN_ACT = CN_AH + 3

SBS_POOL, SBS_FEAT, SBS_SIB = 80, 11, 4
CN_ZM = 17 + SBS_FEAT + CN_ZPA * SBS_SIB      # zone local block, 68
CN_LA, CN_LH = 32, 16
ZONE_MLP_HIDDEN = 32

# Network shape (policy.hidden_size 256, num_layers 2, policy.beta, zone_mlp).
H = 256
NUM_LAYERS = 2
ACT_N = 288                    # decoder output_dim = 256 continuous heads + 16 * 2
VALUE_ROWS, VALUE_DIN = 1 + CN_ZS, H        # r139: policy.value_delay=0 (no 8 delay-summary columns)
ROLE_ROWS = (2 * 2, 4 * 2, 3 * 2)
ROLE_LOCAL = (CN_ZM, CN_LA, CN_LH)
BETA_FLOOR = 0.5               # r139: policy.beta_floor=0.5
BETA_CONC_MAX = 12.0
BETA_EPS = f32(1e-6)
N_PARAMS = (H * SBS_POOL + VALUE_ROWS * VALUE_DIN + ACT_N
            + sum(r * (H + l) for r, l in zip(ROLE_ROWS, ROLE_LOCAL))
            + ZONE_MLP_HIDDEN * (H + CN_ZM) + ROLE_ROWS[0] * ZONE_MLP_HIDDEN
            + NUM_LAYERS * 3 * H * H)
CHECKPOINT_BYTES = N_PARAMS * 4
TWO_PI = f32(6.28318530718)


class DoesNotFit(Exception):
    """The building is larger than the canonical 9 by 10 slots."""


def _sigmoid(x):
    z = np.exp(-np.abs(x)).astype(f32)
    return np.where(x >= 0, f32(1) / (f32(1) + z), z / (f32(1) + z)).astype(f32)


def _lerp(a, b, w):
    diff = (b - a).astype(f32)
    return np.where(np.abs(w) < f32(0.5), a + w * diff, b - diff * (f32(1) - w)).astype(f32)


def _zstat(o, first, count, feature, stat):
    value = f32(0)
    n = 0
    for s in range(first, first + count):
        if o[CN_FZ * s] == 0:
            continue
        x = f32(o[CN_FZ * s + feature])
        if n == 0:
            value = x
        elif stat == 1:
            value = x if x < value else value
        elif stat == 2:
            value = x if x > value else value
        else:
            value = f32(value + x)
        n += 1
    if stat == 0 and n:
        return f32(value / f32(n))
    return value


def _demand(o, feature, kind):
    value = f32(0)
    n = 0
    for s in range(CN_ZS):
        if o[CN_FZ * s] == 0:
            continue
        x = f32(max(0.0, -float(o[CN_FZ * s + feature])))
        if kind == 1:
            value = x if x > value else value
        elif kind == 2:
            value = f32(value + f32(x > 0))
        elif kind == 3:
            value = f32(value + x * x)
        else:
            value = f32(value + x)
        n += 1
    if kind != 1 and n:
        return f32(value / f32(n))
    return value


def _pool_feature(o, f):
    if f < 18:
        return _zstat(o, 0, CN_ZS, 1 + f // 3, f % 3)
    if f < 45:
        q, stat, value, n = (f - 18) // 3, (f - 18) % 3, f32(0), 0
        for k in range(CN_K):
            if o[CN_OA + CN_FA * k] == 0:
                continue
            x = f32(o[CN_OA + CN_FA * k + 1 + q]) if q < 8 else _zstat(o, CN_ZPA * k, CN_ZPA, 1, 0)
            if n == 0:
                value = x
            elif stat == 1:
                value = x if x < value else value
            elif stat == 2:
                value = x if x > value else value
            else:
                value = f32(value + x)
            n += 1
        if stat == 0 and n:
            return f32(value / f32(n))
        return value
    if f < 51:
        return f32(o[CN_OH + 1 + f - 45])
    if f < 70:
        return f32(o[CN_OG + f - 51])
    if f == 70:
        return f32(sum(o[CN_FZ * s] != 0 for s in range(CN_ZS)) / CN_ZS)
    if f == 71:
        return f32(sum(o[CN_OA + CN_FA * k] != 0 for k in range(CN_K)) / CN_K)
    if f == 72:
        return _demand(o, 2, 0)
    if f == 73:
        return _demand(o, 3, 0)
    if f == 74:
        return _demand(o, 2, 1)
    if f == 75:
        return _demand(o, 3, 1)
    if f == 76:
        return _demand(o, 2, 2)
    if f == 77:
        return _demand(o, 3, 2)
    if f == 78:
        return _demand(o, 2, 3)
    return _demand(o, 3, 3)


def _local_feature(o, role, node, f, zone_feat):
    if role == 0:
        if f < 6:
            return f32(o[CN_FZ * node + 1 + f])
        if f < 14:
            return f32(o[CN_OA + CN_FA * (node // CN_ZPA) + 1 + f - 6])
        if f < 16:
            return f32(max(0.0, -float(o[CN_FZ * node + 2 + f - 14])))
        if f == 16:
            return f32(1)
        if f < 17 + SBS_FEAT:
            return f32(zone_feat[f - 17]) if zone_feat is not None else f32(0)
        rel = f - (17 + SBS_FEAT)
        j, field = divmod(rel, SBS_SIB)
        if j == node % CN_ZPA:
            return f32(0)
        s = CN_ZPA * (node // CN_ZPA) + j
        if o[CN_FZ * s] == 0:
            return f32(0)
        if field == 3:
            return f32(1)
        return f32(o[CN_FZ * s + (1, 5, 6)[field]])
    if role == 1:
        if f < 8:
            return f32(o[CN_OA + CN_FA * node + 1 + f])
        if f < 26:
            return _zstat(o, CN_ZPA * node, CN_ZPA, 1 + (f - 8) // 3, (f - 8) % 3)
        if f == 26:
            present = sum(o[CN_FZ * (CN_ZPA * node + j)] != 0 for j in range(CN_ZPA))
            return f32(present / CN_ZPA)
        if f < 29:
            v = f32(0)
            for j in range(CN_ZPA):
                s = CN_ZPA * node + j
                if o[CN_FZ * s] != 0:
                    x = f32(max(0.0, -float(o[CN_FZ * s + 2 + f - 27])))
                    v = x if x > v else v
            return v
        if f == 29:
            return f32(1)
        return f32(o[CN_OH + 1 + f - 30])
    if f < 6:
        return f32(o[CN_OH + 1 + f])
    if f < 8:
        return _demand(o, 2 + f - 6, 0)
    if f < 10:
        return _demand(o, 2 + f - 8, 1)
    if f < 12:
        return _demand(o, 2 + f - 10, 2)
    if f == 12:
        return f32(o[CN_OG + 18])
    if f < 15:
        return f32(o[CN_OG + 16 + f - 13])
    return f32(1)


def _beta_action(u, v):
    """Deterministic Beta decode: mean of the capped concentration, mapped to [-1, 1]."""
    au = np.asarray(u, f32)
    av = np.asarray(v, f32)
    al = (np.maximum(au, f32(0)) + np.log1p(np.exp(-np.abs(au))) + f32(BETA_FLOOR)).astype(f32)
    be = (np.maximum(av, f32(0)) + np.log1p(np.exp(-np.abs(av))) + f32(BETA_FLOOR)).astype(f32)
    k = (f32(BETA_CONC_MAX) / np.maximum(al + be, f32(BETA_CONC_MAX))).astype(f32)
    al = (al * k).astype(f32)
    be = (be * k).astype(f32)
    mean = np.clip((al / (al + be)).astype(f32), BETA_EPS, f32(1) - BETA_EPS).astype(f32)
    return (f32(2) * mean - f32(1)).astype(f32)


# Device ids and measurement names from the starter kit. An observation id is
# `generate_field_id(device, measurement)`: `device_measurement`, or
# `device_measurement_1` when that string already belongs to another pair.
AHS_DEVICE = "ahs"
HWS_DEVICE = "hws"
ZONE_TEMPERATURE = "zone_air_temperature_sensor"
ZONE_HEATING = "zone_air_heating_temperature_setpoint_sensor"
ZONE_COOLING = "zone_air_cooling_temperature_setpoint_sensor"
DAMPER = "supply_air_damper_percentage_command"
REHEAT = "reheat_valve_percentage_command"
AHU_FIELDS = (
    "outside_air_temperature_sensor",
    "supply_air_static_pressure_sensor",
    "supply_air_flowrate_sensor",
    "outside_air_flowrate_sensor",
    "supply_air_temperature_setpoint",
    "recirculation_percentage_command",
    "cooling_request_count",
    "supervisor_run_command",
)
HWS_FIELDS = (
    "supply_water_temperature_sensor",
    "supply_water_temperature_setpoint",
    "differential_pressure_sensor",
    "differential_pressure_setpoint",
    "run_status",
    "heating_request_count",
)


def _assign_field_ids(pairs):
    """Same id rule as `environment.generate_field_id`."""
    by_device = {}
    for device, measurement in pairs:
        by_device.setdefault(device, set()).add(measurement)
    taken = set()
    out = {}
    for device in sorted(by_device):
        for measurement in sorted(by_device[device]):
            new_id = f"{device}_{measurement}"
            counter = 0
            while new_id in taken:
                counter += 1
                new_id = f"{device}_{measurement}_{counter}"
            taken.add(new_id)
            out[(device, measurement)] = new_id
    return out


def _input_tree(root="/kaggle/input", limit=2000):
    base = Path(root)
    if not base.is_dir():
        return f"{root} is not a directory"
    lines = []
    used = 0
    for path in sorted(base.rglob("*")):
        line = str(path)
        lines.append(line)
        used += len(line) + 1
        if used > limit:
            lines.append("...")
            break
    return "\n".join(lines) if lines else f"{root} is empty"


def _find_weights():
    here = Path(__file__).resolve().parent
    ordered = []
    env_dir = os.environ.get("SB_POLICY_MODEL_DIR")
    if env_dir:
        ordered.append(Path(env_dir) / "weights.bin")
    ordered.append(here / "model" / "weights.bin")
    ordered.append(here / "weights.bin")
    hits = []
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        hits = sorted(kaggle_root.rglob("weights.bin"))
    seen = set()
    for path in ordered + hits:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    tree = _input_tree()
    print(tree, flush=True)
    raise FileNotFoundError("weights.bin not found\n" + tree)



# ---------------------------------------------------------------- SAC actor --
# SAC/train/model.py StructuredActor: Linear(80->256) trunk, three per-role readouts and
# the small nonlinear zone branch. 12 tensors, 36,518 floats (the 438 mask buffers are
# not weights).
POOL, ZONE_L, AHU_L, PLANT_L = 80, 68, 32, 16
SAC_H = 256
ZM_H = 32
N_ACTOR = (POOL * SAC_H + SAC_H                                   # trunk
           + (SAC_H + ZONE_L) * 4 + 4                             # r0
           + (SAC_H + AHU_L) * 8 + 8                              # r1
           + (SAC_H + PLANT_L) * 6 + 6                            # r2
           + (SAC_H + ZONE_L) * ZM_H + ZM_H                       # zm1
           + ZM_H * 4 + 4)                                        # zm2
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

# ---------------------------------------------- numpy feature blocks (SAC) --
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


def _beta_ab(p, q, floor, cap):
    """softplus rows -> (a, b); total concentration capped at `cap`, mean preserved."""
    a = np.logaddexp(0.0, p).astype(f32) + f32(floor)
    b = np.logaddexp(0.0, q).astype(f32) + f32(floor)
    if cap and cap > 0:
        total = a + b
        k = np.where(total > cap, f32(cap) / total, f32(1.0)).astype(f32)
        a = (a * k).astype(f32)
        b = (b * k).astype(f32)
    return a, b


def _actor_forward(w, o, head="gauss", floor=0.5, cap=12.0):
    """983-wide canonical observation -> 219-head executed action.

    The readouts emit two rows per head: (mean, log_std) for the Gaussian and (u, v) for
    the Beta. Gaussian: tanh(mean). Beta, the competition decode: a = softplus(u) + floor,
    b = softplus(v) + floor, a + b capped at `cap` with the mean preserved, action =
    2 a / (a + b) - 1.
    """
    pool = np_pool(o[None, :])[0]
    loc0 = np_zone_local(o[None, :])[0]
    loc1 = np_ahu_local(o[None, :])[0]
    loc2 = np_plant_local(o[None, :])[0][0]
    h = (pool @ w["trunk_w"].T + w["trunk_b"]).astype(f32)
    x0 = np.concatenate([np.broadcast_to(h, (CN_ZS, SAC_H)), loc0], -1)
    o0 = (x0 @ w["r0_w"].T + w["r0_b"]).astype(f32)
    zn = np.tanh((x0 @ w["zm1_w"].T + w["zm1_b"]).astype(f32))
    o0 = (o0 + (zn @ w["zm2_w"].T + w["zm2_b"]).astype(f32)).astype(f32)
    x1 = np.concatenate([np.broadcast_to(h, (CN_K, SAC_H)), loc1], -1)
    o1 = (x1 @ w["r1_w"].T + w["r1_b"]).astype(f32)
    x2 = np.concatenate([h, loc2])[None, :]
    o2 = (x2 @ w["r2_w"].T + w["r2_b"]).astype(f32)
    p1 = np.zeros(CN_ACT, f32)
    p2 = np.zeros(CN_ACT, f32)
    p1[0:2 * CN_ZS] = o0[:, 0:2].reshape(-1)
    p2[0:2 * CN_ZS] = o0[:, 2:4].reshape(-1)
    p1[CN_AA:CN_AA + 4 * CN_K] = o1[:, 0:4].reshape(-1)
    p2[CN_AA:CN_AA + 4 * CN_K] = o1[:, 4:8].reshape(-1)
    p1[CN_AH:CN_AH + 3] = o2[0, 0:3]
    p2[CN_AH:CN_AH + 3] = o2[0, 3:6]
    if head == "beta":
        a, b = _beta_ab(p1, p2, floor, cap)
        return (f32(2.0) * a / (a + b) - f32(1.0)).astype(f32)
    return np.tanh(p1).astype(f32)



class StructuredSacPolicy(_Base):

    """One policy instance per scenario; the gateway rebuilds it on every first turn."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.fallback = False
        path = _find_weights()
        self.w = _parse_actor(path)
        meta = _sidecar(path)
        self.head = meta.get("head", "gauss")
        self.beta_floor = float(meta.get("beta_floor", 0.5))
        self.beta_conc_max = float(meta.get("beta_conc_max", 12.0))
        print(f"head {self.head} floor {self.beta_floor} cap {self.beta_conc_max}", flush=True)
        try:
            self._layout(building_meta_data)
        except DoesNotFit as exc:
            print(f"canonical slots do not fit this building: {exc}", flush=True)
            self.fallback = True
    def _layout(self, meta):
        names = list(meta["observation_field_names"])
        idx = {name: i for i, name in enumerate(names)}
        present = set(idx)
        self.n_obs = len(names)
        norm = meta.get("observation_normalization_metadata") or {}
        self.fmean = np.zeros(self.n_obs, np.float32)
        self.fstd = np.ones(self.n_obs, np.float32)

        def scale(index, measurement):
            if index < 0:
                return
            spec = norm.get(measurement) or {}
            var = float(spec.get("sample_variance") or 0.0)
            if var <= 0.0:
                return
            std = spec.get("sample_std")
            self.fmean[index] = f32(spec["sample_mean"])
            self.fstd[index] = f32(float(std) if std else math.sqrt(var))

        assignments = meta["ahu_zone_assignments"]
        self.K = len(meta["temp_indices"])
        if self.K > CN_K or self.K > len(assignments):
            raise DoesNotFit("air handlers do not fit the canonical slots")
        if self.K > len(meta["pressure_indices"]) or self.K > len(meta["recirculation_indices"]):
            raise DoesNotFit("air-handler action indices are short")
        ahu_ids = list(assignments)[: self.K]
        vav_ahu = {}
        for k, ahu_id in enumerate(ahu_ids):
            for room in assignments[ahu_id]:
                vav_ahu[room.replace("room_", "VAV_")] = k

        pairs = []
        for vav_id in vav_ahu:
            for measurement in (ZONE_TEMPERATURE, ZONE_HEATING, ZONE_COOLING, DAMPER, REHEAT):
                pairs.append((vav_id, measurement))
        for k in range(self.K):
            for suffix in AHU_FIELDS:
                pairs.append((AHS_DEVICE, f"ahu_{k + 1}_{suffix}"))
        for measurement in HWS_FIELDS:
            pairs.append((HWS_DEVICE, measurement))
        field_ids = _assign_field_ids(pairs)

        def published(device, measurement):
            field_id = field_ids[(device, measurement)]
            if field_id in present:
                return field_id
            plain = f"{device}_{measurement}"
            if plain in present:
                return plain
            counter = 1
            while counter < 8:
                alt = f"{plain}_{counter}"
                if alt in present:
                    return alt
                counter += 1
            return None

        def locate(device, measurement):
            field_id = published(device, measurement)
            index = idx[field_id] if field_id is not None else -1
            scale(index, measurement)
            return index

        self.nC = int(meta["n_continuous_action"])
        self.nD = int(meta["n_discrete_action"])
        self.cont_slot = np.full(self.nC, -1, np.int64)
        self.disc_slot = np.full(self.nD, -1, np.int64)
        self.zslot = []
        count = [0] * CN_K
        sensor_of = {}
        for vav_id in vav_ahu:
            field_id = published(vav_id, ZONE_TEMPERATURE)
            if field_id is not None:
                sensor_of[field_id] = vav_id
        zone_actions = meta.get("zone_action_indices") or {}
        for name in names:
            vav_id = sensor_of.get(name)
            if vav_id is None:
                continue
            k = vav_ahu[vav_id]
            if count[k] >= CN_ZPA:
                raise DoesNotFit("more than 10 zones on one air handler")
            s = CN_ZPA * k + count[k]
            count[k] += 1
            iT = idx[name]
            scale(iT, ZONE_TEMPERATURE)
            actions = zone_actions.get(vav_id, {})
            self.zslot.append((
                s, iT,
                locate(vav_id, ZONE_HEATING),
                locate(vav_id, ZONE_COOLING),
                locate(vav_id, DAMPER),
                locate(vav_id, REHEAT),
            ))
            damper_at = actions.get(DAMPER)
            reheat_at = actions.get(REHEAT)
            if damper_at is not None:
                self.cont_slot[damper_at] = 2 * s
            if reheat_at is not None:
                self.cont_slot[reheat_at] = 2 * s + 1
        self.zc = count
        self.aslot = []
        for k in range(self.K):
            device_field = f"ahu_{k + 1}"
            self.aslot.append({
                "oat": locate(AHS_DEVICE, f"{device_field}_outside_air_temperature_sensor"),
                "sp": locate(AHS_DEVICE, f"{device_field}_supply_air_static_pressure_sensor"),
                "sf": locate(AHS_DEVICE, f"{device_field}_supply_air_flowrate_sensor"),
                "of": locate(AHS_DEVICE, f"{device_field}_outside_air_flowrate_sensor"),
                "sat": locate(AHS_DEVICE, f"{device_field}_supply_air_temperature_setpoint"),
                "rc": locate(AHS_DEVICE, f"{device_field}_recirculation_percentage_command"),
                "cr": locate(AHS_DEVICE, f"{device_field}_cooling_request_count"),
                "run": locate(AHS_DEVICE, f"{device_field}_supervisor_run_command"),
            })
            self.cont_slot[meta["temp_indices"][k]] = CN_AA + 4 * k
            self.cont_slot[meta["pressure_indices"][k]] = CN_AA + 4 * k + 1
            self.cont_slot[meta["recirculation_indices"][k]] = CN_AA + 4 * k + 2
            if k < self.nD:
                self.disc_slot[k] = CN_AA + 4 * k + 3
        water = meta.get("water_indices") or []
        self.hW = water[0] if water else -1
        self.hDP = meta["pressure_indices"][-1] if len(meta["pressure_indices"]) > self.K else -1
        if self.hW >= 0:
            self.cont_slot[self.hW] = CN_AH
        if self.hDP >= 0:
            self.cont_slot[self.hDP] = CN_AH + 1
        if self.nD > 0:
            self.disc_slot[self.nD - 1] = CN_AH + 2
        self.h_idx = [locate(HWS_DEVICE, measurement) for measurement in HWS_FIELDS]
        self.g_idx = (
            [idx.get(f"hod_sin_{h:03d}", -1) for h in range(4)]
            + [idx.get(f"hod_cos_{h:03d}", -1) for h in range(4)]
            + [idx.get(f"dow_sin_{h:03d}", -1) for h in range(4)]
            + [idx.get(f"dow_cos_{h:03d}", -1) for h in range(4)]
            + [idx.get("comfort_mode_now", -1), idx.get("comfort_mode_soon", -1), idx.get("num_occupants", -1)]
        )
        self.morning = int(meta["morning_start_hour"])
        self.evening = int(meta["evening_start_hour"])
        lo, hi = meta["comfort_temp_window"]
        self.comfort_lo = f32(lo)
        self.comfort_hi = f32(hi)
        self.holidays = [int(h) for h in (meta.get("holidays") or [])]
        self.start_doy = meta.get("start_doy")
        if self.start_doy is not None:
            self.start_doy = int(self.start_doy)

    def _nat(self, obs, index):
        if index < 0:
            return f32(0)
        return f32(f32(obs[index]) * self.fstd[index] + self.fmean[index])

    def _canon(self, obs):
        x = np.zeros(CN_OBS, np.float32)
        for s, iT, iH, iC, iD, iR in self.zslot:
            T = self._nat(obs, iT)
            Hs = self._nat(obs, iH)
            Cs = self._nat(obs, iC)
            b = CN_FZ * s
            x[b] = 1
            x[b + 1] = (T - f32(295)) / f32(5)
            x[b + 2] = (T - Hs) / f32(2)
            x[b + 3] = (Cs - T) / f32(2)
            x[b + 4] = (Hs - f32(294)) / f32(3)
            x[b + 5] = f32(2) * self._nat(obs, iD) - f32(1) if iD >= 0 else f32(0)
            x[b + 6] = f32(2) * self._nat(obs, iR) - f32(1) if iR >= 0 else f32(0)
        for k, a in enumerate(self.aslot):
            b = CN_OA + CN_FA * k
            x[b] = 1
            x[b + 1] = (self._nat(obs, a["oat"]) - f32(283)) / f32(15)
            x[b + 2] = self._nat(obs, a["sp"]) / f32(1000)
            x[b + 3] = self._nat(obs, a["sf"]) / f32(10)
            x[b + 4] = self._nat(obs, a["of"]) / f32(10)
            x[b + 5] = (self._nat(obs, a["sat"]) - f32(295)) / f32(5) if a["sat"] >= 0 else f32(0)
            x[b + 6] = f32(2) * self._nat(obs, a["rc"]) - f32(1)
            x[b + 7] = self._nat(obs, a["cr"]) / f32(10)
            x[b + 8] = f32(2) * self._nat(obs, a["run"]) - f32(1)
            # The shared actor does not run the base controller: those slots stay
            # at the zero action, so a present handler's mean-temperature slot is
            # (0 - 295) / 5 because the average buffer is left at zero.
            x[b + 12] = f32(-1)
            x[b + 13] = (f32(0) - f32(295)) / f32(5) if self.zc[k] else f32(0)
        hw = [self._nat(obs, i) for i in self.h_idx]
        b = CN_OH
        x[b] = 1
        x[b + 1] = (hw[0] - f32(330)) / f32(20)
        x[b + 2] = (hw[1] - f32(330)) / f32(20)
        x[b + 3] = hw[2] / f32(10)
        x[b + 4] = hw[3] / f32(10)
        x[b + 5] = f32(2) * hw[4] - f32(1)
        x[b + 6] = hw[5] / f32(10)
        x[b + 9] = f32(-1)
        for c, i in enumerate(self.g_idx):
            x[CN_OG + c] = f32(obs[i]) if i >= 0 else f32(0)
        return x

    # --- SAC actor ------------------------------------------------------------
    def _forward(self, o):
        return _actor_forward(self.w, o, self.head, self.beta_floor, self.beta_conc_max)


    def _neutral(self):
        continuous = np.zeros(self.action_spec["continuous_action"].shape, np.float32)
        discrete = np.ones(self.action_spec["discrete_action"].shape, np.int32)
        return {"continuous_action": continuous, "discrete_action": discrete}

    def _decode(self, executed):
        """Canonical action vector -> this building's gateway action dict."""
        cont = np.zeros(self.nC, np.float32)
        disc = np.ones(self.nD, np.int32)
        for i, slot in enumerate(self.cont_slot):
            if slot >= 0:
                cont[i] = executed[slot]
        for i, slot in enumerate(self.disc_slot):
            if slot >= 0:
                disc[i] = np.int32(1 if executed[slot] > f32(-0.5) else 0)
        cont = np.nan_to_num(np.clip(cont, -1, 1), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return {"continuous_action": cont, "discrete_action": disc}


    def get_action(self, obs):
        if self.fallback:
            return self._neutral()
        try:
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            canon = self._canon(obs)
            executed = self._forward(canon)
            return self._decode(executed)
        except Exception:
            traceback.print_exc()
            raise


def _sidecar(path):
    """weights.json next to weights.bin (head, floor, cap); {} when absent."""
    p = Path(path).with_name("weights.json")
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"weights.json unreadable ({type(exc).__name__}: {exc}); assuming gauss", flush=True)
        return {}


def _parse_actor(path):
    """Flat float32 dump -> the 12 actor tensors, in the order SAC/.../export_weights.py wrote."""
    w = np.fromfile(path, dtype=np.float32)
    print(f"loaded weights {path} floats {w.size}", flush=True)
    if w.size != N_ACTOR:
        raise ValueError(f"weights has {w.size} floats, the SAC actor has {N_ACTOR}")
    out, o = {}, 0

    def take(name, rows, cols):
        nonlocal o
        n = rows * cols
        out[name + "_w"] = w[o:o + n].reshape(rows, cols).copy()
        o += n

    def bias(name, n):
        nonlocal o
        out[name + "_b"] = w[o:o + n].copy()
        o += n

    take("trunk", SAC_H, POOL); bias("trunk", SAC_H)
    take("r0", 4, SAC_H + ZONE_L); bias("r0", 4)
    take("r1", 8, SAC_H + AHU_L); bias("r1", 8)
    take("r2", 6, SAC_H + PLANT_L); bias("r2", 6)
    take("zm1", ZM_H, SAC_H + ZONE_L); bias("zm1", ZM_H)
    take("zm2", 4, ZM_H); bias("zm2", 4)
    if o != N_ACTOR:
        raise ValueError(f"weight cursor {o} stopped before {N_ACTOR}")
    return out


Policy = StructuredSacPolicy
SacPolicy = StructuredSacPolicy

if __name__ == "__main__":
    from kaggle_evaluation.smart_buildings_inference_server import SmartBuildingsInferenceServer
    SmartBuildingsInferenceServer(StructuredSacPolicy).run()

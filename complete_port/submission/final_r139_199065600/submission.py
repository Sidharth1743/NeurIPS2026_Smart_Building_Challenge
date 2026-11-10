"""Kaggle policy for the shared canonical actor (zone MLP + Beta head).

The network is the r116/r122 recipe: a fixed 80-value pool of the canonical
observation, a 256-wide encoder, two MinGRU layers, and three local readouts —
one per zone slot (damper, reheat), one per air handler (temperature setpoint,
static-pressure setpoint, recirculation, run command) and one for the hot-water
plant (supply-water setpoint, differential-pressure setpoint, run command).
Each readout reads the same trunk state plus that node's own local block, and the
zone readout also adds a shared 32-unit nonlinear branch (policy.zone_mlp).

Under policy.beta every continuous command is a Beta on [0, 1]: alpha and beta are
softplus of two readout rows plus beta_floor, the total concentration is capped at
beta_conc_max with the mean preserved, and the executed value is
`2 * alpha / (alpha + beta) - 1`. A run command is on when that value is above
-0.5, which is the canonical decode in `decode_res_k` (the neutral 0 is ON, so the
equipment runs unless the policy asks it to stop).

The zone memory (11 features per zone: time to the occupied window, occupancy
change, temperature trend, and the two delayed-bill features) is computed here
from the gateway observation and resets with the policy, which the gateway
rebuilds on every scenario. The canonical observation is rebuilt from the same
vector through `building_meta_data`, so no weather label, month, building id or
path is needed.

Weights are the trainer's flat `master_weights` dump: encoder, value rows,
log-std row, three readouts, the zone branch, then the two MinGRU matrices. The
value head and log-std are read only to walk the layout; the actor does not use
them (the log-std row is untouched under Beta, so it stays at init_logstd).
"""
from __future__ import annotations

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


def _parse_weights(path):
    w = np.fromfile(path, dtype=np.float32)
    print(f"loaded weights {path} floats {w.size}", flush=True)
    if w.size != N_PARAMS:
        raise ValueError(f"weights has {w.size} floats, this network has {N_PARAMS}")
    o = 0

    def take(rows, cols, transpose=False):
        nonlocal o
        n = rows * cols
        a = w[o:o + n]
        o += n
        a = a.reshape(rows, cols)
        return np.ascontiguousarray(a.T) if transpose else np.array(a)

    out = {
        "encoder": take(H, SBS_POOL, transpose=True),
        "value": take(VALUE_ROWS, VALUE_DIN),
        "logstd": take(1, ACT_N),
        "zone": take(ROLE_ROWS[0], H + CN_ZM, transpose=True),
        "ahu": take(ROLE_ROWS[1], H + CN_LA, transpose=True),
        "plant": take(ROLE_ROWS[2], H + CN_LH, transpose=True),
        "zone_mlp1": take(ZONE_MLP_HIDDEN, H + CN_ZM, transpose=True),
        "zone_mlp2": take(ROLE_ROWS[0], ZONE_MLP_HIDDEN, transpose=True),
    }
    out["gru"] = [take(3 * H, H, transpose=True) for _ in range(NUM_LAYERS)]
    if o != N_PARAMS:
        raise ValueError(f"weight cursor {o} stopped before {N_PARAMS}")
    return out


class TransferPolicy(_Base):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.fallback = False
        self.w = _parse_weights(_find_weights())
        try:
            self._layout(building_meta_data)
        except DoesNotFit as exc:
            print(f"canonical slots do not fit this building: {exc}", flush=True)
            self.fallback = True
            return
        self._reset_state()

    # --- metadata -> canonical slots -----------------------------------------
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

    def _reset_state(self):
        self.state = [np.zeros(H, np.float32), np.zeros(H, np.float32)]
        self.ring_t = np.zeros((CN_ZS, 48), np.float32)
        self.ring_d = np.zeros((CN_ZS, 48), np.float32)
        self.ring_n = np.zeros(CN_ZS, np.int32)
        self.fresh = True
        self.last_oat = f32(0)
        self.step = 0

    # --- gateway observation -> canonical observation ------------------------
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

    # --- zone memory (env.zone_memory) --------------------------------------
    def _clock(self, obs):
        hs = self._nat(obs, self.g_idx[0])
        hc = self._nat(obs, self.g_idx[4])
        ds = self._nat(obs, self.g_idx[8])
        dc = self._nat(obs, self.g_idx[12])
        ang = f32(math.atan2(float(hs), float(hc)))
        if ang < 0:
            ang = f32(ang + TWO_PI)
        dang = f32(math.atan2(float(ds), float(dc)))
        if dang < 0:
            dang = f32(dang + TWO_PI)
        hour = f32(ang / TWO_PI * f32(24))
        weekday = int(np.rint(dang / TWO_PI * f32(7))) % 7
        minute = int(np.rint(hour * f32(60))) % 1440
        return hour, weekday, minute

    def _schedule(self, weekday, minute, doy):
        start = self.morning * 60
        end = self.evening * 60
        today = (weekday % 7) < 5 and not (doy is not None and doy in self.holidays)
        if today and start <= minute < end:
            to_start, to_end, during = 0, end - minute, 1
        elif today and minute < start:
            to_start, to_end, during = start - minute, end - minute, 0
        else:
            ahead = 1
            while ahead <= 7:
                day = doy + ahead if doy is not None else None
                ok = ((weekday + ahead) % 7) < 5 and not (day is not None and day in self.holidays)
                if ok:
                    break
                ahead += 1
            to_start = ahead * 1440 - minute + start
            to_end = to_start + (end - start)
            during = 0
        tomorrow_day = doy + 1 if doy is not None else None
        tomorrow = 1.0 if ((weekday + 1) % 7) < 5 and not (
            tomorrow_day is not None and tomorrow_day in self.holidays) else 0.0
        return to_start, to_end, tomorrow, during

    def _features(self, obs):
        _, weekday, minute = self._clock(obs)
        doy = None if self.start_doy is None else self.start_doy + (self.morning * 60 + self.step * 5) // 1440
        to_start, to_end, tomorrow, during = self._schedule(weekday, minute, doy)
        oat = self._nat(obs, self.aslot[0]["oat"]) if self.aslot else f32(0)
        oat_delta = f32(0) if self.fresh else f32(oat - self.last_oat)
        self.last_oat = oat
        self.fresh = False
        lo, hi = self.comfort_lo, self.comfort_hi
        feat = np.zeros((CN_ZS, SBS_FEAT), np.float32)
        for s, iT, iH, iC, iD, iR in self.zslot:
            temp = self._nat(obs, iT)
            deficit = f32(max(0.0, max(float(lo - temp), float(temp - hi))))
            nprev = int(self.ring_n[s])
            slot = nprev % 48
            self.ring_t[s, slot] = temp
            self.ring_d[s, slot] = deficit
            n = nprev + 1
            nuse = 48 if n > 48 else n
            self.ring_n[s] = n
            n1 = nuse if nuse < 12 else 12
            d1 = f32(0)
            d4 = f32(0)
            t4 = f32(0)
            for k in range(nuse):
                at = (nprev - k) % 48
                d4 = f32(d4 + self.ring_d[s, at])
                t4 = f32(t4 + self.ring_t[s, at])
                if k < n1:
                    d1 = f32(d1 + self.ring_d[s, at])
            d1 = f32(d1 / f32(n1))
            d4 = f32(d4 / f32(nuse))
            t4 = f32(t4 / f32(nuse))
            hours = f32((nuse - 1) * 5.0 / 60.0)
            oldest = self.ring_t[s, (nprev - (nuse - 1)) % 48]
            slope = f32((temp - oldest) / hours) if hours > 0 else f32(0)
            feat[s, 0] = f32(to_start / 180.0)
            feat[s, 1] = f32(to_end / 180.0)
            feat[s, 2] = f32(tomorrow)
            feat[s, 3] = f32(oat_delta / f32(2))
            feat[s, 4] = f32(deficit / f32(5))
            feat[s, 5] = f32(d1 / f32(5))
            feat[s, 6] = f32(d4 / f32(5))
            feat[s, 7] = f32(slope / f32(5))
            feat[s, 8] = f32((t4 - f32(295)) / f32(5))
            horizon = f32(60) if during else f32(min(float(to_start), 240.0))
            ahead = f32(temp + slope * (horizon / f32(60)))
            dhat = f32(max(0.0, max(float(lo - ahead), float(ahead - hi))))
            feat[s, 9] = f32(min(float(dhat / f32(5)), 10.0))
            damper = f32(min(1.0, max(0.0, float(self._nat(obs, iD))))) if iD >= 0 else f32(0)
            reheat = f32(min(1.0, max(0.0, float(self._nat(obs, iR))))) if iR >= 0 else f32(1)
            kahu = s // CN_ZPA
            sat_i = self.aslot[kahu]["sat"] if kahu < len(self.aslot) else -1
            has_sat = sat_i >= 0
            sat = self._nat(obs, sat_i) if has_sat else temp
            has_r = iR >= 0
            if has_r and has_sat:
                idle = reheat < f32(0.3) and sat <= temp
            elif has_r:
                idle = reheat < f32(0.3)
            elif has_sat:
                idle = sat <= temp
            else:
                idle = False
            heat = self._nat(obs, iH) if iH >= 0 else lo
            below = f32(max(0.0, float(heat - temp)) / 5.0)
            feat[s, 10] = f32(min(float(damper * below * (f32(1) if idle else f32(0))), 10.0))
        return feat

    # --- actor --------------------------------------------------------------
    def _role_input(self, h, o, role, node, local_width, zone_feat):
        exists = (o[CN_FZ * node] != 0) if role == 0 else (
            (o[CN_OA + CN_FA * node] != 0) if role == 1 else (o[CN_OH] != 0))
        if not exists:
            return None
        x = np.empty(H + local_width, np.float32)
        x[:H] = h
        for i in range(local_width):
            x[H + i] = np.arcsinh(_local_feature(o, role, node, i, zone_feat))
        return x

    def _forward(self, o, feat):
        w = self.w
        pool = np.empty(SBS_POOL, np.float32)
        for f in range(SBS_POOL):
            x = _pool_feature(o, f)
            pool[f] = np.arcsinh(x) if (f < 51 or f >= 72) else x
        h = (pool @ w["encoder"]).astype(f32)
        for i in range(NUM_LAYERS):
            c = (h @ w["gru"][i]).astype(f32)
            hidden, gate, proj = c[:H], c[H:2 * H], c[2 * H:]
            z = _sigmoid(gate)
            ht = np.where(hidden >= 0, hidden + f32(0.5), _sigmoid(hidden)).astype(f32)
            hout = _lerp(self.state[i], ht, z)
            self.state[i] = hout
            s = _sigmoid(proj)
            h = (s * hout + (f32(1) - s) * h).astype(f32)
        out = np.zeros(CN_ACT, np.float32)
        for node in range(CN_ZS):
            x = self._role_input(h, o, 0, node, CN_ZM, feat[node])
            if x is None:
                continue
            r = (x @ w["zone"]).astype(f32)
            r = (r + np.tanh(x @ w["zone_mlp1"]).astype(f32) @ w["zone_mlp2"]).astype(f32)
            out[2 * node:2 * node + 2] = _beta_action(r[:2], r[2:4])
        for k in range(CN_K):
            x = self._role_input(h, o, 1, k, CN_LA, None)
            if x is None:
                continue
            r = (x @ w["ahu"]).astype(f32)
            b = CN_AA + 4 * k
            out[b:b + 4] = _beta_action(r[:4], r[4:8])
        x = self._role_input(h, o, 2, 0, CN_LH, None)
        if x is not None:
            r = (x @ w["plant"]).astype(f32)
            out[CN_AH:CN_AH + 3] = _beta_action(r[:3], r[3:6])
        return out

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
            feat = self._features(obs)
            executed = self._forward(canon, feat)
            self.step += 1
            return self._decode(executed)
        except Exception:
            traceback.print_exc()
            raise


Policy = TransferPolicy
R122Policy = TransferPolicy
SharedBetaPolicy = TransferPolicy

if __name__ == "__main__":
    from kaggle_evaluation.smart_buildings_inference_server import SmartBuildingsInferenceServer
    SmartBuildingsInferenceServer(TransferPolicy).run()

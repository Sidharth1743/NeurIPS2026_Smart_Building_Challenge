"""Competition policy: canonical-layout residual PPO over pid_water (numpy, CPU).

Mirrors, for one building, exactly what the GPU env + PufferLib trainer compute (complete_port/gpu/sb_gpu_base.inc,
PufferLib encoder -> minGRU -> decoder):

  1. base = starter-kit PIDWaterResetPolicy with each AHU's zones in observation order (bit-identical to the CUDA
     base: complete_port gate `basecheck` + research/tools/basecheck.py --obs-order);
  2. the canonical observation (CN_OBS scaled per-slot features) from the oracle observation + the base proposal;
  3. the network (fp32): h = obs @ We^T; per minGRU layer: c = x @ Wi^T, gate as PufferLib's mingru_gate; out = x @ Wd^T,
     action mean = tanh(out[slot]) (tanh_mean), deterministic;
  4. residual decode: continuous a = clip(base + res_scale * clip(p, -1, 1), -1, 1); run bit = (2 base - 1) +
     res_dscale * clip(p, -1, 1) > 0;  with the comfort guard (config guard_d1 > 0) p is first multiplied by the
     action's trust (sb_gpu_base.inc guard_k: 1 -> 0 as the zone / the AHU's worst zone / the building's worst zone
     goes guard_d0 .. guard_d0 + guard_d1 K below the base's band).

Reads only building_meta_data and the observation. Actions are always finite and inside [-1, 1] / {0, 1}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

try:   # the competition base class when the gateway is installed; a stand-in otherwise
    from kaggle_evaluation.smart_buildings_inference_server import Policy as _Base
except Exception:  # noqa: BLE001
    class _Base:  # type: ignore[no-redef]
        def __init__(self, building_meta_data, action_spec):
            self.building_meta_data = building_meta_data
            self.action_spec = action_spec

from pidw_base import PIDWaterResetPolicy   # vendored starter-kit baselines (Apache-2.0)

CN_K, CN_ZPA = 9, 10
CN_ZS = CN_K * CN_ZPA
CN_FZ, CN_FA, CN_FH, CN_FG = 9, 16, 10, 19
CN_OA = CN_ZS * CN_FZ
CN_OH = CN_OA + CN_K * CN_FA
CN_OG = CN_OH + CN_FH
CN_OBS = CN_OG + CN_FG
CN_AA = 2 * CN_ZS
CN_AH = CN_AA + 4 * CN_K
f32 = np.float32


def _sigmoid(x):
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, f32(1) / (f32(1) + z), z / (f32(1) + z)).astype(f32)


class ResidualPolicy(_Base):
    MODEL_DIR = Path(__file__).resolve().parent / "model"

    def __init__(self, building_meta_data, action_spec, model_dir=None):
        super().__init__(building_meta_data, action_spec)
        import os
        md = Path(model_dir or os.environ.get("SB_POLICY_MODEL_DIR") or self.MODEL_DIR)
        cfg = json.loads((md / "config.json").read_text())
        self.res_scale, self.res_dscale = f32(cfg["res_scale"]), f32(cfg["res_dscale"])
        self.residual = bool(cfg.get("residual", True))
        self.g_d0, self.g_d1 = f32(cfg.get("guard_d0", 1.0)), f32(cfg.get("guard_d1", 0.0))
        self.g_hot = bool(cfg.get("guard_hot", False))
        H, L, OBS, OD = cfg["hidden"], cfg["layers"], cfg["obs_size"], cfg["out_dim"]
        w = np.fromfile(md / cfg["weights"], dtype=np.float32)
        o = 0
        def take(r, c):
            nonlocal o
            a = w[o:o + r * c].reshape(r, c); o += r * c
            return a
        self.We = take(H, OBS)
        self.Wd = take(OD + 1, H)
        take(1, OD)                                   # log-std (unused: deterministic)
        self.Wg = [take(3 * H, H) for _ in range(L)]
        assert o == len(w), f"weights file has {len(w)} floats, layout needs {o}"
        self.H, self.OBS = H, OBS
        self.state = [np.zeros(H, f32) for _ in range(L)]

        m = building_meta_data
        self.base = PIDWaterResetPolicy(m, action_spec)
        # a building outside the canonical slots (> CN_K AHUs or > CN_ZPA zones on an AHU) runs the base alone
        try:
            self._build_layout(m)
            self.fallback = False
        except (AssertionError, KeyError, IndexError):
            self.fallback = True

    def _build_layout(self, m):
        for ctrl in self.base.ahu_controllers:        # observation order (the GPU base's convention)
            ctrl["zones"] = sorted(ctrl["zones"], key=lambda z: z[1])
            ctrl["zone_obs_indices"] = [z[1] for z in ctrl["zones"]]
        names = list(m["observation_field_names"])
        idx = {n: i for i, n in enumerate(names)}
        self.n_obs = len(names)
        nm = m["observation_normalization_metadata"]
        # native value of each observation entry: obs * std + mean where the field is normalized
        self.fstd = np.ones(self.n_obs, f32); self.fmean = np.zeros(self.n_obs, f32)
        for i, n in enumerate(names):
            meas = re.sub(r"^(VAV_\d+_|ahs_ahu_\d+_|hws_)", "", n)
            if meas in nm and nm[meas].get("sample_variance", 0) > 0:
                self.fmean[i] = f32(nm[meas]["sample_mean"]); self.fstd[i] = f32(np.sqrt(nm[meas]["sample_variance"]))
        zl = m["observation_normalization_metadata"]["zone_air_temperature_sensor"]
        self.tmean, self.tstd = f32(zl["sample_mean"]), f32(zl["sample_std"])
        # slots
        self.nC, self.nD = m["n_continuous_action"], m["n_discrete_action"]
        ahu_ids = sorted(m["ahu_zone_assignments"].keys(), key=lambda a: int(re.search(r"\d+", a).group()))
        self.K = len(m["temp_indices"])
        assert self.K <= CN_K, "building has more AHUs than the canonical slots"
        room_ahu = {r: k for k, a in enumerate(ahu_ids) for r in m["ahu_zone_assignments"][a]}
        self.cont_slot = np.full(self.nC, -1, np.int64)       # continuous action index -> canonical slot
        self.disc_slot = np.full(self.nD, -1, np.int64)
        self.zslot = []                                         # (slot, obs indices)
        cnt = [0] * CN_K
        for n in names:
            mt = re.match(r"(VAV_\d+)_zone_air_temperature_sensor$", n)
            if not mt:
                continue
            vav = mt.group(1); k = room_ahu[vav.replace("VAV_", "room_")]
            assert cnt[k] < CN_ZPA, "AHU serves more zones than the canonical slots"
            s = CN_ZPA * k + cnt[k]; cnt[k] += 1
            g = lambda f: idx.get(f"{vav}_{f}", -1)  # noqa: E731
            self.zslot.append((s, idx[n], g("zone_air_heating_temperature_setpoint_sensor"),
                               g("zone_air_cooling_temperature_setpoint_sensor"),
                               g("supply_air_damper_percentage_command"), g("reheat_valve_percentage_command"),
                               m["zone_action_indices"].get(vav, {}).get("supply_air_damper_percentage_command", -1),
                               m["zone_action_indices"].get(vav, {}).get("reheat_valve_percentage_command", -1)))
            ai = m["zone_action_indices"].get(vav, {})
            if "supply_air_damper_percentage_command" in ai: self.cont_slot[ai["supply_air_damper_percentage_command"]] = 2 * s
            if "reheat_valve_percentage_command" in ai: self.cont_slot[ai["reheat_valve_percentage_command"]] = 2 * s + 1
        self.zc = cnt
        self.aslot = []
        for k in range(self.K):
            pre = f"ahs_ahu_{k + 1}_"
            g = lambda f: idx.get(pre + f, -1)  # noqa: E731
            self.aslot.append(dict(oat=g("outside_air_temperature_sensor"), sp=g("supply_air_static_pressure_sensor"),
                                   sf=g("supply_air_flowrate_sensor"), of=g("outside_air_flowrate_sensor"),
                                   sat=g("supply_air_temperature_setpoint"), rc=g("recirculation_percentage_command"),
                                   cr=g("cooling_request_count"), run=g("supervisor_run_command"),
                                   aT=m["temp_indices"][k], aP=m["pressure_indices"][k], aR=m["recirculation_indices"][k]))
            self.cont_slot[m["temp_indices"][k]] = CN_AA + 4 * k
            self.cont_slot[m["pressure_indices"][k]] = CN_AA + 4 * k + 1
            self.cont_slot[m["recirculation_indices"][k]] = CN_AA + 4 * k + 2
            self.disc_slot[k] = CN_AA + 4 * k + 3
        self.hW = m["water_indices"][0] if m["water_indices"] else -1
        self.hDP = m["pressure_indices"][-1] if len(m["pressure_indices"]) > self.K else -1
        if self.hW >= 0: self.cont_slot[self.hW] = CN_AH
        if self.hDP >= 0: self.cont_slot[self.hDP] = CN_AH + 1
        self.disc_slot[self.nD - 1] = CN_AH + 2
        g = lambda f: idx.get(f, -1)  # noqa: E731
        self.h_idx = [g("hws_supply_water_temperature_sensor"), g("hws_supply_water_temperature_setpoint"),
                      g("hws_differential_pressure_sensor"), g("hws_differential_pressure_setpoint"),
                      g("hws_run_status"), g("hws_heating_request_count")]
        self.g_idx = ([g(f"hod_sin_{h:03d}") for h in range(4)] + [g(f"hod_cos_{h:03d}") for h in range(4)]
                      + [g(f"dow_sin_{h:03d}") for h in range(4)] + [g(f"dow_cos_{h:03d}") for h in range(4)]
                      + [g("comfort_mode_now"), g("comfort_mode_soon"), g("num_occupants")])

    # ---------------------------------------------------------------------------------------------------------------
    def _nat(self, o, i):
        return o[i] * self.fstd[i] + self.fmean[i] if i >= 0 else f32(0)

    def _features(self, o, bc, bd):
        x = np.zeros(self.OBS, f32)
        for (s, iT, iH, iC, iD, iR, aD, aR) in self.zslot:
            T, Hs, Cs = self._nat(o, iT), self._nat(o, iH), self._nat(o, iC)
            b = CN_FZ * s
            x[b:b + CN_FZ] = [1, (T - 295) / 5, (T - Hs) / 2, (Cs - T) / 2, (Hs - 294) / 3,
                              2 * self._nat(o, iD) - 1 if iD >= 0 else 0, 2 * self._nat(o, iR) - 1 if iR >= 0 else 0,
                              bc[aD] if aD >= 0 else 0, bc[aR] if aR >= 0 else 0]
        st = self.base.pid_state
        efforts = getattr(self.base, "_eff_by_ahu", {})
        for k, a in enumerate(self.aslot):
            b = CN_OA + CN_FA * k
            nz = self.zc[k]
            zT = [self._nat(o, z[1]) for z in self.zslot if z[0] // CN_ZPA == k]
            x[b:b + CN_FA] = [1, (self._nat(o, a["oat"]) - 283) / 15, self._nat(o, a["sp"]) / 1000,
                              self._nat(o, a["sf"]) / 10, self._nat(o, a["of"]) / 10,
                              (self._nat(o, a["sat"]) - 295) / 5 if a["sat"] >= 0 else 0, 2 * self._nat(o, a["rc"]) - 1,
                              self._nat(o, a["cr"]) / 10, 2 * self._nat(o, a["run"]) - 1,
                              bc[a["aT"]], bc[a["aP"]], bc[a["aR"]], 2 * f32(bd[k]) - 1,
                              (np.mean(np.asarray(zT, f32)) - 295) / 5 if nz > 0 else 0,
                              st[k]["integral"] / 10, efforts.get(k, 0.0)]
        b = CN_OH
        hw = [self._nat(o, i) for i in self.h_idx]
        x[b:b + CN_FH] = [1, (hw[0] - 330) / 20, (hw[1] - 330) / 20, hw[2] / 10, hw[3] / 10, 2 * hw[4] - 1, hw[5] / 10,
                          bc[self.hW] if self.hW >= 0 else 0, bc[self.hDP] if self.hDP >= 0 else 0,
                          2 * f32(bd[self.nD - 1]) - 1]
        x[CN_OG:CN_OG + CN_FG] = [o[i] if i >= 0 else 0 for i in self.g_idx]
        return x

    def _trust(self, o):
        """Per-slot residual authority (the GPU's guard_k, float32 op by op), indexed by canonical action slot."""
        w = np.ones(CN_AH + 3, f32)
        if not (self.residual and self.g_d1 > 0):
            return w
        hour, day = self.base._get_hour_and_day(o)
        work = day < 5 and self.base.morning_start_hour <= hour < self.base.evening_start_hour
        lo, hi = self.base.comfort_temp_window if work else self.base.eco_temp_window
        lof, hif = f32(lo), f32(hi)
        wk = [f32(1)] * CN_K
        for (s, iT, *_r) in self.zslot:
            T = f32(o[iT] * self.tstd) + self.tmean
            dev = lof - T if T < lof else (T - hif if self.g_hot and T > hif else f32(0))
            t = f32(min(f32(1), max(f32(0), f32(1) - f32(f32(dev - self.g_d0) / self.g_d1))))
            w[2 * s] = w[2 * s + 1] = t
            k = s // CN_ZPA
            wk[k] = min(wk[k], t)
        for k in range(CN_K):
            w[CN_AA + 4 * k:CN_AA + 4 * k + 4] = wk[k]
        w[CN_AH:CN_AH + 3] = min(wk[:self.K]) if self.K else f32(1)
        return w

    def _net(self, x):
        h = (self.We @ x).astype(f32)
        for i, W in enumerate(self.Wg):
            c = (W @ h).astype(f32)
            H = self.H
            hid, gate, proj = c[:H], c[H:2 * H], c[2 * H:]
            z = _sigmoid(gate)
            ht = np.where(hid >= 0, hid + f32(0.5), _sigmoid(hid)).astype(f32)
            s0 = self.state[i]
            diff = ht - s0
            hout = np.where(np.abs(z) < 0.5, s0 + z * diff, ht - diff * (f32(1) - z)).astype(f32)
            self.state[i] = hout
            sg = _sigmoid(proj)
            h = (sg * hout + (f32(1) - sg) * h).astype(f32)
        return (self.Wd @ h).astype(f32)

    def get_action(self, obs):
        o = np.asarray(obs, f32)
        if self.fallback:
            a = self.base.get_action(o.copy())
            return {"continuous_action": np.nan_to_num(np.clip(np.asarray(a["continuous_action"], np.float32), -1, 1)),
                    "discrete_action": np.asarray(a["discrete_action"], np.int32)}
        # base proposal (updates the PID state once per step, as on the GPU); record each AHU's effort for features
        a = self.base.get_action(o.copy())
        bc = np.asarray(a["continuous_action"], f32)
        bd = np.asarray(a["discrete_action"], np.int32)
        eff = {}
        j = 0
        for k, ctrl in enumerate(self.base.ahu_controllers):
            if ctrl["zone_obs_indices"] and bd[k] == 1:
                eff[k] = self.base._last_efforts[j]; j += 1
        self.base._eff_by_ahu = eff
        y = self._net(self._features(o, bc, bd))
        p = np.clip(np.tanh(y), -1, 1).astype(f32)
        p[:CN_AH + 3] = (p[:CN_AH + 3] * self._trust(o)).astype(f32)
        cont = bc.copy()
        for i in range(self.nC):
            s = self.cont_slot[i]
            if s < 0:
                continue
            cont[i] = np.clip(bc[i] + self.res_scale * p[s], -1, 1) if self.residual else np.clip(p[s], -1, 1)
        disc = bd.copy()
        for i in range(self.nD):
            s = self.disc_slot[i]
            if s < 0:
                continue
            v = (f32(2) * f32(bd[i]) - 1 + self.res_dscale * p[s]) if self.residual else p[s]
            disc[i] = 1 if v > 0 else 0
        cont = np.nan_to_num(np.clip(cont, -1, 1), nan=0.0).astype(np.float32)
        return {"continuous_action": cont, "discrete_action": disc.astype(np.int32)}

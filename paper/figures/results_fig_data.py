#!/usr/bin/env python3
"""Data layer for the Results figures: 4 policies x 90 months, plus the 54-month extension.

Policies: PID, Long (r122 @89.6M), Mid (r139 @159.3M), Final (r139 @199.1M).
Writes nothing; other scripts import load().
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CP = Path(os.environ.get("SB_ROOT", Path(__file__).resolve().parents[1])) / "complete_port"
RES = CP / "results"
SET = CP / "artifacts/sets/eval5_heldout18"
H, N = 8640, 18
BUILDINGS = ["building_train_1", "building_train_2", "building_train_3",
             "building_train_6", "building_train_8"]
EXT = ["building_train_4", "building_train_5", "building_train_7"]
STATIONS = ["atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore"]
POL = ["PID", "SAC", "Long", "Mid", "Final"]
COL = {"PID": "#B64342", "SAC": "#9A4D8E", "Long": "#767676", "Mid": "#42949E", "Final": "#0F4D92"}
SAC90 = CP.parent / "SAC/results/r15_structured_beta_b2048_s75/scored90"
SAC54 = CP.parent / "SAC/results/r15_structured_beta_b2048_s75/scored54"

N90 = RES / "study_r139_eval90/gpu"
CK = {
    "Final": N90 / "r139/0000000199065600",
    "Mid": N90 / "r139_159252480/0000000159252480",
    "Long": RES / "study_r122_eval90/gpu/r122/0000000089579520",
}
PID90 = RES / "study_r122_eval90/pid_water/scores"
N54 = RES / "study_eval457"
CK54 = {
    "Final": N54 / "r139/0000000199065600",
    "Mid": N54 / "r139_159m/0000000159252480",
    "Long": N54 / "r122/0000000089579520",
}
PID54 = N54 / "pid_water"

# checkpoint steps / public leaderboard (from the run records)
STEPS = {
    "Long": {"local": [89.579520, 119.439360], "public": [59.719680, 89.579520, 119.439360, 209.018880, 238.878720]},
    "Mid": {"local": [159.252480], "public": [159.252480]},
    "Final": {"local": [199.065600], "public": [199.065600]},
}
PUBLIC = {  # steps (M) -> score
    "Long": {59.719680: 0.01842, 89.579520: 0.01818, 119.439360: 0.02104,
             209.018880: 0.02262, 238.878720: 0.02392},
    "Mid": {159.252480: 0.01750},
    "Final": {199.065600: 0.02944},
}
BUDGET = {"Long": (1500180480 / 1e6, 507617280 / 1e6), "Anneal": (199895040 / 1e6, 199895040 / 1e6)}


def station(name: str) -> str:
    return name.split("__")[1].rsplit("_", 1)[0]


def load_scores():
    """policy -> {scenario: (score, comfort, energy, carbon)} on the 90 months."""
    out = {}
    for p, d in CK.items():
        j = json.loads((d / "score.json").read_text())["scenarios"]
        out[p] = {k: (abs(v["mean_reward"]), 0.6 * abs(v["productivity"]),
                      0.2 * v["energy_cost"], 0.2 * v["carbon"]) for k, v in j.items()}
    pid = {}
    for s in PID90.glob("*/summary.json"):
        v = json.loads(s.read_text())
        pid[v["scenario"]] = (v["score"], 0.6 * abs(v["productivity"]),
                              0.2 * v["energy_cost"], 0.2 * v["carbon"])
    out["PID"] = pid
    sac = {}
    for f in SAC90.glob("50135040_b*.json"):
        for k, v in json.loads(f.read_text())["scenarios"].items():
            sac[k] = (abs(v["mean_reward"]), 0.6 * abs(v["productivity"]),
                      0.2 * v["energy_cost"], 0.2 * v["carbon"])
    out["SAC"] = sac
    return out


def load_scores54():
    out = {}
    for p, d in CK54.items():
        j = json.loads((d / "score.json").read_text())["scenarios"]
        out[p] = {k: abs(v["mean_reward"]) for k, v in j.items()}
    pid = {}
    for s in PID54.glob("*/summary.json"):
        v = json.loads(s.read_text())
        pid[v["scenario"]] = v["score"]
    out["PID"] = pid
    sac = {}
    for f in SAC54.glob("50135040_b*.json"):
        for k, v in json.loads(f.read_text())["scenarios"].items():
            sac[k] = abs(v["mean_reward"])
    out["SAC"] = sac
    return out


def scenario_names(b: str) -> list[str]:
    return sorted(p.stem for p in SET.glob(f"{b}__*.sbb"))


def load_steps():
    """policy -> {scenario: reward per step (float32, 8640)} on the 90 months."""
    out = {}
    for p, d in CK.items():
        out[p] = {}
        for b in BUILDINGS:
            parts = np.fromfile(d / b / "record/parts.bin", dtype=np.float64).reshape(H, N, 8)
            r = (0.6 * parts[:, :, 0] - 0.2 * parts[:, :, 1] - 0.2 * parts[:, :, 2]).astype(np.float32)
            for i, n in enumerate(scenario_names(b)):
                out[p][n] = r[:, i]
    out["SAC"] = {}
    sac_rec = SAC90.parent / "record90" / "50135040"
    for b in BUILDINGS:
        parts = np.fromfile(sac_rec / b / "parts.bin", dtype=np.float64).reshape(H, N, 8)
        r = (0.6 * parts[:, :, 0] - 0.2 * parts[:, :, 1] - 0.2 * parts[:, :, 2]).astype(np.float32)
        for i, n in enumerate(scenario_names(b)):
            out["SAC"][n] = r[:, i]
    out["PID"] = {}
    for n in sorted({k for k in load_scores()["PID"]}):
        import pandas as pd
        r = pd.read_csv(PID90 / n / "reward_components.csv", usecols=["agent_reward"])["agent_reward"].to_numpy(np.float32)
        assert r.size == H, (n, r.size)
        out["PID"][n] = r
    return out


def month_of(scenario: str) -> int:
    import pandas as pd
    ts = pd.to_datetime(pd.read_csv(PID90 / scenario / "reward_components.csv",
                                    usecols=["timestamp"])["timestamp"].iloc[0])
    return ts.month


# time index: every month starts Sunday 06:00
T = np.arange(H)
MINW = (360 + 5 * T) % (7 * 1440)
HOUR = (MINW % 1440) // 60
WEEKEND = np.isin(MINW // 1440, [0, 6])
BIN = np.where(WEEKEND & (HOUR >= 6) & (HOUR < 19), "weekend day",
      np.where(WEEKEND, "weekend night",
      np.where((HOUR >= 19) | (HOUR < 6), "weekday night",
      np.where(HOUR < 12, "weekday morning",
      np.where(HOUR < 16, "weekday afternoon", "weekday evening")))))
BINS = ["weekday night", "weekday morning", "weekday afternoon", "weekday evening",
        "weekend day", "weekend night"]

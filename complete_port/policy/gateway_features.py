"""Features computed inside the policy from the gateway observation and metadata.

The competition observation is unchanged. Anticipation uses the timestamp
already in the observation (hour and weekday sin/cos) and the schedule
metadata evaluate.py passes to the policy: morning and evening hours, the
comfort and eco windows, the time zone, and the holiday set. The outside-air
trend is the difference from the previous step. Per-zone memory is a ring the
policy updates in get_action. Nothing here reads the simulator, arrival
parameters, or zone_reward_infos.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

import numpy as np

RING = 48
STEP_MIN = 5
FEAT = 11


def _spec(meta: Mapping, name: str):
    if not isinstance(meta, Mapping):
        return None
    if name in meta:
        return meta[name]
    best = None
    for key, spec in meta.items():
        if name.endswith(str(key)) and (best is None or len(str(key)) > len(best[0])):
            best = (str(key), spec)
    return None if best is None else best[1]


def _denorm(obs: np.ndarray, index: int, meta: Mapping, name: str) -> float:
    spec = _spec(meta, name)
    value = float(obs[index])
    if not spec:
        return value
    return value * float(spec["sample_std"]) + float(spec["sample_mean"])


def _angle(sin_v: float, cos_v: float) -> float:
    ang = math.atan2(sin_v, cos_v)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return ang


def _clock(obs, names, meta) -> tuple[float, int]:
    """Local hour in [0, 24) and weekday with Monday = 0."""
    hs = names.index("hod_sin_000")
    hc = names.index("hod_cos_000")
    ds = names.index("dow_sin_000")
    dc = names.index("dow_cos_000")
    hour = _angle(_denorm(obs, hs, meta, "hod_sin_000"), _denorm(obs, hc, meta, "hod_cos_000")) / (2.0 * math.pi) * 24.0
    dow = _angle(_denorm(obs, ds, meta, "dow_sin_000"), _denorm(obs, dc, meta, "dow_cos_000")) / (2.0 * math.pi) * 7.0
    return hour % 24.0, int(round(dow)) % 7


def _holiday(doy: int | None, holidays) -> bool:
    if doy is None or not holidays:
        return False
    return int(doy) in set(int(h) for h in holidays)


class ZoneMemory:
    """Per-zone rings. Reset at the start of every episode."""

    def __init__(self, n_zones: int):
        self.n = n_zones
        self.temp = np.zeros((n_zones, RING), np.float64)
        self.deficit = np.zeros((n_zones, RING), np.float64)
        self.count = np.zeros(n_zones, np.int32)
        self.last_oat = None
        self.last_minute = None
        self.doy = None

    def reset(self) -> None:
        self.temp[:] = 0.0
        self.deficit[:] = 0.0
        self.count[:] = 0
        self.last_oat = None
        self.last_minute = None
        self.doy = None


def schedule_minutes(hour: float, weekday: int, morning: int, evening: int,
                     holidays, doy: int | None) -> tuple[float, float, float, float]:
    """Minutes to the next comfort start, minutes to the matching end, tomorrow-is-workday, in comfort."""
    minute = hour * 60.0
    start = float(morning) * 60.0
    end = float(evening) * 60.0
    today_h = _holiday(doy, holidays)

    def work(day_index: int, day_of_year: int | None) -> bool:
        return (day_index % 7) < 5 and not _holiday(day_of_year, holidays)

    during = work(weekday, doy) and start <= minute < end
    if during:
        to_start, to_end = 0.0, end - minute
    elif work(weekday, doy) and minute < start:
        to_start, to_end = start - minute, end - minute
    else:
        ahead = 1
        while ahead <= 7 and not work(weekday + ahead, None if doy is None else doy + ahead):
            ahead += 1
        to_start = ahead * 1440.0 - minute + start
        to_end = to_start + (end - start)
    tomorrow = 1.0 if work(weekday + 1, None if doy is None else doy + 1) else 0.0
    return to_start, to_end, tomorrow, 1.0 if during else 0.0


def _zone_index(names: list[str], zone_name: str, tail: str, new_tail: str) -> int | None:
    if not zone_name.endswith(tail):
        return None
    target = zone_name[: -len(tail)] + new_tail
    try:
        return names.index(target)
    except ValueError:
        return None


def _sat_index(names: list[str], zone_name: str, assignments) -> int | None:
    match = re.search(r"VAV_(\d+)", zone_name)
    if match is None or not isinstance(assignments, Mapping):
        return None
    room = f"room_{match.group(1)}"
    vav = f"VAV_{match.group(1)}"
    ahu = None
    for key, rooms in assignments.items():
        rooms = [str(item) for item in rooms]
        if room in rooms or vav in rooms:
            ahu = str(key).lower()
            break
    if ahu is None:
        return None
    hits = [
        i for i, name in enumerate(names)
        if name.endswith("supply_air_temperature_setpoint") and ahu in name.lower()
    ]
    return hits[0] if hits else None


def _delay_pair(temp: float, slope: float, to_start: float, during: float,
                lo: float, hi: float, heat_sp: float | None,
                damper: float | None, reheat: float | None,
                sat: float | None) -> tuple[float, float]:
    """Deficit at the delayed bill, and the open air path that is not heating."""
    horizon = 60.0 if during else min(to_start, 240.0)
    ahead = temp + slope * (horizon / 60.0)
    deficit_hat = max(0.0, lo - ahead, ahead - hi)
    arrival = min(deficit_hat / 5.0, 10.0)
    line = lo if heat_sp is None else heat_sp
    below = max(0.0, line - temp) / 5.0
    has_r = reheat is not None
    has_sat = sat is not None
    reheats = 1.0 if reheat is None else min(1.0, max(0.0, reheat))
    damp = 0.0 if damper is None else min(1.0, max(0.0, damper))
    if has_r and has_sat:
        idle = reheats < 0.3 and sat <= temp
    elif has_r:
        idle = reheats < 0.3
    elif has_sat:
        idle = sat <= temp
    else:
        idle = False
    fan = min(damp * below * (1.0 if idle else 0.0), 10.0)
    return arrival, fan


def update_features(obs: np.ndarray, meta_data: Mapping, state: ZoneMemory,
                    zone_temp_indices: list[int], oat_index: int) -> tuple[np.ndarray, dict]:
    """Write the 11 features for every zone and advance the rings. Returns (zones, 11) and a summary."""
    obs = np.asarray(obs, dtype=np.float64)
    names = list(meta_data["observation_field_names"])
    norm = meta_data.get("observation_normalization_metadata") or {}
    hour, weekday = _clock(obs, names, norm)
    minute = int(round(hour * 60.0)) % 1440
    if state.last_minute is not None and minute < state.last_minute:
        if state.doy is not None:
            state.doy += 1
    state.last_minute = minute
    doy = state.doy
    morning = int(meta_data["morning_start_hour"])
    evening = int(meta_data["evening_start_hour"])
    holidays = meta_data.get("holidays") or []
    to_start, to_end, tomorrow, during = schedule_minutes(
        hour, weekday, morning, evening, holidays, doy)
    lo, hi = meta_data["comfort_temp_window"]
    lo, hi = float(lo), float(hi)
    ramp = 1.0 if during else ((180.0 - to_start) / 180.0 if 0.0 < to_start <= 180.0 else 0.0)
    oat = _denorm(obs, oat_index, norm, names[oat_index])
    oat_delta = 0.0 if state.last_oat is None else oat - state.last_oat
    state.last_oat = oat
    out = np.zeros((state.n, FEAT), np.float64)
    phi = 0.0
    present = 0
    for z, ti in enumerate(zone_temp_indices):
        temp = _denorm(obs, ti, norm, names[ti])
        deficit = max(0.0, lo - temp, temp - hi)
        slot = int(state.count[z]) % RING
        state.temp[z, slot] = temp
        state.deficit[z, slot] = deficit
        state.count[z] = slot_count = int(state.count[z]) + 1
        n = min(slot_count, RING)
        d_hist = [state.deficit[z, (slot_count - 1 - k) % RING] for k in range(n)]
        t_hist = [state.temp[z, (slot_count - 1 - k) % RING] for k in range(n)]
        d1 = float(np.mean(d_hist[: min(12, n)]))
        d4 = float(np.mean(d_hist))
        t4 = float(np.mean(t_hist))
        hours = (n - 1) * STEP_MIN / 60.0
        slope = 0.0 if hours <= 0.0 else (t_hist[0] - t_hist[-1]) / hours
        band = max(0.0, lo - temp, temp - hi)
        short = band / 5.0 if band <= 5.0 else 1.0 + ((band - 5.0) / 5.0) ** 2
        phi += ramp * short
        present += 1
        zone_name = names[ti]
        damper_i = _zone_index(names, zone_name, "zone_air_temperature_sensor",
                               "supply_air_damper_percentage_command")
        reheat_i = _zone_index(names, zone_name, "zone_air_temperature_sensor",
                               "reheat_valve_percentage_command")
        heat_i = _zone_index(names, zone_name, "zone_air_temperature_sensor",
                             "zone_air_heating_temperature_setpoint_sensor")
        sat_i = _sat_index(names, zone_name, meta_data.get("ahu_zone_assignments"))
        damper = None if damper_i is None else _denorm(obs, damper_i, norm, names[damper_i])
        reheat = None if reheat_i is None else _denorm(obs, reheat_i, norm, names[reheat_i])
        heat_sp = None if heat_i is None else _denorm(obs, heat_i, norm, names[heat_i])
        sat = None if sat_i is None else _denorm(obs, sat_i, norm, names[sat_i])
        arrival, fan = _delay_pair(temp, slope, to_start, during, lo, hi, heat_sp, damper, reheat, sat)
        out[z] = (
            to_start / 180.0, to_end / 180.0, tomorrow, oat_delta / 2.0,
            deficit / 5.0, d1 / 5.0, d4 / 5.0, slope / 5.0, (t4 - 295.0) / 5.0,
            arrival, fan,
        )
    summary = {
        "hour": hour, "weekday": weekday, "minutes_to_start": to_start,
        "minutes_to_end": to_end, "tomorrow_workday": tomorrow, "during": during,
        "oat_delta": oat_delta, "phi": phi / present if present else 0.0,
        "features": out,
    }
    return out, summary

#!/usr/bin/env python3
"""M0b bake extension: HVAC statics, env contract, reward, schedule, exogenous.

Imported and called by bake_scenario.py. Every value is read from the live
scenario; nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np


def _str(w, name: str, items) -> None:
    w.add(name, np.frombuffer("\n".join(str(x) for x in items).encode("utf-8"),
                              dtype=np.uint8), "str")


def _scalar(w, name: str, v: float) -> None:
    w.add(name, np.asarray([float(v)], dtype=np.float64), "f64")


def extend(w, manifest: dict, env, sim, pb, hvac, gin_text: str,
           random_start_days: int = 0) -> None:
    from smart_buildings.smart_control.environment import environment as E

    # ================= HVAC =================
    ahs = hvac._air_handler
    ahus = getattr(ahs, "_ahus", [ahs])
    ahu_ids = [a.device_id() for a in ahus]
    ahu_zones = {}
    if hasattr(ahs, "_map"):
        for a, zones in ahs._map.items():
            ahu_zones[a.device_id()] = list(zones)
    _str(w, "ahu_ids", ahu_ids)
    _str(w, "ahu_zones", [" ".join(ahu_zones.get(i, [])) for i in ahu_ids])
    ahu_const = []
    for a in ahus:
        ahu_const.append([
            a._init_recirculation, a._init_max_air_flow_rate,
            a._min_supply_air_static_pressure, a._max_recirculation,
            a._cooling_cop, a._init_fan_efficiency,
            a._init_cooling_air_temp_setpoint, a._init_fan_static_pressure,
            float(a._init_run_command),
        ])
    w.add("ahu_const", np.asarray(ahu_const, dtype=np.float64), "f64")

    vav_ids = list(hvac._vavs.keys())
    _str(w, "vav_ids", vav_ids)
    vav_const = []
    for vid in vav_ids:
        v = hvac._vavs[vid]
        vav_const.append([
            v._init_max_air_flow_rate, v._reheat_max_water_flow_factor,
            v._min_damper_setting, v._max_discharge_air_temperature,
            v._max_air_flow_static_pressure,
            v._zone_air_heating_temperature_setpoint,
            v._zone_air_cooling_temperature_setpoint,
            v._init_damper_setting, v._init_reheat_valve_setting,
        ])
    w.add("vav_const", np.asarray(vav_const, dtype=np.float64), "f64")
    _str(w, "vav_zone", [hvac._vavs[vid]._zone_id for vid in vav_ids])
    _str(w, "vav_ahu", [hvac._vavs[vid].air_handler.device_id() for vid in vav_ids])
    w.add("vav_ahu_idx", np.asarray(
        [ahu_ids.index(hvac._vavs[vid].air_handler.device_id()) for vid in vav_ids],
        dtype=np.int32), "i32")
    w.add("coupled_vav_idx", np.asarray(
        [vav_ids.index(v) for v in sim._coupled_vavs], dtype=np.int32), "i32")
    z2v = {z: v for z, v in hvac._zone_to_vav_ids.items()}
    _str(w, "zone_ids", list(z2v.keys()))
    _str(w, "zone_vav", [z2v[z][0] for z in z2v])
    manifest["ahu_ids"] = ahu_ids
    manifest["vav_ids"] = vav_ids

    hws = hvac._hot_water_system
    _scalar(w, "hws_header_resistance", hws._header_resistance)
    hs = hws._heat_source
    is_boiler = type(hs).__name__ == "Boiler"
    _scalar(w, "hws_is_boiler", 1.0 if is_boiler else 0.0)
    if is_boiler:
        w.add("boiler_const", np.asarray([
            hs._init_supply_water_temperature_setpoint,
            hs._init_return_water_temperature_sensor,
            hs._heating_rate, hs._cooling_rate, hs._convection_coefficient,
            hs._tank_length, hs._tank_radius, hs._water_capacity,
            hs._insulation_conductivity, hs._insulation_thickness,
        ], dtype=np.float64), "f64")
    else:
        w.add("ashp_const", np.asarray([
            hs._max_capacity_w, hs._nominal_cop, hs._init_return_water_temp,
            hs._supply_water_temperature_setpoint,
        ], dtype=np.float64), "f64")
    p = hws._pump
    w.add("pump_const", np.asarray([
        p._init_water_pump_differential_head, p._init_water_pump_efficiency,
    ], dtype=np.float64), "f64")
    hvac_scalars = [
        hvac._vav_max_air_flow_rate, hvac._vav_reheat_max_water_flow_factor,
        hvac._vav_max_air_flow_static_pressure, hvac._vav_min_damper_setting,
    ]
    w.add("hvac_scalars", np.asarray(hvac_scalars, dtype=np.float64), "f64")

    # ================= schedule =================
    sch = next(iter(hvac._vavs.values())).thermostat.get_setpoint_schedule()
    w.add("sched_scalars", np.asarray([
        sch.morning_start_hour, sch.evening_start_hour,
        sch.comfort_temp_window[0], sch.comfort_temp_window[1],
        sch.eco_temp_window[0], sch.eco_temp_window[1],
    ], dtype=np.float64), "f64")
    _str(w, "sched_tz", [sch._time_zone])
    w.add("sched_holidays", np.asarray(sorted(sch.holidays), dtype=np.int64), "i64")

    # ================= env contract =================
    field_names = list(env.field_names)
    _str(w, "field_names", field_names)
    id_map_inv = env.id_map.inv
    dev_per_field, meas_per_field = [], []
    for fid in field_names:
        if fid in id_map_inv:
            d, m = id_map_inv[fid]
        else:
            d, m = "", fid
        dev_per_field.append(d)
        meas_per_field.append(m)
    _str(w, "field_dev", dev_per_field)
    _str(w, "field_meas", meas_per_field)

    nc = env.observation_normalizer._normalization_constants
    names = list(nc.keys())
    means = [nc[k].sample_mean for k in names]
    variances = [nc[k].sample_variance for k in names]
    _str(w, "norm_names", names)
    w.add("norm_mean", np.asarray(means, dtype=np.float64), "f64")
    w.add("norm_var", np.asarray(variances, dtype=np.float64), "f64")

    aux = list(env._auxiliary_features)
    _str(w, "aux_features", aux)
    w.add("obs_meta", np.asarray([
        env._num_hod_features, env._num_dow_features,
        env.occupancy_normalization_constant, float(env.steps_per_episode),
        float(env.time_step_sec),
    ], dtype=np.float64), "f64")
    _str(w, "time_zone", [env.time_zone])
    _scalar(w, "start_epoch", env.start_timestamp.timestamp())

    action_names = list(env.action_names)
    _str(w, "action_names", action_names)
    is_disc, mn, mx, mnrm, mxrm = [], [], [], [], []
    for an in action_names:
        dev, sp = env.id_map.inv[an]
        a = env.action_normalizers[sp]
        is_disc.append(1 if E.is_discrete_setpoint(an) else 0)
        mn.append(a.setpoint_min)
        mx.append(a.setpoint_max)
        mnrm.append(a.min_normalized_value)
        mxrm.append(a.max_normalized_value)
    w.add("act_is_discrete", np.asarray(is_disc, dtype=np.uint8), "u8")
    w.add("act_min_native", np.asarray(mn, dtype=np.float64), "f64")
    w.add("act_max_native", np.asarray(mx, dtype=np.float64), "f64")
    w.add("act_min_norm", np.asarray(mnrm, dtype=np.float64), "f64")
    w.add("act_max_norm", np.asarray(mxrm, dtype=np.float64), "f64")
    manifest["n_field"] = len(field_names)
    manifest["n_action"] = len(action_names)
    manifest["n_cont"] = int(sum(1 for d in is_disc if d == 0))
    manifest["n_disc"] = int(sum(is_disc))

    # ---- M2: action routing (which device/actuator each action drives) ----
    hws_id = hvac._hot_water_system.device_id()
    ahs_id = ahs.device_id()
    kw, ki, ka = [], [], []
    for an in action_names:
        dev, sp = env.id_map.inv[an]
        if dev == hws_id:
            kw.append(2); ki.append(0)
            ka.append(0 if sp == "supply_water_temperature_setpoint"
                      else 1 if sp == "differential_pressure_setpoint" else 2)
        elif dev == ahs_id:
            idx = int(sp.split("_")[1]) - 1     # 'ahu_1_...' -> 0
            kw.append(0); ki.append(idx)
            ka.append(0 if sp.endswith("supply_air_temperature_setpoint")
                      else 1 if sp.endswith("supply_air_static_pressure_setpoint")
                      else 2 if sp.endswith("recirculation_percentage_command")
                      else 3)
        else:
            kw.append(1); ki.append(vav_ids.index(dev))
            ka.append(0 if sp == "supply_air_damper_percentage_command" else 1)
    w.add("act_owner_kind", np.asarray(kw, dtype=np.uint8), "u8")
    w.add("act_owner_index", np.asarray(ki, dtype=np.int32), "i32")
    w.add("act_actuator", np.asarray(ka, dtype=np.uint8), "u8")

    # ---- M2: vav -> zones + zone areas (physical zones, room order) ----
    phys = [z for z in pb.room_dict
            if z not in ("exterior_space", "interior_wall")]
    zidx = {z: i for i, z in enumerate(phys)}
    voff = [0]; vidx = []
    for vid in vav_ids:
        zs = [z for z in hvac.get_zones_for_vav(vid) if z in zidx]
        vidx.extend(zidx[z] for z in zs)
        voff.append(len(vidx))
    _str(w, "phys_zone_names", phys)
    room_names_all = list(pb.room_dict.keys())
    w.add("phys_room_idx", np.asarray(
        [room_names_all.index(z) for z in phys], dtype=np.int32), "i32")
    w.add("vav_zone_offsets", np.asarray(voff, dtype=np.int32), "i32")
    w.add("vav_zone_idx", np.asarray(vidx, dtype=np.int32), "i32")
    w.add("zone_area", np.asarray([len(pb.room_dict[z]) for z in phys],
                                  dtype=np.float64), "f64")
    zoff = [0]; zv = []
    for z in phys:
        vs = [vav_ids.index(v) for v in hvac._zone_to_vav_ids.get(z, [])]
        zv.extend(vs); zoff.append(len(zv))
    w.add("zone_vav_offsets", np.asarray(zoff, dtype=np.int32), "i32")
    w.add("zone_vav_idx", np.asarray(zv, dtype=np.int32), "i32")
    manifest["n_phys_zones"] = len(phys)

    # ================= M3b: observation layout =================
    import pandas as _pd3
    from smart_buildings.smart_control.utils import regression_building_utils as rbu
    from smart_buildings.smart_control.utils import conversion_utils as cu3
    # The episode remains unchanged. Random-start blobs contain the extra real
    # weather and calendar needed to finish a full episode from every start.
    if random_start_days and int(sim._time_step_sec) != 300:
        raise ValueError("random-start baking currently requires five-minute steps")
    extra = max(0, random_start_days - 1) * 288
    n1 = int(env.steps_per_episode) + extra + 1
    start = sim._start_timestamp
    step = int(sim._time_step_sec)
    AUX = list(env._auxiliary_features)
    dev_fields = [f for f in field_names if f not in set(AUX)]
    AHU_ID = {"cooling_request_count": 0,
              "supply_air_static_pressure_setpoint": 1,
              "supply_air_static_pressure_sensor": 1,
              "supply_air_flowrate_setpoint": 2, "supply_air_flowrate_sensor": 2,
              "supply_air_temperature_sensor": 3, "supply_air_temperature_setpoint": 3,
              "supply_fan_run_command": 4, "exhaust_fan_run_command": 4,
              "supply_fan_speed_percentage_command": 5,
              "discharge_fan_speed_percentage_command": 5,
              "outside_air_flowrate_sensor": 6, "supervisor_run_command": 7,
              "recirculation_percentage_command": 8,
              "outside_air_temperature_sensor": 9}
    VAV_ID = {"reheat_valve_percentage_command": 0,
              "supply_air_damper_percentage_command": 1,
              "supply_air_flowrate_setpoint": 2,
              "zone_air_cooling_temperature_setpoint_sensor": 3,
              "zone_air_heating_temperature_setpoint_sensor": 4,
              "zone_air_temperature_sensor": 5}
    HWS_ID = {"differential_pressure_setpoint": 0, "supply_water_temperature_sensor": 1,
              "differential_pressure_sensor": 2, "supervisor_run_command": 3,
              "heating_request_count": 4, "run_status": 5,
              "supply_water_temperature_setpoint": 6}
    fk, fi, fo = [], [], []
    fm = np.zeros(len(dev_fields)); fv = np.zeros(len(dev_fields))
    for i, f in enumerate(dev_fields):
        dev, meas = env.id_map.inv[f]
        if dev == hws_id:
            fk.append(2); fi.append(0); fo.append(HWS_ID[meas])
        elif dev == ahs_id:
            fk.append(0); fi.append(int(meas.split("_")[1]) - 1)
            fo.append(AHU_ID[meas.split("_", 2)[2]])
        else:
            fk.append(1); fi.append(vav_ids.index(dev)); fo.append(VAV_ID[meas])
        ci = nc.get(meas)
        fm[i] = ci.sample_mean if ci else 0.0
        fv[i] = ci.sample_variance if ci else 1.0
    w.add("f_dev_kind", np.asarray(fk, dtype=np.uint8), "u8")
    w.add("f_dev_idx", np.asarray(fi, dtype=np.int32), "i32")
    w.add("f_obs_id", np.asarray(fo, dtype=np.uint8), "u8")
    w.add("f_norm_mean", fm, "f64")
    w.add("f_norm_var", fv, "f64")
    w.add("n_dev", np.asarray([len(dev_fields)], dtype=np.int64), "i64")
    w.add("n_aux", np.asarray([len(AUX)], dtype=np.int64), "i64")
    nh = int(env._num_hod_features); ndw = int(env._num_dow_features)
    aux_arr = np.zeros((n1, len(AUX) - 1))
    ts5 = start
    for k in range(n1):
        d = {}
        hr = cu3.get_radian_time(
            ts5, cu3.TimeIntervalEnum.HOUR_OF_DAY)
        for nm, v in rbu.expand_time_features(nh, hr, "hod").items():
            d[f"{nm[0]}_{nm[1]}"] = v
        dr = cu3.get_radian_time(
            ts5, cu3.TimeIntervalEnum.DAY_OF_WEEK)
        for nm, v in rbu.expand_time_features(ndw, dr, "dow").items():
            d[f"{nm[0]}_{nm[1]}"] = v
        d["comfort_mode_now"] = 1.0 if sch.is_comfort_mode(ts5) else 0.0
        d["comfort_mode_soon"] = 1.0 if sch.is_comfort_mode(
            ts5 + _pd3.Timedelta(60, unit="minute")) else 0.0
        aux_arr[k] = [d[nm] for nm in AUX[:-1]]
        ts5 = ts5 + _pd3.Timedelta(int(step), unit="s")
    w.add("aux_time", aux_arr, "f64")

    # ================= reward =================
    rf = env.reward_function
    w.add("reward_scalars", np.asarray([
        rf._max_productivity_personhour_usd, rf._min_productivity_personhour_usd,
        rf._productivity_deviation_ref, rf._max_electricity_rate,
        rf._max_natural_gas_rate, rf._productivity_weight,
        rf._energy_cost_weight, rf._carbon_emission_weight,
    ], dtype=np.float64), "f64")
    ee = rf._electricity_energy_cost
    ng = rf._natural_gas_energy_cost
    w.add("elec_weekday_price", np.asarray(ee._weekday_energy_prices, dtype=np.float64), "f64")
    w.add("elec_weekend_price", np.asarray(ee._weekend_energy_prices, dtype=np.float64), "f64")
    w.add("elec_carbon_rate", np.asarray(ee._carbon_emission_rates, dtype=np.float64), "f64")
    w.add("gas_month_price", np.asarray(ng._month_gas_price, dtype=np.float64), "f64")
    _scalar(w, "gas_carbon_rate", ng._carbon_rate)

    # ================= exogenous weather series =================
    wc = sim._weather_controller
    start = sim._start_timestamp
    import pandas as pd
    step = sim._time_step_sec
    n_steps = int(env.steps_per_episode) + extra
    # n_steps+1: the observation after the last step is taken at ts_{n_steps},
    # so amb needs one entry past the episode (the AHU outside-air-temp field).
    amb = np.empty(n_steps + 1, dtype=np.float64)
    conv = np.empty(n_steps, dtype=np.float64)
    ts = start
    for k in range(n_steps + 1):
        amb[k] = wc.get_current_temp(ts)
        if k < n_steps:
            conv[k] = wc.get_air_convection_coefficient(ts)
        ts = ts + pd.Timedelta(int(step), unit="s")
    w.add("weather_ambient", amb, "f64")
    w.add("weather_convection", conv, "f64")

    # per-step schedule + occupancy calendar (calendar logic stays in Python)
    from smart_buildings.smart_control.utils import conversion_utils
    n1 = n_steps + 1
    sc = np.zeros(n1); sh = np.zeros(n1); sco = np.zeros(n1)
    ohour = np.zeros(n1, dtype=np.int64); owork = np.zeros(n1, dtype=np.uint8)
    ts2 = start
    for k in range(n1):
        sc[k] = 1.0 if sch.is_comfort_mode(ts2) else 0.0
        win = sch.get_temperature_window(ts2)
        sh[k] = win[0]; sco[k] = win[1]
        tz = env.building._occupancy._time_zone
        lt = (ts2.tz_convert(tz) if ts2.tz is not None
              else ts2.tz_localize(tz, nonexistent="shift_forward", ambiguous=True))
        ohour[k] = int(lt.hour)
        day = pd.Timestamp(year=int(lt.year), month=int(lt.month), day=int(lt.day))
        owork[k] = 1 if conversion_utils.is_work_day(day) else 0
        ts2 = ts2 + pd.Timedelta(int(step), unit="s")
    w.add("sched_comfort", sc, "f64")
    w.add("sched_heat_sp", sh, "f64")
    w.add("sched_cool_sp", sco, "f64")
    w.add("occ_hour", ohour, "i64")
    w.add("occ_workday", owork, "u8")
    ts_m1 = start - pd.Timedelta(int(step), unit="s")
    tzm = env.building._occupancy._time_zone
    # A random reset must prime occupancy using the preceding interval of its
    # selected date, not the original scenario's preceding Sunday interval.
    previous_count = n1 if random_start_days else 1
    previous_hour = np.empty(previous_count, dtype=np.int64)
    previous_workday = np.empty(previous_count, dtype=np.uint8)
    for k in range(previous_count):
        ltm = (ts_m1.tz_convert(tzm) if ts_m1.tz is not None
               else ts_m1.tz_localize(tzm, nonexistent="shift_forward", ambiguous=True))
        dm = pd.Timestamp(year=int(ltm.year), month=int(ltm.month), day=int(ltm.day))
        previous_hour[k] = int(ltm.hour)
        previous_workday[k] = 1 if conversion_utils.is_work_day(dm) else 0
        ts_m1 += pd.Timedelta(int(step), unit="s")
    w.add("occ_hour_m1", previous_hour, "i64")
    w.add("occ_workday_m1", previous_workday, "u8")
    if random_start_days:
        offsets = np.arange(random_start_days, dtype=np.int64) * 288
        w.add("random_start_offsets", offsets, "i64")
        manifest["random_start"] = {"days": random_start_days,
                                    "offsets_steps": offsets.tolist(),
                                    "table_steps": n_steps}
    # reward timestamps: proto round-trip hour/month/workday at each ts_j
    from smart_buildings.smart_control.utils import conversion_utils as cu
    rh = np.zeros(n1, dtype=np.int64); rm = np.zeros(n1, dtype=np.int64)
    rw = np.zeros(n1, dtype=np.uint8)
    ts3 = start
    for k in range(n1):
        back = cu.proto_to_pandas_timestamp(cu.pandas_to_proto_timestamp(ts3))
        rh[k] = int(back.hour); rm[k] = int(back.month)
        rw[k] = 1 if cu.is_work_day(back) else 0
        ts3 = ts3 + pd.Timedelta(int(step), unit="s")
    w.add("rew_hour", rh, "i64")
    w.add("rew_month", rm, "i64")
    w.add("rew_workday", rw, "u8")
    manifest["weather"] = {"n": n_steps, "amb_min": float(amb.min()),
                           "amb_max": float(amb.max()),
                           "conv": float(conv[0])}

    # ================= occupancy params (ported, not baked) =================
    occ = env.building._occupancy
    w.add("occ_params", np.asarray([
        occ._zone_assignment, occ._earliest_expected_arrival_hour,
        occ._latest_expected_arrival_hour, occ._earliest_expected_departure_hour,
        occ._latest_expected_departure_hour, occ._step_size.total_seconds(),
    ], dtype=np.float64), "f64")
    _str(w, "occ_tz", [occ._time_zone])
    import re as _re
    m = _re.search(r"(?m)^\s*randomized_occupancy/RandomizedArrivalDepartureOccupancy\.seed\s*=\s*([^\n#]+)", gin_text)
    _scalar(w, "occ_seed", int(m.group(1).strip()) if m else 17321)

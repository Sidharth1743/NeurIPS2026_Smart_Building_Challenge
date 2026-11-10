#!/usr/bin/env python3
"""L0 gate (M0a): the baked blob must equal the LIVE scenario exactly.

Re-loads the scenario from scratch (ground truth) and compares every baked
section bit-for-bit, then functionally reconstructs the SuperLU solve from the
baked factorization and checks it against the live factorization.

Run from starter-kit/:
    .venv/bin/python ../complete_port/pysim/check_l0.py \
        /tmp/opencode/baked/building_train_1__fairbanks_1.sbb \
        smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from sbb import BlobReader  # noqa: E402

FAIL = []


def check(cond: bool, what: str) -> None:
    if cond:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAIL.append(what)


def eq(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a, b) or np.allclose(a, b, rtol=0, atol=0)
    return np.array_equal(a, b)


def same_sparse(a, b) -> bool:
    if a.shape != b.shape:
        return False
    d = (a - b).tocsr()
    d.eliminate_zeros()
    return d.nnz == 0


def main() -> None:
    blob_path, cfg = Path(sys.argv[1]), Path(sys.argv[2])
    r = BlobReader(blob_path)
    from smart_buildings import evaluate

    env = evaluate.load_hybrid_environment(cfg)
    env.reset()
    b = env.building
    sim = b.simulator
    pb = sim.building
    solver = sim._solver
    hvac = sim._hvac
    cs = pb.convection_simulator

    print("### geometry")
    check(eq(r.get("grid"), pb.temp.shape), "grid")
    check(eq(r.get("temp_init"), pb.temp.reshape(-1)), "temp_init")
    check(eq(r.get("density"), np.asarray(pb.density).reshape(-1)), "density")
    check(eq(r.get("heat_capacity"), np.asarray(pb.heat_capacity).reshape(-1)), "heat_capacity")
    check(eq(r.get("conductivity"), np.asarray(pb.conductivity).reshape(-1)), "conductivity")
    check(eq(r.get("diffusers"), np.asarray(pb.diffusers).reshape(-1)), "diffusers")
    check(float(r.get("cv_size_cm")[0]) == pb.cv_size_cm, "cv_size_cm")
    check(float(r.get("floor_height_cm")[0]) == pb.floor_height_cm, "floor_height_cm")
    check(float(r.get("initial_temp")[0]) == pb.initial_temp, "initial_temp")
    if pb.interior_mass is not None:
        check(eq(r.get("im_mask"), np.asarray(pb.interior_mass.mask, np.uint8).reshape(-1)),
              "im_mask")
        check(eq(r.get("im_material"),
                 [pb.interior_mass.material.conductivity,
                  pb.interior_mass.material.heat_capacity,
                  pb.interior_mass.material.density]), "im_material")

    print("### rooms")
    names = r.get_str("room_names").split("\n")
    check(names == list(pb.room_dict.keys()), "room_names order")
    off = r.get("room_offsets")
    cells = r.get("room_cells").reshape(-1, 2)
    ok = True
    for i, nm in enumerate(names):
        live = list(pb.room_dict[nm])
        baked = [tuple(x) for x in cells[off[i]:off[i + 1]].tolist()]
        if live != baked:
            ok = False
            break
    check(ok, "room cells per room (order+content)")

    print("### solver scalars")
    check(eq(r.get("solved"), np.asarray(solver._solved)), "solved")
    check(eq(r.get("dirichlet_conductance"), np.asarray(solver._dirichlet_conductance)), "dirichlet_conductance")
    check(eq(r.get("capacitance_rate"), np.asarray(solver._capacitance_rate)), "capacitance_rate")
    check(eq(r.get("mass_coupling"), np.asarray(solver._mass_coupling)), "mass_coupling")
    check(eq(r.get("mass_retained"), np.asarray(solver._mass_retained)), "mass_retained")
    check(eq(r.get("mass_gained"), np.asarray(solver._mass_gained)), "mass_gained")
    check(eq(r.get("mass_temp_init"), np.asarray(solver._mass_temp)), "mass_temp_init")
    check(eq(r.get("horizontal"), np.asarray(solver._horizontal).reshape(-1)), "horizontal")
    check(eq(r.get("vertical"), np.asarray(solver._vertical).reshape(-1)), "vertical")
    check(eq(r.get("solver_conductivity"), np.asarray(solver._conductivity).reshape(-1)), "solver_conductivity")
    check(eq(r.get("ambient_face_areas_0"), np.asarray(solver._ambient_face_areas[0]).reshape(-1)), "ambient_face_areas_0")
    check(eq(r.get("ambient_face_areas_1"), np.asarray(solver._ambient_face_areas[1]).reshape(-1)), "ambient_face_areas_1")
    check(same_sparse(r.get_matrix_csc("A_conduction"), solver._conduction), "A_conduction")

    print("### LU factorization + replay")
    if solver._factorization is None:
        import scipy.sparse.linalg as _spla
        cc_live = float(sim._weather_controller.convection_coefficient)
        solver._factorization = _spla.splu(
            solver.matrix(cc_live), permc_spec="MMD_AT_PLUS_A")
        solver._factorized_at = cc_live
    check(same_sparse(r.get_matrix_csc("lu_L"), solver._factorization.L), "lu_L")
    check(same_sparse(r.get_matrix_csc("lu_U"), solver._factorization.U), "lu_U")
    check(eq(r.get("lu_perm_r"), solver._factorization.perm_r), "lu_perm_r")
    check(eq(r.get("lu_perm_c"), solver._factorization.perm_c), "lu_perm_c")
    import scipy.sparse.linalg as spla
    L = r.get_matrix_csc("lu_L")
    U = r.get_matrix_csc("lu_U")
    ir = np.argsort(r.get("lu_perm_r"))
    b = r.get("solref_b")
    tgt = solver._factorization.solve(b)
    y = spla.spsolve_triangular(L, b[ir], lower=True, unit_diagonal=True)
    z = spla.spsolve_triangular(U, y, lower=False)
    x = z[r.get("lu_perm_c")]
    check(np.allclose(x, tgt, rtol=0, atol=1e-10),
          f"replay matches live solve (max {np.max(np.abs(x-tgt)):.2e})")
    check(np.allclose(r.get("solref_x"), tgt, rtol=0, atol=1e-10), "solref_x matches live solve")

    print("### air-side coupling")
    check(same_sparse(r.get_matrix_csc("applied"), solver._applied), "applied")
    check(same_sparse(r.get_matrix_csr("sensed").tocsc(), solver._sensed.tocsc()), "sensed")
    check(same_sparse(r.get_matrix_csr("air_side_sensed").tocsc(), sim._air_side_sensed.tocsc()),
          "air_side_sensed")
    check(r.get_str("coupled_vavs").split("\n") == list(sim._coupled_vavs), "coupled_vavs")

    print("### convection")
    check(float(r.get("conv_p")[0]) == cs._p, "conv_p")
    check(float(r.get("conv_distance")[0]) == cs._distance, "conv_distance")
    names_c = r.get_str("conv_room_names").split("\n")
    live_rooms = [z for z in pb.room_dict if z not in ("exterior_space", "interior_wall")]
    check(names_c == live_rooms, "conv room order")
    coff = r.get("conv_room_offsets")
    ccells = r.get("conv_room_cells").reshape(-1, 2)
    cand_off = r.get("conv_cand_offsets")
    cand_idx = r.get("conv_cand_idx")
    ok = True
    cbase = 0        # global cell index base (indexes into cand_off)
    cand_base = 0    # global candidate base (values in cand_off)
    for i, rn in enumerate(names_c):
        v = pb.room_dict[rn]
        cvs, ptr, idx = cs._room_arrays(rn, v, cs._distance, cs._distance == -1)
        baked_cells = [tuple(x) for x in ccells[coff[i]:coff[i + 1]].tolist()]
        if baked_cells != [tuple(x) for x in cvs.tolist()]:
            ok = False
            break
        for j in range(len(v)):
            g0 = int(cand_off[cbase + j])
            g1 = int(cand_off[cbase + j + 1])
            if g0 != cand_base + int(ptr[j]) or g1 != cand_base + int(ptr[j + 1]):
                ok = False
                break
            if [int(x) for x in cand_idx[g0:g1].tolist()] != \
               [coff[i] + int(x) for x in idx[ptr[j]:ptr[j + 1]].tolist()]:
                ok = False
                break
        if not ok:
            break
        cbase += len(cvs)
        cand_base += len(idx)
    check(ok, "convection room cells + candidate CSR")

    print("### simulator scalars")
    for attr, key in [("_internal_gain_w_per_m2", "internal_gain_w_per_m2"),
                      ("_internal_gain_standby_fraction", "internal_gain_standby_fraction"),
                      ("_occupant_sensible_heat_w", "occupant_sensible_heat_w")]:
        check(float(r.get(key)[0]) == getattr(sim, attr), key)

    print("### HVAC")
    ahs = hvac._air_handler
    ahus = getattr(ahs, "_ahus", [ahs])
    ahu_ids = [a.device_id() for a in ahus]
    check(r.get_str("ahu_ids").split("\n") == ahu_ids, "ahu_ids")
    live_const = [[a._init_recirculation, a._init_max_air_flow_rate,
                   a._min_supply_air_static_pressure, a._max_recirculation,
                   a._cooling_cop, a._init_fan_efficiency,
                   a._init_cooling_air_temp_setpoint, a._init_fan_static_pressure,
                   float(a._init_run_command)] for a in ahus]
    check(eq(r.get("ahu_const").tolist(), live_const), "ahu_const")
    live_az = []
    for a in ahus:
        live_az.append(" ".join(ahs._map.get(a, [])))
    check(r.get_str("ahu_zones").split("\n") == live_az, "ahu_zones")
    vav_ids = list(hvac._vavs.keys())
    check(r.get_str("vav_ids").split("\n") == vav_ids, "vav_ids")
    live_vc = [[hvac._vavs[v]._init_max_air_flow_rate,
                hvac._vavs[v]._reheat_max_water_flow_factor,
                hvac._vavs[v]._min_damper_setting,
                hvac._vavs[v]._max_discharge_air_temperature,
                hvac._vavs[v]._max_air_flow_static_pressure,
                hvac._vavs[v]._zone_air_heating_temperature_setpoint,
                hvac._vavs[v]._zone_air_cooling_temperature_setpoint,
                hvac._vavs[v]._init_damper_setting,
                hvac._vavs[v]._init_reheat_valve_setting] for v in vav_ids]
    check(eq(r.get("vav_const").tolist(), live_vc), "vav_const")
    check(r.get_str("vav_zone").split("\n") == [hvac._vavs[v]._zone_id for v in vav_ids], "vav_zone")
    check(r.get_str("vav_ahu").split("\n") ==
          [hvac._vavs[v].air_handler.device_id() for v in vav_ids], "vav_ahu")
    check(r.get_str("zone_ids").split("\n") == list(hvac._zone_to_vav_ids.keys()), "zone_ids")
    check(r.get_str("zone_vav").split("\n") ==
          [hvac._zone_to_vav_ids[z][0] for z in hvac._zone_to_vav_ids], "zone_vav")
    hws = hvac._hot_water_system
    check(float(r.get("hws_header_resistance")[0]) == hws._header_resistance, "hws_header_resistance")
    check(float(r.get("hws_is_boiler")[0]) == (1.0 if type(hws._heat_source).__name__ == "Boiler" else 0.0), "hws_is_boiler")
    if type(hws._heat_source).__name__ == "Boiler":
        hs = hws._heat_source
        check(eq(r.get("boiler_const").tolist(), [
            hs._init_supply_water_temperature_setpoint,
            hs._init_return_water_temperature_sensor, hs._heating_rate,
            hs._cooling_rate, hs._convection_coefficient, hs._tank_length,
            hs._tank_radius, hs._water_capacity, hs._insulation_conductivity,
            hs._insulation_thickness]), "boiler_const")
    check(eq(r.get("pump_const").tolist(), [
        hws._pump._init_water_pump_differential_head,
        hws._pump._init_water_pump_efficiency]), "pump_const")
    check(eq(r.get("hvac_scalars").tolist(), [
        hvac._vav_max_air_flow_rate, hvac._vav_reheat_max_water_flow_factor,
        hvac._vav_max_air_flow_static_pressure, hvac._vav_min_damper_setting]),
        "hvac_scalars")

    print("### schedule")
    sch = next(iter(hvac._vavs.values())).thermostat.get_setpoint_schedule()
    check(eq(r.get("sched_scalars").tolist(), [
        sch.morning_start_hour, sch.evening_start_hour,
        sch.comfort_temp_window[0], sch.comfort_temp_window[1],
        sch.eco_temp_window[0], sch.eco_temp_window[1]]), "sched_scalars")
    check(r.get_str("sched_tz") == sch._time_zone, "sched_tz")
    check(eq(r.get("sched_holidays").tolist(), sorted(sch.holidays)), "sched_holidays")

    print("### env contract")
    check(r.get_str("field_names").split("\n") == list(env.field_names), "field_names")
    check(r.get_str("action_names").split("\n") == list(env.action_names), "action_names")
    inv = env.id_map.inv
    check(r.get_str("field_meas").split("\n") ==
          [inv[f][1] if f in inv else f for f in env.field_names], "field_meas")
    check(r.get_str("aux_features").split("\n") == list(env._auxiliary_features), "aux_features")
    check(eq(r.get("obs_meta").tolist(), [env._num_hod_features, env._num_dow_features,
          env.occupancy_normalization_constant, env.steps_per_episode,
          env.time_step_sec]), "obs_meta")
    check(r.get_str("time_zone") == env.time_zone, "time_zone")
    check(abs(float(r.get("start_epoch")[0]) - env.start_timestamp.timestamp()) < 1e-6, "start_epoch")
    from smart_buildings.smart_control.environment import environment as E
    nc = env.observation_normalizer._normalization_constants
    check(r.get_str("norm_names").split("\n") == list(nc.keys()), "norm_names")
    check(eq(r.get("norm_mean").tolist(), [nc[k].sample_mean for k in nc]), "norm_mean")
    check(eq(r.get("norm_var").tolist(), [nc[k].sample_variance for k in nc]), "norm_var")
    disc = r.get("act_is_discrete")
    check(eq(disc.tolist(), [1 if E.is_discrete_setpoint(a) else 0 for a in env.action_names]), "act_is_discrete")
    check(eq(r.get("act_min_native").tolist(),
             [env.action_normalizers[inv[a][1]].setpoint_min for a in env.action_names]), "act_min_native")
    check(eq(r.get("act_max_native").tolist(),
             [env.action_normalizers[inv[a][1]].setpoint_max for a in env.action_names]), "act_max_native")

    print("### M2 routing + vav zones")
    hv = hvac
    hws_id = hv._hot_water_system.device_id()
    ahs_id = hv._air_handler.device_id()
    vav_ids = list(hv._vavs.keys())
    kw, ki, ka = [], [], []
    for an in env.action_names:
        dev, sp = inv[an]
        if dev == hws_id:
            kw.append(2); ki.append(0)
            ka.append(0 if sp == "supply_water_temperature_setpoint"
                      else 1 if sp == "differential_pressure_setpoint" else 2)
        elif dev == ahs_id:
            ki.append(int(sp.split("_")[1]) - 1); kw.append(0)
            ka.append(0 if sp.endswith("supply_air_temperature_setpoint")
                      else 1 if sp.endswith("supply_air_static_pressure_setpoint")
                      else 2 if sp.endswith("recirculation_percentage_command") else 3)
        else:
            kw.append(1); ki.append(vav_ids.index(dev))
            ka.append(0 if sp == "supply_air_damper_percentage_command" else 1)
    check(eq(r.get("act_owner_kind"), kw), "act_owner_kind")
    check(eq(r.get("act_owner_index"), ki), "act_owner_index")
    check(eq(r.get("act_actuator"), ka), "act_actuator")
    phys = [z for z in pb.room_dict
            if z not in ("exterior_space", "interior_wall")]
    check(r.get_str("phys_zone_names").split("\n") == phys, "phys_zone_names")
    check(eq(r.get("zone_area"), [len(pb.room_dict[z]) for z in phys]), "zone_area")
    zidx = {z: i for i, z in enumerate(phys)}
    voff = r.get("vav_zone_offsets")
    vidx = r.get("vav_zone_idx")
    ok = True
    for i, vid in enumerate(vav_ids):
        zs = [z for z in hv.get_zones_for_vav(vid) if z in zidx]
        if [int(x) for x in vidx[voff[i]:voff[i + 1]].tolist()] != [zidx[z] for z in zs]:
            ok = False
            break
    check(ok, "vav -> zones")
    check(eq(r.get("vav_ahu_idx"),
             [ahu_ids.index(hv._vavs[v].air_handler.device_id()) for v in vav_ids]),
          "vav_ahu_idx")
    check(eq(r.get("coupled_vav_idx"), [vav_ids.index(v) for v in sim._coupled_vavs]),
          "coupled_vav_idx")
    room_all = list(pb.room_dict.keys())
    check(eq(r.get("phys_room_idx"), [room_all.index(z) for z in phys]), "phys_room_idx")
    # per-step schedule + occupancy calendar
    import pandas as pd
    from smart_buildings.smart_control.utils import conversion_utils
    sch2 = next(iter(hv._vavs.values())).thermostat.get_setpoint_schedule()
    occ2 = env.building._occupancy
    tz = occ2._time_zone
    n1 = int(r.get("sched_comfort").shape[0])
    sc, sH, sC, oh, ow = [], [], [], [], []
    ts = sim._start_timestamp
    for _ in range(n1):
        sc.append(1.0 if sch2.is_comfort_mode(ts) else 0.0)
        w = sch2.get_temperature_window(ts)
        sH.append(w[0]); sC.append(w[1])
        lt = (ts.tz_convert(tz) if ts.tz is not None
              else ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=True))
        oh.append(int(lt.hour))
        day = pd.Timestamp(year=int(lt.year), month=int(lt.month), day=int(lt.day))
        ow.append(1 if conversion_utils.is_work_day(day) else 0)
        ts = ts + pd.Timedelta(int(sim._time_step_sec), unit="s")
    check(eq(r.get("sched_comfort"), sc), "sched_comfort")
    check(eq(r.get("sched_heat_sp"), sH), "sched_heat_sp")
    check(eq(r.get("sched_cool_sp"), sC), "sched_cool_sp")
    check(eq(r.get("occ_hour"), oh), "occ_hour")
    check(eq(r.get("occ_workday"), ow), "occ_workday")

    print("### reward")
    rf = env.reward_function
    check(eq(r.get("reward_scalars").tolist(), [
        rf._max_productivity_personhour_usd, rf._min_productivity_personhour_usd,
        rf._productivity_deviation_ref, rf._max_electricity_rate,
        rf._max_natural_gas_rate, rf._productivity_weight,
        rf._energy_cost_weight, rf._carbon_emission_weight]), "reward_scalars")
    check(eq(r.get("elec_weekday_price"), rf._electricity_energy_cost._weekday_energy_prices), "elec_weekday_price")
    check(eq(r.get("elec_weekend_price"), rf._electricity_energy_cost._weekend_energy_prices), "elec_weekend_price")
    check(eq(r.get("elec_carbon_rate"), rf._electricity_energy_cost._carbon_emission_rates), "elec_carbon_rate")
    check(eq(r.get("gas_month_price"), rf._natural_gas_energy_cost._month_gas_price), "gas_month_price")
    check(float(r.get("gas_carbon_rate")[0]) == rf._natural_gas_energy_cost._carbon_rate, "gas_carbon_rate")

    print("### exogenous weather")
    import pandas as pd
    wc = sim._weather_controller
    n = int(env.steps_per_episode)
    ts = sim._start_timestamp
    amb = np.array([wc.get_current_temp(ts + pd.Timedelta(int(sim._time_step_sec) * k, unit="s"))
                    for k in range(n)])
    check(eq(r.get("weather_ambient"), amb), "weather_ambient (8640 pts)")
    check(float(r.get("weather_convection")[0]) == wc.convection_coefficient, "weather_convection")

    print("### occupancy params")
    occ = env.building._occupancy
    check(eq(r.get("occ_params").tolist(), [
        occ._zone_assignment, occ._earliest_expected_arrival_hour,
        occ._latest_expected_arrival_hour, occ._earliest_expected_departure_hour,
        occ._latest_expected_departure_hour, occ._step_size.total_seconds()]), "occ_params")
    check(r.get_str("occ_tz") == occ._time_zone, "occ_tz")

    print()
    if FAIL:
        print(f"L0 FAILED: {len(FAIL)} checks")
        for f in FAIL:
            print("   -", f)
        sys.exit(1)
    print("L0 PASSED (all checks)")


if __name__ == "__main__":
    main()

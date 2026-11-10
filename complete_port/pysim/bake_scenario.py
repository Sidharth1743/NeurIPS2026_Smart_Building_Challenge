#!/usr/bin/env python3
"""Bake a scenario's static geometry/solver/convection data into an SBB1 blob.

M0a scope: everything the finite-difference step and the convection step need,
plus the geometry, so the C core can advance temperatures. HVAC / env contract /
exogenous traces are added in later M0 steps.

Nothing here is hardcoded from a building id or a weather seed: every value is
read from a freshly loaded scenario built by the unmodified starter kit.

Run from starter-kit/:
    .venv/bin/python ../complete_port/pysim/bake_scenario.py \
        --out ../complete_port/csim/baked --name building_train_1__fairbanks_1 \
        smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sbb import BlobWriter  # noqa: E402
from bake_env import extend as bake_env_extend  # noqa: E402


def _gin_scalar(text: str, block: str, key: str):
    """Reads the last `block.key = value` from a gin file, or None."""
    rx = re.compile(
        r"(?m)^\s*" + re.escape(block) + r"\." + re.escape(key) + r"\s*=\s*([^\n#]+)"
    )
    hits = rx.findall(text)
    return hits[-1].strip() if hits else None


def bake(config_gin: Path, outdir: Path, name: str, random_start_days: int = 0) -> dict:
    from smart_buildings import evaluate

    env = evaluate.load_hybrid_environment(config_gin)
    env.reset()
    b = env.building
    sim = b.simulator
    pb = sim.building
    solver = sim._solver
    hvac = sim._hvac

    gin_text = Path(config_gin).read_text()

    w = BlobWriter()
    manifest: dict = {"name": name}
    manifest["config_gin"] = str(config_gin)
    manifest["grid"] = list(pb.temp.shape)
    manifest["time_step_sec"] = float(sim._time_step_sec)
    manifest["start_timestamp"] = str(sim._start_timestamp)

    # ---------- geometry ----------
    w.add("grid", np.asarray(pb.temp.shape, dtype=np.int64), "i64")
    w.add("temp_init", pb.temp.reshape(-1), "f64")
    w.add("density", np.asarray(pb.density, dtype=np.float64).reshape(-1), "f64")
    w.add("heat_capacity", np.asarray(pb.heat_capacity, dtype=np.float64).reshape(-1), "f64")
    w.add("conductivity", np.asarray(pb.conductivity, dtype=np.float64).reshape(-1), "f64")
    w.add("diffusers", np.asarray(pb.diffusers, dtype=np.float64).reshape(-1), "f64")
    w.add("cv_size_cm", np.asarray([pb.cv_size_cm], dtype=np.float64), "f64")
    w.add("floor_height_cm", np.asarray([pb.floor_height_cm], dtype=np.float64), "f64")
    w.add("initial_temp", np.asarray([pb.initial_temp], dtype=np.float64), "f64")

    im = pb.interior_mass
    if im is not None:
        w.add("im_mask", np.asarray(im.mask, dtype=np.uint8).reshape(-1), "u8")
        w.add("im_material", np.asarray(
            [im.material.conductivity, im.material.heat_capacity, im.material.density],
            dtype=np.float64), "f64")
        manifest["interior_mass"] = {
            "conductivity": im.material.conductivity,
            "heat_capacity": im.material.heat_capacity,
            "density": im.material.density}

    # ---------- room dict (order preserved) ----------
    room_names = list(pb.room_dict.keys())
    room_offsets = [0]
    flat_cells = []
    for rn in room_names:
        coords = list(pb.room_dict[rn])
        for (r, c) in coords:
            flat_cells.append((r, c))
        room_offsets.append(len(flat_cells))
    w.add("room_names", np.frombuffer(
        ("\n".join(room_names)).encode("utf-8"), dtype=np.uint8), "str")
    w.add("room_offsets", np.asarray(room_offsets, dtype=np.int32), "i32")
    w.add("room_cells", np.asarray(flat_cells, dtype=np.int32).reshape(-1), "i32")
    manifest["room_names"] = room_names
    manifest["room_offsets"] = room_offsets

    # ---------- solver ----------
    w.add("solved", np.asarray(solver._solved, dtype=np.int64), "i64")
    w.add("dirichlet_conductance", np.asarray(solver._dirichlet_conductance, dtype=np.float64), "f64")
    w.add("capacitance_rate", np.asarray(solver._capacitance_rate, dtype=np.float64), "f64")
    w.add("mass_coupling", np.asarray(solver._mass_coupling, dtype=np.float64), "f64")
    w.add("mass_retained", np.asarray(solver._mass_retained, dtype=np.float64), "f64")
    w.add("mass_gained", np.asarray(solver._mass_gained, dtype=np.float64), "f64")
    w.add("mass_temp_init", np.asarray(solver._mass_temp, dtype=np.float64), "f64")
    w.add("horizontal", np.asarray(solver._horizontal, dtype=np.float64).reshape(-1), "f64")
    w.add("vertical", np.asarray(solver._vertical, dtype=np.float64).reshape(-1), "f64")
    w.add("solver_conductivity", np.asarray(solver._conductivity, dtype=np.float64).reshape(-1), "f64")
    aha = solver._ambient_face_areas
    w.add("ambient_face_areas_0", np.asarray(aha[0], dtype=np.float64).reshape(-1), "f64")
    w.add("ambient_face_areas_1", np.asarray(aha[1], dtype=np.float64).reshape(-1), "f64")
    w.add_matrix_csc("A_conduction", solver._conduction)

    # LU factorization from scipy's SuperLU (the same library the Python oracle
    # uses), baked as CSC L/U plus row/column permutations. The C core replays
    # it exactly, with ir = inv(perm_r), ic = inv(perm_c):
    #     bp = b[ir];  y = L\bp;  z = U\y;  x = z[perm_c]
    # Verified to 9e-17 against `lu.solve`. Fill is the MMD-LU fill (~3.1M), so
    # the per-step cost stays small; no C-side ordering or factorization needed.
    cc = float(sim._weather_controller.convection_coefficient)
    w.add("convection_coefficient", np.asarray([cc], dtype=np.float64), "f64")
    import scipy.sparse.linalg as spla
    # The system that is actually factorized (conduction + capacitance/dt
    # diagonal + Dirichlet + mass coupling), baked as its own matrix so a device
    # direct solver (cuDSS) can factorize the *same* system the CPU replays.
    # `A_conduction` above is only the conduction part and is NOT it.
    A_system = solver.matrix(cc)
    w.add_matrix_csc("A_system", A_system)
    lu = spla.splu(A_system, permc_spec="MMD_AT_PLUS_A")
    w.add("lu_perm_r", np.asarray(lu.perm_r, dtype=np.int32), "i32")
    w.add("lu_perm_c", np.asarray(lu.perm_c, dtype=np.int32), "i32")
    w.add_matrix_csc("lu_L", lu.L)
    w.add_matrix_csc("lu_U", lu.U)
    manifest["lu"] = {"nnzL": int(lu.L.nnz), "nnzU": int(lu.U.nnz)}
    # A deterministic reference solve for the C side to diff against.
    n_solved = int(solver._solved.size)
    bvec = np.sin(0.1 * np.arange(n_solved) + 1.0)
    xref = lu.solve(bvec)
    w.add("solref_b", bvec, "f64")
    w.add("solref_x", xref, "f64")
    del lu

    # ---------- air-side coupling ----------
    w.add_matrix_csc("applied", solver._applied)
    w.add_matrix_csr("sensed", solver._sensed)
    # simulator's own copy (n_cvs x k), used to recompute air_side_q post-step
    w.add_matrix_csr("air_side_sensed", sim._air_side_sensed)
    coupled = list(sim._coupled_vavs)
    w.add("coupled_vavs", np.frombuffer(
        ("\n".join(coupled)).encode("utf-8"), dtype=np.uint8), "str")
    manifest["coupled_vavs"] = coupled

    # ---------- convection ----------
    cs = pb.convection_simulator
    w.add("conv_p", np.asarray([cs._p], dtype=np.float64), "f64")
    w.add("conv_distance", np.asarray([cs._distance], dtype=np.float64), "f64")
    seed_txt = _gin_scalar(gin_text, "StochasticConvectionSimulator", "seed")
    w.add("conv_seed", np.asarray([int(seed_txt) if seed_txt else -1], dtype=np.int64), "i64")
    manifest["conv"] = {"p": cs._p, "distance": cs._distance, "seed": seed_txt}
    phys_rooms = [r for r in room_names
                  if r not in ("exterior_space", "interior_wall")]
    conv_off = [0]           # per-room start in conv_room_cells
    conv_cells = []
    cand_off = []            # global CSR indptr over conv_room_cells
    cand_idx = []            # global candidate cell positions
    for rn in phys_rooms:
        v = pb.room_dict[rn]
        cvs, ptr, idx = cs._room_arrays(rn, v, cs._distance, cs._distance == -1)
        base = len(conv_cells)                # this room's first cell in conv_cells
        for (r, c) in cvs.tolist():
            conv_cells.append((r, c))
        conv_off.append(len(conv_cells))
        cand_base = len(cand_idx)             # this room's first candidate slot
        cand_idx.extend(base + int(x) for x in idx.tolist())
        n = int(cvs.shape[0])
        cand_off.extend(cand_base + int(ptr[i]) for i in range(n))
    cand_off.append(len(cand_idx))
    assert len(cand_off) == len(conv_cells) + 1, (len(cand_off), len(conv_cells))
    w.add("conv_room_names", np.frombuffer(
        ("\n".join(phys_rooms)).encode("utf-8"), dtype=np.uint8), "str")
    w.add("conv_room_offsets", np.asarray(conv_off, dtype=np.int32), "i32")
    w.add("conv_room_cells", np.asarray(conv_cells, dtype=np.int32).reshape(-1), "i32")
    w.add("conv_cand_offsets", np.asarray(cand_off, dtype=np.int32), "i32")
    w.add("conv_cand_idx", np.asarray(cand_idx, dtype=np.int32), "i32")
    manifest["conv_phys_rooms"] = phys_rooms

    # ---------- simulator scalars ----------
    for attr in ["_internal_gain_w_per_m2", "_internal_gain_standby_fraction",
                 "_occupant_sensible_heat_w"]:
        w.add(attr.lstrip("_"), np.asarray([getattr(sim, attr)], dtype=np.float64), "f64")
        manifest[attr.lstrip("_")] = getattr(sim, attr)

    # ---------- HVAC + env contract + reward + schedule + exogenous ----------
    bake_env_extend(w, manifest, env, sim, pb, hvac, gin_text, random_start_days)

    outdir.mkdir(parents=True, exist_ok=True)
    blob = outdir / f"{name}.sbb"
    size = w.write(blob)
    manifest["blob"] = blob.name
    manifest["blob_bytes"] = size
    (outdir / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(f"baked {name}: {size/1e6:.1f} MB, {len(w._sections)} sections")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_gin", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--random-start-days", type=int, choices=[0, 7], default=0,
                    help="bake seven 06:00 weekday starts with continuous weather")
    args = ap.parse_args()
    bake(args.config_gin, args.out, args.name, args.random_start_days)


if __name__ == "__main__":
    main()

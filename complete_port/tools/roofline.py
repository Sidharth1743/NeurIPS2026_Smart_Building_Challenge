#!/usr/bin/env python3
"""Offline cost model of an SBB1 scenario blob: the solve's terms, the bytes its state loads move,
and the hardware floors those imply.  CPU only, reads the section directory of the blob.

  tools/roofline.py [--dirs "artifacts/train artifacts/sets/diverseB35"] [--envs-per-weather 32]
                    [--json FILE] [--md FILE]

The GPU step's cost is set by the triangular solve (docs/HISTORY.md, "Measured cost"): per env-step it
visits `nnz(L)+nnz(U)` terms, and each term loads the state vector once -- one value per env, so
`8*(nnzL+nnzU)` bytes per env-step in fp64 (half in fp32), duplicated over the 32-lane slabs it reads
(ld = 32*ceil(envs/32)).  The FP64 pipe needs `(nnzL+nnzU)/278e3` us per env-step on an RTX 3090.
Both floors are printed; the step cannot beat either.

Blob layout: see pysim/sbb.py / csim/sb_baked.h.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

HDR = struct.Struct("<4sII5Q")          # magic ver n_sections dir_off names_off names_len total_size rsv
SEC = struct.Struct("<IIII5Q")          # name_off dtype ndim reserved shape[4] data_off
DTYPE_BYTES = {0: 8, 1: 4, 2: 8, 3: 4, 4: 1, 5: 1, 6: 4, 7: 1}

# Hardware floors (HISTORY.md, "Measured cost"): nominal rates.
FMA_FP64 = {  # G FMA/s
    "3090": 278.0,
    "3060": 100.0,
}
BANDWIDTH = {  # GB/s
    "3090": 936.0,
    "3060": 360.0,
}


def blob_sections(path: os.PathLike) -> dict[str, tuple[int, tuple[int, ...], int, int]]:
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    magic, _ver, n_sec, dir_off, names_off, names_len, total, _rsv = HDR.unpack_from(head, 0)
    if magic != b"SBB1":
        raise ValueError(f"{path}: bad magic {magic!r}")
    out: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
    for i in range(n_sec):
        name_off, dtype, ndim, _rsv, *rest = SEC.unpack_from(head, dir_off + i * SEC.size)
        shape = tuple(rest[:ndim])
        data_off = rest[4]
        end = head.index(b"\0", names_off + name_off)
        name = head[names_off + name_off : end].decode()
        count = 1
        for s in shape:
            count *= s
        out[name] = (dtype, shape, count, data_off)
    out["__size__"] = (0, (), total, 0)
    return out


def count_of(sec: dict, name: str) -> int:
    return sec[name][2] if name in sec else 0


def scenario_cost(path: os.PathLike) -> dict:
    sec = blob_sections(path)
    n = count_of(sec, "lu_perm_c")               # unknowns of the linear system
    ncv = count_of(sec, "solved")               # control volumes of the field
    nnz_l = count_of(sec, "lu_L_data")
    nnz_u = count_of(sec, "lu_U_data")
    nnz_z = count_of(sec, "applied_data")        # air-side correction (Z = L^-1 applied)
    nnz_r = count_of(sec, "air_side_sensed_data")
    k = count_of(sec, "coupled_vav_idx")
    zones = count_of(sec, "zone_ids")
    terms = nnz_l + nnz_u + nnz_z
    name = Path(path).stem
    return {
        "name": name,
        "blob_mb": sec["__size__"][2] / 1e6,
        "n": n,
        "ncv": ncv,
        "zones": zones,
        "k": k,
        "nnzL": nnz_l,
        "nnzU": nnz_u,
        "nnzZ": nnz_z,
        "nnzR": nnz_r,
        "terms": terms,
        "state_mb_f64": 8 * terms / 1e6,
        "state_mb_f32": 4 * terms / 1e6,
        "fp64_us": terms / (FMA_FP64["3090"] * 1e3),
        "dram_us_3090": 8 * terms / (BANDWIDTH["3090"] * 1e3),
        "dram_us_3060": 8 * terms / (BANDWIDTH["3060"] * 1e3),
    }


def fmt_row(c: dict) -> str:
    return (
        f"| {c['name']:36s} | {c['n']/1e3:8.0f}K | {c['zones']:4d} | {c['terms']/1e6:6.2f}M |"
        f" {c['blob_mb']:7.0f} | {c['state_mb_f64']:6.1f} | {c['fp64_us']:6.1f} |"
        f" {c['dram_us_3090']:6.1f} | {c['dram_us_3060']:6.1f} |"
    )


HEADER = (
    "| scenario | unknowns | zones | solve terms | blob MB | state MB/env-step | FP64 us/env-step | DRAM us 3090 | DRAM us 3060 |\n"
    "|---|--:|--:|--:|--:|--:|--:|--:|--:|"
)


def collect(dirs: list[str], blobs: list[str]) -> list[dict]:
    paths: list[Path] = []
    for d in dirs:
        paths += sorted(Path(d).glob("*.sbb"))
    paths += [Path(b) for b in blobs]
    rows = []
    seen = set()
    for p in paths:
        if not p.exists():
            continue
        c = scenario_cost(p)
        # one weather per blob; de-duplicate by building so aggregates are per building
        if p.stem not in seen:
            seen.add(p.stem)
            c["path"] = str(p)
            c["building"] = p.stem.split("__")[0]
            rows.append(c)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="artifacts/train artifacts/sets/diverseB35 artifacts/sets/diverseC35 artifacts/sets/gen-train-100")
    ap.add_argument("--blobs", default="")
    ap.add_argument("--envs-per-weather", type=int, default=32)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    rows = collect(a.dirs.split(), a.blobs.split())
    if not rows:
        print("no blobs found")
        return 1

    # one row per building (dedupe weathers: keep the first blob of each building)
    buildings: dict[str, dict] = {}
    for c in rows:
        buildings.setdefault(c["building"], c)
    bl = list(buildings.values())

    lines = ["# Solver cost model (offline, from the baked blobs)", "",
             f"generated by `tools/roofline.py`; per-env-step = the triangular solve's terms "
             f"(nnzL+nnzU+nnzZ), each term loading the state once per env.", "",
             HEADER]
    lines += [fmt_row(c) for c in sorted(bl, key=lambda c: c["terms"])]
    mean_terms = sum(c["terms"] for c in bl) / len(bl)
    mean_mb64 = sum(c["state_mb_f64"] for c in bl) / len(bl)
    lines += ["", f"{len(bl)} buildings; mean solve terms {mean_terms/1e6:.2f}M/env-step, "
                  f"mean state {mean_mb64:.1f} MB/env-step (fp64).", "",
              "Floors per env-step: FP64 pipe = terms/278e3 us (RTX 3090); DRAM = state bytes/936 GB/s "
              "(3090) / 360 GB/s (3060).  The whole step is >= max(pipe, DRAM) plus the non-solve phases.", "",
              "Predicted aggregate SPS (all envs of the batch stepping together, bandwidth-bound):",
              "",
              "| batch | envs | mean MB/env-step | 3090 SPS | 3090+3060 SPS | fp32 3090 SPS |",
              "|---|--:|--:|--:|--:|--:|"]
    for d in a.dirs.split() + ([b for b in a.blobs.split()] if a.blobs else []):
        sel = [c for c in rows if c["path"].startswith(str(Path(d).resolve())) or Path(c["path"]).parent == Path(d)]
        if not sel:
            continue
        sel_b = {}
        for c in sel:
            sel_b.setdefault(c["building"], c)
        vals = list(sel_b.values())
        envs = a.envs_per_weather * 5 * len(vals)
        meanb = sum(c["state_mb_f64"] for c in vals) / len(vals)
        lines.append(f"| {Path(d).name} | {envs} | {meanb:.1f} | "
                     f"{BANDWIDTH['3090']*1e3/meanb:,.0f} | "
                     f"{(BANDWIDTH['3090']+BANDWIDTH['3060'])*1e3/meanb:,.0f} | "
                     f"{BANDWIDTH['3090']*2e3/meanb:,.0f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1))
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

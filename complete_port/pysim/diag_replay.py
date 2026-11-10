#!/usr/bin/env python3
"""Feed a verified trace's actions through the bridge and compare obs/reward."""
import os
import ctypes, os, struct, sys
from pathlib import Path
import numpy as np
SK = SB_ROOT / "starter-kit"; os.chdir(SK)
sys.path.insert(0, str(SB_ROOT / "complete_port/pysim"))
import sweep_scores as S
SB_ROOT = Path(os.environ.get("SB_ROOT", next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents]
     if (p / 'complete_port').is_dir()), Path(__file__).resolve().parent)))


building, weather, trace, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
f = open(trace, "rb")
h = struct.unpack("<8i", f.read(32))
nx, nz, nc, nd, nsteps, kk, nfld, nprobe = h
na = nc + nd
rec = (nc + nd + na + nz + 2 * kk + 1 + 1 + nfld + nprobe) * 8
obs0 = np.frombuffer(f.read(8 * nfld), dtype="<f8")
f.read(4 * nprobe)
print("trace", h)

hnd = S.LIB.sb_load(str(S.BLOBS / f"{building}__{weather}.sbb").encode())
inf = S.info(hnd)
obs = np.zeros(inf["n_obs"], dtype=np.float32)
cont = np.zeros(nc, dtype=np.float32)
disc = np.zeros(nd, dtype=np.int32)
rew = ctypes.c_double(0.0)
S.LIB.sb_reset(hnd, obs.ctypes.data_as(S.F))
w = abs(float(obs0[0]) - float(obs[0]))
for i in range(nfld):
    w = max(w, abs(float(obs0[i]) - float(obs[i])))
print(f"reset obs max|d| = {w:.3e}")
for k in range(min(steps, nsteps)):
    buf = f.read(rec)
    o = 0
    c = np.frombuffer(buf[o:o + 8 * nc], dtype="<f8"); o += 8 * nc
    d = np.frombuffer(buf[o:o + 8 * nd], dtype="<f8"); o += 8 * nd
    o += 8 * na + 8 * nz + 2 * 8 * kk + 8              # nat, occ, g, T, swt
    ro = float(np.frombuffer(buf[o:o + 8], dtype="<f8")[0]); o += 8
    oo = np.frombuffer(buf[o:o + 8 * nfld], dtype="<f8"); o += 8 * nfld
    cont[:] = c
    disc[:] = d
    S.LIB.sb_step(hnd, cont.ctypes.data_as(S.F), disc.ctypes.data_as(S.I),
                  obs.ctypes.data_as(S.F), ctypes.byref(rew))
    do = float(np.max(np.abs(oo - np.asarray(obs, dtype=np.float64))))
    dr = abs(ro - rew.value)
    if k < 3 or do > 0 or dr > 0:
        j = int(np.argmax(np.abs(oo - np.asarray(obs, np.float64))))
        print(f"  k={k:3d} obs d={do:.3e} (idx {j}: trace={oo[j]:.6f} bridge={obs[j]:.6f})  "
              f"rew trace={ro:+.9f} bridge={rew.value:+.9f}")
    if do > 0 or dr > 0:
        print("  -> divergence"); break
print("done")

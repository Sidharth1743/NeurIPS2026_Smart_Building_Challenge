#!/usr/bin/env python3
"""Lockstep diagnostic: oracle env vs the C bridge, same policy, per-step."""
import os
import ctypes, os, sys
from pathlib import Path
import numpy as np

SK = SB_ROOT / "starter-kit"
os.chdir(SK)
sys.path.insert(0, str(SB_ROOT / "complete_port/pysim"))
import sweep_scores as S

b, w, steps, pkey = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = SK / "smart_buildings" / "training_scenarios" / b / w / "config.gin"
from smart_buildings import evaluate
SB_ROOT = Path(os.environ.get("SB_ROOT", next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents]
     if (p / 'complete_port').is_dir()), Path(__file__).resolve().parent)))

env = evaluate.load_hybrid_environment(cfg)
meta = evaluate.create_building_meta_data(env, cfg.parent)
po = evaluate.initialize_policy(evaluate.POLICY_MAP[pkey], meta, env.action_spec())
pb_ = S.make_policy(pkey, b, w, Path("/tmp/opencode/sb_sweep"))

h = S.LIB.sb_load(str(S.BLOBS / f"{b}__{w}.sbb").encode())
inf = S.info(h)
print("info", inf)
obs_b = np.zeros(inf["n_obs"], dtype=np.float32)
cont = np.zeros(inf["n_cont"], dtype=np.float32)
disc = np.zeros(inf["n_disc"], dtype=np.int32)
rew = ctypes.c_double(0.0)
S.LIB.sb_reset(h, obs_b.ctypes.data_as(S.F))

obs_o, _ = env.reset()
obs_o = np.asarray(obs_o, dtype=np.float32)
d0 = float(np.max(np.abs(obs_o - obs_b)))
print(f"reset obs max|d| = {d0:.3e}")
for k in range(steps):
    ao = evaluate.validate_action(evaluate.get_policy_action(po, obs_o), env.action_spec())
    ab = evaluate.get_policy_action(pb_, obs_b)
    ca = np.asarray(ao["continuous_action"], dtype=np.float32)
    cb = np.asarray(ab["continuous_action"], dtype=np.float32)
    da = np.asarray(ao["discrete_action"], dtype=np.int32)
    db = np.asarray(ab["discrete_action"], dtype=np.int32)
    dca = float(np.max(np.abs(ca - cb))) if ca.size else 0.0
    dda = int(np.max(np.abs(da - db))) if da.size else 0
    res = env.step(ao)
    obs_o = np.asarray(res[0], dtype=np.float32)
    ro = float(res[1])
    cont[:] = cb
    disc[:] = db
    S.LIB.sb_step(h, cont.ctypes.data_as(S.F), disc.ctypes.data_as(S.I),
                  obs_b.ctypes.data_as(S.F), ctypes.byref(rew))
    do = float(np.max(np.abs(obs_o - obs_b)))
    if k < 5 or do > 0 or dca > 0 or dda > 0 or ro != rew.value:
        print(f"  k={k:3d} act: dcont={dca:.2e} ddisc={dda}  obs d={do:.3e}  "
              f"rew oracle={ro:+.9f} bridge={rew.value:+.9f}")
    if do > 1e-7:
        j = int(np.argmax(np.abs(obs_o - obs_b)))
        print(f"     first differing obs idx={j}: oracle={obs_o[j]!r} "
              f"bridge={obs_b[j]!r}")
    if dca > 0 or dda > 0:
        print("  -> first ACTION divergence"); break
    if do > 1e-7:
        print("  -> first OBS divergence"); break
print("done")

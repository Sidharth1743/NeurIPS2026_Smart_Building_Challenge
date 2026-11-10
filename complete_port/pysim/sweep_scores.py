#!/usr/bin/env python3
"""Score sweep: shipped baselines x scenarios, driven through the verified C core
via the ctypes bridge, compared against reference_baselines.json.

The Python side supplies the REAL policy objects (no policy port, so no porting
risk); the C core supplies the simulation. Step order mirrors the PufferLib env.

  # bridge sanity: same policy, same steps, bridge total vs the Python oracle
  sweep_scores.py --pycheck <building> <weather> [steps] [policy]

  # sweep (resumable; writes JSON per task under --cache)
  sweep_scores.py --policies bang_bang,pid --scenarios 3 --workers 3 [--steps 8640]
"""
from __future__ import annotations

ROOT = Path(os.environ.get("SB_ROOT", Path(__file__).resolve().parents[2]))

import argparse, ctypes, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = ctypes.CDLL(str(HERE.parent / "csim" / "libsb_bridge.so"))
F = ctypes.POINTER(ctypes.c_float)
I = ctypes.POINTER(ctypes.c_int)
D = ctypes.POINTER(ctypes.c_double)
LIB.sb_load.restype = ctypes.c_void_p
LIB.sb_load.argtypes = [ctypes.c_char_p]
LIB.sb_free.argtypes = [ctypes.c_void_p]
LIB.sb_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.sb_action_dims.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]
LIB.sb_reset.argtypes = [ctypes.c_void_p, F]
LIB.sb_step.argtypes = [ctypes.c_void_p, F, I, F, D]

SK = ROOT / "starter-kit"
# The kit's gin configs reference scenario paths relative to its own root,
# so every env load must happen with that as the working directory.
os.chdir(SK)
BLOBS = HERE.parent / "csim" / "baked"


def info(h):
    out = (ctypes.c_int * 8)()
    LIB.sb_info(h, out, 8)
    nc = ctypes.c_int()
    nd = ctypes.c_int()
    LIB.sb_action_dims(h, ctypes.byref(nc), ctypes.byref(nd))
    return dict(n_act=out[0], n_obs=out[1], n_steps=out[2], n_ahu=out[3],
                n_vav=out[4], n_zones=out[5], n_cvs=out[6], k=out[7],
                n_cont=nc.value, n_disc=nd.value)


def scenarios():
    import yaml
    with open(SK / "smart_buildings" / "training_set.yaml") as f:
        return [(s["building"], s["weather"]) for s in yaml.safe_load(f)["scenarios"]]


def blob_fresh(blob: Path) -> bool:
    """A blob must carry the observation sections (M3b) or the bridge cannot
    produce an observation at all; blobs baked before that have none. Parsed
    with the authoritative reader (hand-rolled header offsets got this wrong)."""
    try:
        from sbb import BlobReader
        b = BlobReader(str(blob))
        return b.has("n_dev") and b.has("aux_time") and b.has("f_dev_kind")
    except Exception:
        return False


def meta_for(b, w, cdir: Path):
    """Policy meta data is derived from a loaded Python env; cache it, since
    loading the Python simulator is the slow part (the C side needs none of it)."""
    cdir.mkdir(parents=True, exist_ok=True)
    cp = cdir / f"{b}_{w}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    from smart_buildings import evaluate
    cfg = SK / "smart_buildings" / "training_scenarios" / b / w / "config.gin"
    env = evaluate.load_hybrid_environment(cfg)
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    with open(cp, "wb") as f:
        pickle.dump(meta, f)
    return meta


def make_policy(pkey, b, w, cdir: Path):
    from smart_buildings import evaluate
    meta = meta_for(b, w, cdir)
    spec = {"continuous_action": {"shape": (meta["n_continuous_action"],)},
            "discrete_action": {"shape": (meta["n_discrete_action"],)}}
    return evaluate.initialize_policy(evaluate.POLICY_MAP[pkey], meta, spec)


def run_bridge(blob, pkey, b, w, cdir, steps):
    from smart_buildings import evaluate
    policy = make_policy(pkey, b, w, cdir)
    h = LIB.sb_load(str(blob).encode())
    if not h:
        raise RuntimeError(f"sb_load failed for {blob}")
    try:
        inf = info(h)
        n_obs, nc, nd = inf["n_obs"], inf["n_cont"], inf["n_disc"]
        obs = np.zeros(n_obs, dtype=np.float32)
        cont = np.zeros(nc, dtype=np.float32)
        disc = np.zeros(nd, dtype=np.int32)
        obs_p = obs.ctypes.data_as(F)
        cont_p = cont.ctypes.data_as(F)
        disc_p = disc.ctypes.data_as(I)
        rew = ctypes.c_double(0.0)
        LIB.sb_reset(h, obs_p)
        total = 0.0
        n = min(steps, inf["n_steps"] or steps)
        for _ in range(n):
            act = evaluate.get_policy_action(policy, obs)
            c = np.asarray(act["continuous_action"], dtype=np.float32)
            d = np.asarray(act["discrete_action"], dtype=np.int32)
            cont[:] = c
            disc[:] = d
            LIB.sb_step(h, cont_p, disc_p, obs_p, ctypes.byref(rew))
            total += rew.value
        return total, n
    finally:
        LIB.sb_free(h)


def run_oracle(b, w, pkey, steps):
    from smart_buildings import evaluate
    cfg = SK / "smart_buildings" / "training_scenarios" / b / w / "config.gin"
    env = evaluate.load_hybrid_environment(cfg)
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    policy = evaluate.initialize_policy(evaluate.POLICY_MAP[pkey], meta,
                                        env.action_spec())
    obs, _ = env.reset()
    total = 0.0
    for _ in range(steps):
        act = evaluate.get_policy_action(policy, obs)
        act = evaluate.validate_action(act, env.action_spec())
        res = env.step(act)
        obs = res[0]
        total += float(res[1])
    return total


def one_task(args):
    b, w, pkey, steps, cdir, blob = args
    t0 = time.time()
    total, n = run_bridge(Path(blob), pkey, b, w, Path(cdir), steps)
    return dict(building=b, weather=w, policy=pkey, total=total, steps=n,
                mean_per_step=total / n, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="bang_bang,pid")
    ap.add_argument("--scenarios", type=int, default=0, help="0 = all 30")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--cache", default="/tmp/opencode/sb_sweep")
    ap.add_argument("--out", default="/tmp/opencode/sb_sweep/scores.json")
    ap.add_argument("--pycheck", nargs=2, metavar=("BUILDING", "WEATHER"))
    ap.add_argument("--pycheck-policy", default="bang_bang")
    ap.add_argument("--pycheck-steps", type=int, default=40)
    a = ap.parse_args()
    cdir = Path(a.cache)
    cdir.mkdir(parents=True, exist_ok=True)

    if a.pycheck:
        b, w = a.pycheck
        blob = BLOBS / f"{b}__{w}.sbb"
        if not blob.exists():
            print(f"missing blob {blob}")
            return 1
        t0 = time.time()
        tot_b, n = run_bridge(blob, a.pycheck_policy, b, w, cdir, a.pycheck_steps)
        tb = time.time() - t0
        t0 = time.time()
        tot_o = run_oracle(b, w, a.pycheck_policy, a.pycheck_steps)
        to = time.time() - t0
        print(f"policy={a.pycheck_policy} steps={n}")
        print(f"  bridge total = {tot_b!r}  ({tb:.1f}s)")
        print(f"  oracle total = {tot_o!r}  ({to:.1f}s)")
        d = abs(tot_b - tot_o)
        print(f"  |delta| = {d:.3e}   oracle {to/tb:.1f}x slower than bridge")
        print("BRIDGE CHECK: " + ("OK" if d < 1e-6 else "FAIL"))
        return 0 if d < 1e-6 else 1

    polys = [p for p in a.policies.split(",") if p]
    sc = scenarios()
    if a.scenarios:
        sc = sc[:a.scenarios]
    tasks = []
    for b, w in sc:
        blob = BLOBS / f"{b}__{w}.sbb"
        if not blob.exists():
            print(f"  (skip: no blob for {b}/{w})")
            continue
        if not blob_fresh(blob):
            print(f"  (skip: stale blob for {b}/{w} -- re-bake it)")
            continue
        for p in polys:
            tasks.append((b, w, p, a.steps, str(cdir), str(blob)))
    print(f"sweep: {len(tasks)} tasks, {a.workers} workers, {a.steps} steps")
    results = []
    if a.workers <= 1:
        for t in tasks:
            r = one_task(t)
            print(f"  {r['building']}/{r['weather']} {r['policy']}: "
                  f"mean={r['mean_per_step']:+.6f} ({r['seconds']:.0f}s)", flush=True)
            results.append(r)
    else:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for r in pool.imap_unordered(one_task, tasks):
                print(f"  {r['building']}/{r['weather']} {r['policy']}: "
                      f"mean={r['mean_per_step']:+.6f} ({r['seconds']:.0f}s)", flush=True)
                results.append(r)

    out = {"results": results}
    ref = ROOT / "reference_baselines.json"
    if ref.exists():
        rj = json.load(open(ref))
        out["reference"] = rj.get("scores")
        out["verification"] = rj.get("verification")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    ref_runs = {}
    if ref.exists():
        for r in rj.get("runs", []):
            ref_runs[(r["scenario"], r["policy"])] = r["total_reward"]

    print("\n-- per-run vs reference (tol rel 1e-3 / abs 1e-3) --")
    worst_abs = worst_rel = 0.0
    ncmp = nfail = 0
    for r in sorted(results, key=lambda x: (x["building"], x["weather"], x["policy"])):
        key = (f"{r['building']}/{r['weather']}", r["policy"])
        rv = ref_runs.get(key)
        if rv is None:
            continue
        a_ = abs(r["total"] - rv)
        rel = a_ / abs(rv) if rv else 0.0
        worst_abs = max(worst_abs, a_)
        worst_rel = max(worst_rel, rel)
        ncmp += 1
        if not (rel <= 1e-3 or a_ <= 1e-3):
            nfail += 1
            print(f"  FAIL {key}: mine={r['total']:.6f} ref={rv:.6f} "
                  f"abs={a_:.3e} rel={rel:.3e}")
    print(f"  compared {ncmp} runs | worst abs={worst_abs:.4e} worst rel={worst_rel:.4e} "
          f"| failures={nfail}")

    print("\n-- aggregate --")
    for p in polys:
        rr = [r for r in results if r["policy"] == p]
        if not rr:
            continue
        score = float(np.mean([abs(r["mean_per_step"]) for r in rr]))
        line = f"  {p:16s} n={len(rr):2d}  score={score:.5f}"
        if ref.exists() and rj.get("scores", {}).get(p) is not None:
            rv = rj["scores"][p]
            rv = rv["score"] if isinstance(rv, dict) else rv
            line += f"  reference={float(rv):.5f}  delta={score-float(rv):+.2e}"
        print(line)
    print(f"wrote {a.out}")
    ok = (ncmp > 0 and nfail == 0)
    print("SCORE SWEEP: " + ("PASS" if ok else ("INCOMPLETE" if ncmp == 0 else "FAIL")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

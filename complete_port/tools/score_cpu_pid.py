"""Official `pid_water` (the competition baseline) on a scenario list, one month each, parallel.

The policy and the simulator are the starter-kit's own code, driven the same way `evaluate.py` drives
them (`load_hybrid_environment` -> `create_building_meta_data` -> `policy.get_action` -> `SimulationTracker.step`),
so the number is the same quantity the leaderboard uses: mean per-step reward over one 8,640-step month,
reported as score = -mean. Restart-safe: a scenario whose summary.json exists is skipped.

    score_cpu_pid.py <scenario-dir-list-file> <out-dir> [workers] [steps]

Each line of the list file is a scenario directory (it must contain config.gin).
Writes <out>/<building>__<weather>/summary.json and reward_components.csv.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("SB_ROOT", Path(__file__).resolve().parents[2]))
SB = REPO / 'starter-kit'
sys.path.insert(0, str(SB))

STEPS = 8640


def run_one(args):
    scen = Path(args[0]); out = Path(args[2]); steps = int(args[3])
    name = f'{scen.parent.name}__{scen.name}'
    dest = out / name
    if (dest / 'summary.json').is_file():
        return f'skip {name}'
    dest.mkdir(parents=True, exist_ok=True)

    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy  # noqa: PLC0415
    from smart_buildings.evaluate import (  # noqa: PLC0415
        SimulationTracker, create_building_meta_data, load_hybrid_environment, validate_action,
    )

    env = load_hybrid_environment(scen / 'config.gin')
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, scen)
    spec = env.action_spec()
    policy = PIDWaterResetPolicy(meta, spec)

    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    tracker.reset()
    obs = tracker.last_obs
    t0 = time.time()
    n = 0
    for _ in range(steps):
        action = validate_action(policy.get_action(obs), env.action_spec())
        obs, _, term, trunc, _ = tracker.step(action)
        n += 1
        if term or trunc:
            break
    elapsed = time.time() - t0

    rc = tracker.reward_components
    import numpy as np  # noqa: PLC0415
    prod = float(np.mean([r['normalized_productivity_regret'] for r in rc]))
    ener = float(np.mean([r['normalized_energy_cost'] for r in rc]))
    carb = float(np.mean([r['normalized_carbon_emission'] for r in rc]))
    elec = float(np.sum([r['electricity_energy_cost'] for r in rc]))
    gas = float(np.sum([r['natural_gas_energy_cost'] for r in rc]))
    summary = {
        'scenario': name, 'policy': 'pid_water', 'steps': n, 'seconds': elapsed,
        'score': -0.6 * prod + 0.2 * ener + 0.2 * carb, 'mean_reward': float(np.mean(tracker.rewards)),
        'productivity': prod, 'energy_cost': ener, 'carbon': carb,
        'electricity_cost_sum': elec, 'gas_cost_sum': gas,
        'zone_temps_last': None,
    }
    (dest / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(rc)
    df['timestamp'] = [t.isoformat() for t in tracker.timestamps]
    df.to_csv(dest / 'reward_components.csv', index=False)
    return f'{name}: score {summary["score"]:.5f} ({n} steps, {elapsed/60:.1f} min)'


def main():
    ap = argparse.ArgumentParser(
        description='Official pid_water baseline on a scenario list, one month each, in parallel.')
    ap.add_argument('list', help='text file, one scenario directory per line (each holds config.gin)')
    ap.add_argument('out', help='output directory: <out>/<building>__<weather>/{summary.json,reward_components.csv}')
    ap.add_argument('workers', type=int, nargs='?', default=4, help='parallel scenarios (default 4)')
    ap.add_argument('steps', type=int, nargs='?', default=8640, help='steps per scenario (default 8640, one month)')
    a = ap.parse_args()
    lst, out, workers, steps = a.list, Path(a.out), a.workers, a.steps
    out.mkdir(parents=True, exist_ok=True)
    dirs = [l.strip() for l in Path(lst).read_text().splitlines() if l.strip()]
    jobs = [(d, None, str(out), steps) for d in dirs]
    print(f'{len(jobs)} scenarios, {workers} workers, {steps} steps each', flush=True)
    with mp.Pool(workers) as pool:
        for msg in pool.imap_unordered(run_one, jobs):
            print(msg, flush=True)


if __name__ == '__main__':
    main()

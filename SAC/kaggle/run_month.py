"""One deterministic month of a submission checkpoint through the official simulator.

Reports the competition score (and its three terms) plus the action statistics the
submission actually emitted, so a checkpoint can be checked end to end without a
GPU and without running the whole 30-scenario set.

    run_month.py <weights.bin> <building>/<weather> [steps]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    """The root that holds complete_port/, wherever this file sits: results/r139_submission/ in the
    development tree, complete_port/submission/<policy>/ in the release."""
    env = os.environ.get("SB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "complete_port").is_dir():
            return parent
    return here


REPO = _repo_root()
SB = REPO / 'starter-kit'
sys.path.insert(0, str(SB))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_buildings.evaluate import (  # noqa: E402
    SimulationTracker, create_building_meta_data, load_hybrid_environment, validate_action,
)
import submission as S  # noqa: E402

CKPT = str(Path(sys.argv[1]).resolve())
SCEN = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8640
path = (SB / 'smart_buildings/training_scenarios' / SCEN).resolve()
# a scenario config names its floor plan and zone map relative to the simulator root, so the
# loader has to run from there; the weights path is absolute before the change of directory
os.chdir(SB)

env = load_hybrid_environment(path / 'config.gin')
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = create_building_meta_data(env, path)
spec = env.action_spec()
policy = S.Policy(meta, spec)
assert not policy.fallback, 'building does not fit the canonical slots'

tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
tracker.reset()
obs = np.asarray(tracker.last_obs, dtype=np.float32)
cont = np.zeros((STEPS, policy.nC), np.float32)
disc = np.zeros((STEPS, policy.nD), np.int32)
t0 = time.time()
n = 0
for t in range(STEPS):
    action = validate_action(policy.get_action(obs), env.action_spec())
    cont[t] = action['continuous_action']
    disc[t] = action['discrete_action']
    obs, _, term, trunc, _ = tracker.step(action)
    obs = np.asarray(obs, dtype=np.float32)
    n += 1
    if term or trunc:
        break
elapsed = time.time() - t0

rc = tracker.reward_components
rewards = np.asarray(tracker.rewards, dtype=np.float64)
prod = np.mean([r['normalized_productivity_regret'] for r in rc])
ener = np.mean([r['normalized_energy_cost'] for r in rc])
carb = np.mean([r['normalized_carbon_emission'] for r in rc])
score = -0.6 * prod + 0.2 * ener + 0.2 * carb
print(f'\n{SCEN}   {n} steps   {elapsed / max(n, 1) * 1000:.2f} ms/step ({elapsed / max(n, 1):.3f} s/step)')
print(f'  mean reward {rewards.mean():+.6f}   score {score:.6f}')
print(f'  productivity {-prod:+.6f}   energy {ener:.6f}   carbon {carb:.6f}')
print(f'  continuous action: mean {cont[:n].mean():+.3f}  min {cont[:n].min():+.3f}  '
      f'max {cont[:n].max():+.3f}  |a|>0.95 {np.mean(np.abs(cont[:n]) > 0.95):.3f}')
print(f'  discrete action: on fraction {disc[:n].mean():.3f}')

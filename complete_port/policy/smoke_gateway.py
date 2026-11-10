"""Run the anticipation policy through starter-kit's evaluate loader.

One Fairbanks episode fragment. Confirms get_action uses the flat observation
and the metadata, resets its rings, and stays inside the action spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter-kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_buildings.evaluate import (  # noqa: E402
    create_building_meta_data,
    load_hybrid_environment,
    validate_action,
)
from anticipation_policy import AnticipationPolicy  # noqa: E402


def main() -> None:
    gin = (ROOT / "starter-kit/smart_buildings/training_scenarios"
           / "building_train_1/fairbanks_1/config.gin")
    env = load_hybrid_environment(gin)
    meta = create_building_meta_data(env, gin.parent)
    spec = env.action_spec()
    policy = AnticipationPolicy(meta, spec)
    obs, _ = env.reset()
    policy.reset()
    first = None
    for step in range(400):
        action = validate_action(policy.get_action(obs), spec)
        if step == 0:
            first = dict(policy.last)
        obs, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    last = policy.last
    assert first is not None
    hour = first["hour"]
    if not 5.5 <= hour <= 6.5:
        raise SystemExit(f"step 0 local hour {hour:.3f}, expected 06:00")
    # Sunday 1 Jan 2023: comfort is off, the next start is Monday morning.
    if first["during"] != 0.0:
        raise SystemExit("weekend morning was marked as comfort")
    if not 1400 <= first["minutes_to_start"] <= 1480:
        raise SystemExit(f"minutes to Monday start {first['minutes_to_start']:.1f}")
    if first["tomorrow_workday"] != 1.0:
        raise SystemExit("Monday was not a workday")
    if abs(first["oat_delta"]) > 1e-6:
        raise SystemExit("first step outside-air trend was not zero")
    feats = first["features"]
    if not np.isfinite(feats).all() or not np.isfinite(last["features"]).all():
        raise SystemExit("features were not finite")
    if policy.steps < 400 or int(policy.state.count.min()) < 48:
        raise SystemExit(f"memory did not advance: steps={policy.steps} count={policy.state.count.min()}")
    # 294 K init against a 294 K heating line is no deficit. A normalized value
    # read as Kelvin would show a deficit near 294.
    if abs(float(feats[:, 4].mean())) > 0.2:
        raise SystemExit(f"step-0 deficit/5 {float(feats[:, 4].mean()):.3f}, temperatures were not denormalized")
    print(
        f"gateway smoke ok steps={policy.steps} hour={hour:.3f} "
        f"to_start={first['minutes_to_start']:.1f} tomorrow={first['tomorrow_workday']:.0f} "
        f"phi0={first['phi']:.4f} phi_end={last['phi']:.4f} "
        f"deficit0={float(feats[:, 4].mean()):.3f} "
        f"slope_end={float(last['features'][:, 7].mean()):.3f}"
    )
    policy.reset()
    if policy.state.count.sum() != 0 or policy.state.last_oat is not None:
        raise SystemExit("reset left zone memory in place")


if __name__ == "__main__":
    main()

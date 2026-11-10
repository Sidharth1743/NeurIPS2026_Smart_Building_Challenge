#!/usr/bin/env python3
"""Oracle and trainer checks for seven full-episode calendar start offsets."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

CP = Path(__file__).resolve().parents[1]
SK = CP.parent / "starter-kit"
sys.path.insert(0, str(CP / "pysim"))
from sbb import BlobReader  # noqa: E402
PL_ROOT = os.environ.get("SB_PL_ROOT", os.path.expanduser("~/.cache/sb/pufferlib"))


def run(args, *, cwd=None, env=None, timeout=600):
    p = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {args}\n{p.stdout[-5000:]}")
    return p.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    out = CP / "results" / a.name
    out.mkdir(parents=True, exist_ok=False)
    py = str(SK / ".venv/bin/python")
    cfg = SK / "smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin"
    source = cfg.read_text()
    found = re.search(r"(?m)^start_timestamp\s*=\s*'([^']+)'\s*$", source)
    if not found:
        raise RuntimeError("scenario start_timestamp not found")
    start = pd.Timestamp(found.group(1))
    run([py, str(CP / "pysim/bake_scenario.py"), "--out", str(out),
         "--name", "random_start_fairbanks", "--random-start-days", "7", str(cfg)], cwd=SK)
    cfg2 = SK / "smart_buildings/training_scenarios/building_train_2/phoenix_6/config.gin"
    run([py, str(CP / "pysim/bake_scenario.py"), "--out", str(out),
         "--name", "random_start_phoenix", "--random-start-days", "7", str(cfg2)], cwd=SK)
    blob = BlobReader(out / "random_start_fairbanks.sbb")
    if list(blob.get("random_start_offsets")) != [288 * d for d in range(7)]:
        raise RuntimeError("incorrect start offsets")
    if int(blob.get("obs_meta")[3]) != 8640 or len(blob.get("weather_ambient")) != 10369:
        raise RuntimeError("episode length or table horizon is incorrect")

    # Check both ends of every full episode, including the last reachable row,
    # against the oracle's weather, schedules and civil-time calculations.
    os.chdir(SK)
    sys.path.insert(0, str(SK))
    from smart_buildings import evaluate
    from smart_buildings.smart_control.utils import conversion_utils as cu
    env = evaluate.load_hybrid_environment(cfg)
    env.reset()
    sim = env.building.simulator
    sch = next(iter(sim._hvac._vavs.values())).thermostat.get_setpoint_schedule()
    tz = env.building._occupancy._time_zone
    for d in range(7):
        for elapsed in (0, 1, 8639, 8640):
            k = d * 288 + elapsed
            ts = sim._start_timestamp + pd.Timedelta(minutes=5 * k)
            prior = ts - pd.Timedelta(minutes=5)
            local = ts.tz_convert(tz) if ts.tz is not None else ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            local_prev = prior.tz_convert(tz) if prior.tz is not None else prior.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            workday = lambda v: int(cu.is_work_day(pd.Timestamp(year=v.year, month=v.month, day=v.day)))
            rt = cu.proto_to_pandas_timestamp(cu.pandas_to_proto_timestamp(ts))
            heat, cool = sch.get_temperature_window(ts)
            checks = {
                "weather_ambient": sim._weather_controller.get_current_temp(ts),
                "sched_comfort": float(sch.is_comfort_mode(ts)),
                "sched_heat_sp": heat, "sched_cool_sp": cool,
                "occ_hour": local.hour, "occ_workday": workday(local),
                "occ_hour_m1": local_prev.hour, "occ_workday_m1": workday(local_prev),
                "rew_hour": rt.hour, "rew_month": rt.month, "rew_workday": workday(rt),
            }
            for key, expected_value in checks.items():
                if blob.get(key)[k] != expected_value:
                    raise RuntimeError(f"{key} disagrees with oracle at start {d}, elapsed {elapsed}")

    # Generate the independent oracle traces for all seven starts. These verify
    # the reset observation and each physics/reward/observation transition.
    for d in range(7):
        date = start + pd.Timedelta(days=d)
        cfg_d = out / f"day_{d}.gin"
        cfg_d.write_text(source[:found.start(1)] + date.strftime("%Y-%m-%d %H:%M") + source[found.end(1):])
        run([py, str(CP / "pysim/dump_step.py"), str(cfg_d), str(out / f"trace_{d}.step"), "16"], cwd=SK)

    docker = ["docker", "run", "--rm", "--gpus", f"device={a.gpu}",
              "-v", f"{CP}:/cp", "-v", PL_ROOT + ":/work",
              "-v", f"{out}:/check", "-w", "/work/plverify", "pufferai/puffertank:5.0"]
    for d in range(7):
        text = run(docker + ["env", f"SB_GATE_START_OFFSET={d * 288}", "SB_GRAPH=1",
                            "/cp/artifacts/bin/gate", "/check/random_start_fairbanks.sbb",
                            f"/check/trace_{d}.step", "2"], timeout=180)
        (out / f"oracle_{d}.log").write_text(text)
        if "SB GPU ENV STEP: OK" not in text:
            raise RuntimeError(f"oracle comparison failed for day {d}:\n{text[-2500:]}")

    trainer = ["./puffer_shared", "train", "--env.blob_dir=/check", "--env.groups=2",
               "--env.random_start=1", "--env.random_start_seed=75", "--env.desync=1",
               "--env.canonical=1", "--env.zone_memory=1", "--env.zone_critic=1",
               "--env.max_steps=8", "--env.reward_scale=1", "--policy.hidden_size=64",
               "--policy.num_layers=1", "--policy.tanh_mean=0", "--policy.beta=1",
               "--policy.zone_mlp=1", "--policy.beta_cn_tau=12", "--policy.beta_cn_tau_ahu=48",
               "--policy.beta_conc_max=12", "--policy.value_shared=1", "--policy.value_delay=1",
               "--train.zone_critic=1", "--train.valve_zone=1", "--train.adv_norm=2",
               "--train.role_adam=1", "--train.horizon=16", "--train.minibatch_size=256",
               "--train.replay_ratio=1", "--train.total_timesteps=2048",
               "--base.async=1", "--base.eval_episodes=0", "--base.cudagraphs=1",
               "--base.checkpoint_interval=1", "--base.log_dir=/check/train",
               "--base.checkpoint_dir=/check/train", "--vec.total_agents=64"]
    text = run(docker + ["timeout", "180"] + trainer, timeout=220)
    (out / "trainer.log").write_text(text)
    if "random_start on" not in text or "desync disabled" not in text:
        raise RuntimeError("random_start startup banner missing")

    # The viewer environment has TensorBoard. Check every initial draw against
    # the seeded stream, and every short episode completion under graph replay.
    reader = r'''import json,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
e=EventAccumulator(sys.argv[1],size_guidance={'scalars':0});e.Reload()
print(json.dumps({t:[x.value for x in e.Scalars(t)] for t in e.Tags()['scalars']}))'''
    tbdirs = list((out / "train").rglob("tb"))
    if len(tbdirs) != 1:
        raise RuntimeError(f"expected one TensorBoard directory, got {tbdirs}")
    vals = json.loads(run([str(CP / "viewer/.venv/bin/python"), "-c", reader, str(tbdirs[0])]))
    mask = (1 << 64) - 1
    gamma = 0x9e3779b97f4a7c15
    def mix(x):
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    stems = ["scenario/random_start_fairbanks/start_day_", "scenario/random_start_phoenix/start_day_"]
    days_seen = set()
    for group, stem in enumerate(stems):
        expected = collections.Counter(mix((mix(75 ^ ((gamma * (i + 1)) & mask)) + gamma) & mask) % 7
                                       for i in range(group * 32, (group + 1) * 32))
        days_seen.update(expected)
        for d in range(7):
            actual = sum(vals.get(f"{stem}{d}/initial_draws", []))
            if actual != expected[d]:
                raise RuntimeError(f"group {group} initial day {d}: {actual} != {expected[d]}")
    later = sum(sum(v) for k, v in vals.items() if k.startswith(tuple(stems)) and k.endswith("/later_draws"))
    completed = sum(sum(v) for k, v in vals.items() if k.startswith(tuple(stems)) and k.endswith("/count"))
    if later != 256 or completed != 256:
        raise RuntimeError(f"graph reset/episode counts wrong: later={later}, complete={completed}, expected 256")
    if not vals.get("env/episode_length") or any(v != 8 for v in vals["env/episode_length"]):
        raise RuntimeError("episode was shortened by desync")
    if len(days_seen) != 7:
        raise RuntimeError("initial draws did not cover all seven days")
    print("seven shifted GPU/oracle traces, two groups, seeded first draws, graph resets and full smoke episodes: OK")


if __name__ == "__main__":
    main()

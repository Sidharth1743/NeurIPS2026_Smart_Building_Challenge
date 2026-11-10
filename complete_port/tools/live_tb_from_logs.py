#!/usr/bin/env python3
"""Bridge the live r139 app log into a local TensorBoard run.

    live_tb_from_logs.py <app-id> <profile> <logdir> [steps-per-epoch]

The checkpoint snapshots on the volume are at best one save behind (every 144 epochs,
~52 min), and pulling more often cannot change that. The app's stdout, however, is live:
the trainer prints its console dashboard once per epoch (~20 s). This script follows
`modal app logs <app> -f` (a read-only log subscription: no CPU, GPU or memory of the
training container) and writes one scalar point per dashboard block into a local event
file, under `live/<name>` tags so they cannot be confused with the trainer's own run.

If the log stream drops, the script reconnects after 5 s. It never writes to the volume.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

FIELDS = ('perf', 'score', 'episode_return', 'episode_length', 'comfort_dev_K',
          'worst_dev_K', 'out_of_band_frac', 'productivity', 'energy_cost_norm',
          'carbon_norm', 'elec_kwh', 'gas_kwh', 'episode_score_scen_mean',
          'episode_score_n_scen')
PAT = {k: re.compile(rf'\b{re.escape(k)}\s+(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)') for k in FIELDS}
SB = re.compile(r'^SB_UPDATE\s+(\d+)\s')


def main():
    app, profile, logdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    spe = int(sys.argv[4]) if len(sys.argv) > 4 else 276480
    logdir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(logdir), max_queue_size=64, flush_secs=2)
    print(f'live bridge: app {app}, profile {profile}, logdir {logdir}', flush=True)

    epoch = None
    while True:
        cmd = ['modal', 'app', 'logs', app, '--profile', profile, '-f']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, errors='replace', bufsize=1)
        except OSError as exc:
            print(f'cannot start modal: {exc}', flush=True)
            time.sleep(5)
            continue
        try:
            for line in proc.stdout:
                m = SB.match(line)
                if m:
                    epoch = int(m.group(1))
                    continue
                if '│' not in line:
                    continue
                vals = {}
                for k, pat in PAT.items():
                    mm = pat.search(line)
                    if mm:
                        vals[k] = float(mm.group(1))
                if not vals:
                    continue
                step = (epoch or 0) * spe
                for k, v in vals.items():
                    s = Summary(value=[Summary.Value(tag=f'live/{k}', simple_value=v)])
                    writer.add_event(Event(step=step, wall_time=time.time(), summary=s))
                writer.flush()
                print(f'live step {step} ' + ' '.join(f'{k}={v:g}' for k, v in vals.items()), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print('log stream ended; reconnecting in 5 s', flush=True)
        time.sleep(5)


if __name__ == '__main__':
    main()

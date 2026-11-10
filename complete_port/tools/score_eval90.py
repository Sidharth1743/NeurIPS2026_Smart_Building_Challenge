"""Score checkpoints on the five-building held-out set on the GPU, deterministic fp64, with recordings.

Each checkpoint is evaluated in three parts of 30 scenarios (one env per scenario, one full 8,640-step
month each) and merged into one 90-scenario JSON by `tools/merge_eval_json.py`, exactly the split
`tools/eval_policy.sh` uses. The flags come from the run's own `args.txt` (only policy / env / train
keys, the same filter the r110 mean-action scorer used), so the eval build decodes the checkpoint
the way it was trained. Every part writes `SB_RECORD` (per-step obs, executed action, reward, parts)
so month, day, day-part, per-zone and per-action views can be produced afterwards without re-running.

    score_eval90.py --run <run dir> --runtime <runtime dir> --set <set dir> --out <results dir> \
                    --gpu 0 --checkpoints <step>=<ckpt path> [<step>=<ckpt> ...]

Refuses to overwrite an existing part, log or merged JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

CP = Path(os.environ.get("SB_CP", Path(__file__).resolve().parents[1]))
RESULTS = CP / 'results'
PREF = ('--policy.', '--env.canonical=', '--env.residual=', '--env.zone_memory=',
        '--env.zone_critic=', '--env.reward_scale=', '--env.comfort_shape=',
        '--train.zone_critic=', '--train.valve_zone=', '--train.zone_adv=',
        '--train.room_value=', '--train.component_values=', '--train.freeze_return=',
        '--train.role_adam=')


def ist():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%Y-%m-%d %H:%M:%S IST')


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def flags(run: Path):
    """The flags that decide how a checkpoint is decoded. A run archived without its launch
    arguments carries them in flags.txt instead; both files hold the same token syntax."""
    src = run / 'args.txt'
    if not src.exists():
        src = run / 'flags.txt'
    args = shlex.split(src.read_text())
    out = {a.split('=', 1)[0]: a for a in args if a.startswith(PREF)}
    return list(out.values())


def parts_of(set_dir: Path):
    """One part per building.  The recorder (SB_RECORD) asserts a single env group, and the env
    merges the weathers of one building into one group, so a part must hold exactly one building."""
    blobs = sorted(set_dir.glob('*.sbb'))
    by_building = {}
    for b in blobs:
        by_building.setdefault(b.stem.split('__')[0], []).append(b)
    return [(k, v) for k, v in sorted(by_building.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--set', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--checkpoints', nargs='+', required=True, help='step=path')
    ap.add_argument('--ckpt-bytes', type=int, default=1814944,
                    help='expected weight-file size (network-specific; r122 1814944, r139 1812032)')
    a = ap.parse_args()

    run = Path(a.run).resolve(); runtime = Path(a.runtime).resolve()
    # a relative --out lands beside the other scores, not at the root of complete_port/
    out = Path(a.out) if Path(a.out).is_absolute() else RESULTS / a.out
    set_dir = CP / a.set if not Path(a.set).is_absolute() else Path(a.set)
    binary = runtime / 'puffer_shared'
    assert binary.is_file(), binary
    assert b'EEv5LuDevPf' not in binary.read_bytes(), f'{binary} contains the fp32 solve symbol; it is not a score binary'
    blobs = sorted(set_dir.glob('*.sbb'))
    parts = parts_of(set_dir)
    assert parts, f'no scenarios in {set_dir}'
    assert all(len(v) == 18 for _, v in parts), [(k, len(v)) for k, v in parts]
    assert len(blobs) == 18 * len(parts), f'expected 18 per building in {set_dir}, found {len(blobs)}'

    out.mkdir(parents=True, exist_ok=True)
    (out / 'inputs').mkdir(exist_ok=True)
    part_dirs = []
    for i, (building, part) in enumerate(parts):
        d = out / 'inputs' / building
        d.mkdir(exist_ok=True)
        for b in part:
            link = d / b.name
            # The set directory holds symlinks to host paths, and inside the container only
            # /res and /art exist. Hard-link the RESOLVED blob (os.link on a symlink path
            # links the symlink, not the file), and refuse to keep a link that points
            # outside the mounted tree.
            if link.is_symlink():
                link.unlink()
            if not link.exists():
                try:
                    os.link(b.resolve(), link)
                except OSError:
                    shutil.copy2(b.resolve(), link)
            assert link.is_file() and not link.is_symlink(), f'part entry is not a real file: {link}'
            assert link.stat().st_size > 20_000_000, f'part blob too small: {link}'   # smallest shipped building is ~90 MB
        part_dirs.append(d)
    (out / 'plan.json').write_text(json.dumps({
        'created': ist(), 'run': str(run), 'runtime': str(runtime), 'binary_sha256': digest(binary),
        'set': str(set_dir), 'scenarios': [b.stem for b in blobs],
        'parts': {k: [b.stem for b in v] for k, v in parts},
        'checkpoints': a.checkpoints, 'gpu': a.gpu,
        'flags': flags(run),
    }, indent=1) + '\n')

    base_flags = flags(run)
    for spec in a.checkpoints:
        step, ck = spec.split('=', 1)
        step = int(step); ck = Path(ck)
        assert ck.is_file() and ck.stat().st_size == a.ckpt_bytes, f'bad checkpoint {ck} ({ck.stat().st_size} B)'
        # The container sees results/ as /res and artifacts/ as /art; a host path is not visible.
        # A checkpoint shipped in experiments/ is staged into results/ first.
        ck_abs = ck.resolve()
        if RESULTS.resolve() not in ck_abs.parents:
            stage = RESULTS / '_score_input'
            stage.mkdir(parents=True, exist_ok=True)
            staged = stage / ck_abs.name
            if not staged.exists():
                shutil.copy2(ck_abs, staged)
            ck_abs = staged.resolve()
        assert RESULTS.resolve() in ck_abs.parents, f'checkpoint is outside results/: {ck}'
        ck_in = '/res/' + str(ck_abs.relative_to(RESULTS.resolve()))
        dest = out / f'{step:016d}'
        dest.mkdir(exist_ok=True)
        (dest / 'checkpoint.json').write_text(json.dumps(
            {'step': step, 'path': str(ck), 'bytes': ck.stat().st_size, 'sha256': digest(ck)}, indent=1) + '\n')
        merged = dest / 'score.json'
        if merged.is_file():
            print(f'{ist()} skip {step}: {merged} exists', flush=True)
            continue
        part_jsons = []
        for (building, part), pd in zip(parts, part_dirs):
            rel = (dest / building)
            rel.mkdir(exist_ok=True)
            json_out = rel / 'score.json'
            log = rel / 'eval.log'
            if json_out.is_file():
                print(f'{ist()} skip {building} of {step}', flush=True)
                part_jsons.append(json_out); continue
            assert not log.exists(), f'preserving existing log {log}'
            (rel / 'record').mkdir(exist_ok=True)
            cmd = ['docker', 'run', '--rm', '--name', f'sb-eval90-{step}-{building}', '--gpus', f'device={a.gpu}',
                   '-e', f'SB_EVAL_OUT=/res/{json_out.relative_to(RESULTS)}',
                   '-e', f'SB_RECORD=/res/{(rel / "record").relative_to(RESULTS)}',
                   '-v', f'{runtime}:/work:ro', '-v', f'{CP / "artifacts"}:/art:ro',
                   '-v', f'{RESULTS}:/res', '-w', '/work', 'pufferai/puffertank:5.0',
                   './puffer_shared', 'eval', ck_in, *base_flags,
                   '--headless', f'--env.blob_dir=/res/{pd.relative_to(RESULTS)}',
                   # one scenario per env, all 18 weathers of this building merged into ONE group
                   # (the env's merge_weathers default; a group is what the recorder accepts)
                   f'--env.groups={len(part)}', f'--base.eval_agents={len(part)}',
                   f'--base.eval_episodes={len(part)}', '--env.merge_weathers=1', '--env.desync=0',
                   '--base.deterministic=1', '--base.cudagraphs=-1', '--train.horizon=288']
            (rel / 'command.json').write_text(json.dumps(cmd, indent=1) + '\n')
            t0 = time.monotonic()
            print(f'{ist()} START {step} {building} ({len(part)} scenarios, one group)', flush=True)
            with log.open('xb') as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            dt = time.monotonic() - t0
            if rc != 0 or not json_out.is_file():
                tail = log.read_text(errors='replace')[-1200:]
                print(f'{ist()} FAIL {step} {building} rc={rc} after {dt/60:.1f} min\n{tail}', flush=True)
                sys.exit(1)
            d = json.loads(json_out.read_text())
            print(f'{ist()} PASS {step} {building} {d["n_scenarios_done"]}/{len(part)} '
                  f'score {d["score"]:.6f} in {dt/60:.1f} min', flush=True)
            part_jsons.append(json_out)
        r = subprocess.run([sys.executable, str(CP / 'tools/merge_eval_json.py'), str(merged)] +
                           [str(p) for p in part_jsons], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip(), flush=True)
        assert r.returncode == 0 and merged.is_file(), 'merge failed'
        d = json.loads(merged.read_text())
        print(f'{ist()} CHECKPOINT {step} score {d["score"]:.6f} complete {d["complete"]} '
              f'{d["n_scenarios_done"]}/90', flush=True)


if __name__ == '__main__':
    main()

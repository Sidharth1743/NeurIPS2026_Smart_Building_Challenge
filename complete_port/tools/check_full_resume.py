#!/usr/bin/env python3
"""Exercise a real graph/async complete checkpoint inside the trainer gate."""
import argparse
import hashlib
import os
import re
import shlex
import shutil
import struct
import subprocess
from pathlib import Path

CP = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--checkout', default='/data/puffer-work/plverify')
    p.add_argument('--restore-gpu', default=os.environ.get('SB_VERIFY_CROSS_GPU'))
    a = p.parse_args()
    out = CP / 'results' / a.name
    out.mkdir(parents=True, exist_ok=False)
    fixture = Path(a.fixture).resolve()
    if sorted(p.name for p in fixture.glob('*.sbb')) != ['random_start_fairbanks.sbb', 'random_start_phoenix.sbb']:
        raise RuntimeError('random-start two-building fixture missing')
    docker = ['docker', 'run', '--rm', '--gpus', f'device={a.gpu}',
              '-v', f'{a.checkout}:/work', '-v', f'{fixture}:/fixture:ro',
              '-v', f'{out}:/out', '-w', '/work', '-e', 'SB_CREDIT=8',
              '-e', 'SB_DETAIL_METRICS=1', 'pufferai/puffertank:5.0']
    common = ['--env.blob_dir=/fixture', '--env.groups=2', '--env.random_start=1',
              '--env.random_start_seed=75', '--env.desync=0', '--env.canonical=1',
              '--env.zone_memory=1', '--env.zone_critic=1', '--env.max_steps=7', '--env.reward_scale=1',
              '--policy.hidden_size=64', '--policy.num_layers=2', '--policy.tanh_mean=0',
              '--policy.tanh_squash=0', '--policy.beta=1', '--policy.beta_floor=1',
              '--policy.zone_mlp=1', '--policy.beta_cn_tau=12', '--policy.beta_cn_tau_ahu=48',
              '--policy.beta_conc_max=12', '--policy.value_shared=1', '--policy.value_delay=1',
              '--train.zone_critic=1', '--train.valve_zone=1', '--train.headwise_ratio=1',
              '--train.adv_norm=2', '--train.role_adam=1', '--train.rew_clamp=0',
              '--train.horizon=16', '--train.minibatch_size=256', '--train.replay_ratio=2',
              '--train.total_timesteps=4096', '--train.anneal_lr=1', '--train.anneal_steps=4096',
              '--train.update_diagnostics=1', '--base.seed=75', '--base.async=1',
              '--base.eval_episodes=0', '--base.cudagraphs=1', '--base.checkpoint_optim=1',
              '--base.full_checkpoint=1', '--base.checkpoint_interval=2', '--base.run_id=resume_gate',
              '--vec.total_agents=64']

    def run(name, extra=(), expected=0, gpu=None):
        invocation = docker.copy()
        if gpu is not None:
            invocation[4] = f'device={gpu}'
        command = invocation + ['timeout', '240', './puffer_shared', 'train', *common,
                            f'--base.log_dir=/out/{name}', f'--base.checkpoint_dir=/out/{name}', *extra]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=270)
        (out / (name + '.log')).write_text('COMMAND ' + shlex.join(command) + '\n' + proc.stdout)
        if (proc.returncode == 0) != (expected == 0):
            raise RuntimeError(f'{name} returned {proc.returncode}:\n' + '\n'.join(proc.stdout.splitlines()[-35:]))
        return proc.stdout

    log = run('original')
    checkpoints = sorted((out / 'original').rglob('*.bin'))
    if len(checkpoints) != 2 or not all(Path(str(p) + '.full').is_file() for p in checkpoints):
        raise RuntimeError(f'expected two committed checkpoints: {checkpoints}')
    first, last = checkpoints
    first_container = '/out/' + str(first.relative_to(out))
    # Emulate interruption during the next save. The replay must preserve the
    # incomplete components and publish the expected committed checkpoint.
    pending = out / 'resumed' / 'smart_buildings' / 'resume_gate' / last.name
    pending.parent.mkdir(parents=True)
    saved = pending.parent / first.name
    for suffix in ('', '.mb', '.hadam', '.full'):
        shutil.copyfile(Path(str(first) + suffix), Path(str(saved) + suffix))
    for suffix in ('', '.mb', '.hadam'):
        Path(str(pending) + suffix).write_bytes(b'interrupted save')
        for pid in range(1, 65):
            Path(str(pending) + suffix + f'.orphan.{pid}.0').write_bytes(b'prior interruption')
    for pid in range(1, 65):
        Path(str(pending) + f'.full.part.{pid}.0').write_bytes(b'prior staging')
    log2 = run('resumed', ['--base.load_model_path=latest'])
    if 'FULL_CHECKPOINT restored ' not in log2:
        raise RuntimeError('complete checkpoint restoration banner missing')
    match = re.search(r'\.orphan\.(\d+)\.1\b', log2)
    if not match:
        raise RuntimeError('reused process ID did not preserve the existing orphan')
    resumed_pid = int(match.group(1))
    if not 1 <= resumed_pid <= 64 or Path(str(pending) + f'.full.part.{resumed_pid}.0').read_bytes() != b'prior staging':
        raise RuntimeError('existing snapshot staging file was not preserved')
    last2s = list((out / 'resumed').rglob(last.name))
    if len(last2s) != 1:
        raise RuntimeError('resumed process did not reach the original final step')
    last2 = last2s[0]
    for suffix in ('', '.mb', '.hadam'):
        left, right = Path(str(last) + suffix), Path(str(last2) + suffix)
        if not left.is_file() or not right.is_file():
            raise RuntimeError(f'{suffix or "weights"} checkpoint component missing')
        lh = hashlib.sha256(left.read_bytes()).hexdigest()
        rh = hashlib.sha256(right.read_bytes()).hexdigest()
        if lh != rh:
            raise RuntimeError(f'{suffix or "weights"} mismatch after resuming: {lh} != {rh}')
        preserved = list(last2.parent.glob(last2.name + suffix + f'.orphan.{resumed_pid}.1'))
        if len(preserved) != 1 or preserved[0].read_bytes() != b'interrupted save':
            raise RuntimeError(f'interrupted {suffix or "weights"} was not preserved')
        if Path(str(last2) + suffix + f'.orphan.{resumed_pid}.0').read_bytes() != b'prior interruption':
            raise RuntimeError('existing orphan was not preserved')

    # Reject incompatible configuration, missing sidecars, and corruption
    # before a resumed update can run.
    def rejected(name, path, extra=(), reason='full checkpoint:'):
        text = run(name, [f'--base.load_model_path={path}', *extra], expected=1)
        if reason not in text or 'FULL_CHECKPOINT restored ' in text:
            raise RuntimeError(f'{name} did not reject checkpoint before restoration')
    rejected('changed_config', first_container, ['--train.learning_rate=0.0002'], 'mismatch')
    broken = out / 'broken'
    broken.mkdir()
    ck = broken / 'checkpoint.bin'
    shutil.copyfile(first, ck)
    shutil.copyfile(Path(str(first) + '.mb'), Path(str(ck) + '.mb'))
    rejected('missing_snapshot', '/out/broken/checkpoint.bin', reason='complete snapshot missing')
    with Path(str(first) + '.full').open('rb') as src, Path(str(ck) + '.full').open('wb') as dst:
        dst.write(src.read(16))
    rejected('truncated_snapshot', '/out/broken/checkpoint.bin', reason='truncated header')
    shutil.copyfile(Path(str(first) + '.full'), Path(str(ck) + '.full'))
    with Path(str(ck) + '.full').open('r+b') as f:
        header = f.read(16)
        header_size = struct.unpack_from('<I', header, 12)[0]
        f.seek(header_size + 80)
        byte = f.read(1)
        if len(byte) != 1:
            raise RuntimeError('cannot corrupt checkpoint payload')
        f.seek(-1, 1)
        f.write(bytes([byte[0] ^ 1]))
    rejected('corrupt_snapshot', '/out/broken/checkpoint.bin', reason='checksum mismatch')
    if a.restore_gpu:
        if a.restore_gpu == a.gpu:
            raise RuntimeError('cross-GPU test requires a different GPU')
        strict = run('cross_strict', [f'--base.load_model_path={first_container}'],
                     expected=1, gpu=a.restore_gpu)
        if 'mismatch' not in strict or 'FULL_CHECKPOINT restored ' in strict:
            raise RuntimeError('strict mode accepted a different GPU')
        portable = run('cross_portable', [f'--base.load_model_path={first_container}',
                                          '--base.portable_resume=1'], gpu=a.restore_gpu)
        if 'FULL_CHECKPOINT portable platform change' not in portable or 'FULL_CHECKPOINT restored ' not in portable:
            raise RuntimeError('cross-GPU complete restore was not reported')
        if not list((out / 'cross_portable').rglob(last.name + '.full')):
            raise RuntimeError('cross-GPU continuation failed to commit the final snapshot')
        print('cross-GPU strict rejection and portable state restoration passed')
    print('async graph resume matched uninterrupted weights, Muon and role Adam at the final checkpoint')


if __name__ == '__main__':
    main()

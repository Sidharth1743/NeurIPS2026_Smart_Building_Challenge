#!/usr/bin/env python3
"""One graph smoke and checkpoint round-trip, inside the existing trainer gate."""
import argparse, math, re, shlex, struct, subprocess
from pathlib import Path
import os
PL_ROOT = os.environ.get("SB_PL_ROOT", os.path.expanduser("~/.cache/sb/pufferlib"))
CP=Path(__file__).resolve().parents[1]

def hadam(path):
    raw=path.read_bytes()
    if len(raw)<32: raise RuntimeError('short Adam header')
    magic,version,n,off,step,reserved=struct.unpack_from('<8sIiqiI',raw)
    if magic!=b'SBHADAM1' or version!=1 or n<=0 or off<0 or step<=0 or reserved or len(raw)!=32+8*n:
        raise RuntimeError(f'invalid role Adam sidecar {path}')
    state=struct.unpack_from(f'<{2*n}f',raw,32)
    if not all(math.isfinite(x) for x in state) or not any(state[:n]) or any(x<0 for x in state[n:]):
        raise RuntimeError('invalid or empty Adam moments')
    return n,off,step

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--name',required=True);a=p.parse_args()
    base=CP/'results'/a.name
    if base.exists() or (CP/'results'/(a.name+'_restore')).exists():raise RuntimeError('refusing to reuse smoke output')
    empty=subprocess.check_output(['python3',str(CP/'tools/zone_empty_obs.py'),str(CP/'artifacts/verify_multi'),'--count','3'],text=True).strip()
    common=['--env.blob_dir=/art/verify_multi','--env.groups=3','--env.canonical=1','--env.zone_memory=1',
      '--env.zone_critic=1','--env.max_steps=8','--env.desync=0','--env.reward_scale=1',
      '--policy.hidden_size=64','--policy.num_layers=1','--policy.tanh_mean=0','--policy.tanh_squash=0',
      '--policy.beta=1','--policy.zone_mlp=1','--policy.zone_value_residual=1',
      '--policy.beta_cn_tau=12','--policy.beta_cn_tau_ahu=48','--policy.beta_conc_max=12',
      '--policy.value_shared=1','--policy.value_delay=1','--train.zone_critic=1','--train.valve_zone=1',
      '--train.headwise_ratio=1','--train.adv_norm=2','--train.role_adam=1','--train.rew_clamp=0',
      '--train.horizon=64','--train.minibatch_size=256','--train.replay_ratio=2','--train.update_diagnostics=1',
      '--base.async=1','--base.eval_episodes=0','--base.cudagraphs=1','--base.checkpoint_optim=1',
      '--base.checkpoint_interval=1','--vec.total_agents=24']
    docker=['docker','run','--rm','--gpus',f'device={a.gpu}',
      '-v',PL_ROOT + ":/work",'-v',f'{CP}/artifacts:/art','-v',f'{CP}/results:/res',
      '-w','/work/plverify','-e','SB_CREDIT=8','-e','SB_DETAIL_METRICS=1','-e','SB_ZONE_STATS=1',
      '-e','SB_ZONE_EMPTY_OBS='+empty,'pufferai/puffertank:5.0']
    def run(name,total,extra=()):
        log=CP/'results'/(name+'.log')
        args=common+[f'--train.total_timesteps={total}',f'--base.log_dir=/res/{name}',
                     f'--base.checkpoint_dir=/res/{name}',*extra]
        cmd=docker+['timeout','180','./puffer_shared','train',*args]
        done=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=200)
        log.write_text('COMMAND '+shlex.join(cmd)+'\n'+done.stdout)
        if done.returncode: raise RuntimeError(f'{name} failed rc={done.returncode}:\n'+ '\n'.join(done.stdout.splitlines()[-30:]))
        return done.stdout
    log=run(a.name,6144)
    for token in ['headwise Beta marginal PPO ratio on','zone MLP on role Adam',
                  'shared local zone value residual on','zone value residual on role Adam',
                  'SB_ZONE_MC48 ','SB_ZONE_RESIDUAL ','SB_ZONE_VALUE ','SB_HEAD_FIRST ']:
        if token not in log:raise RuntimeError('missing '+token)
    bad=[x for x in log.splitlines() if x.startswith('SB_ZONE_BAD ')]
    if not bad or any(float(x.split()[3])!=0 for x in bad):raise RuntimeError('zone telemetry has invalid values')
    rows=[x.split() for x in log.splitlines() if x.startswith('SB_ZONE_RESIDUAL ')]
    if not rows or not any(float(x[4])>0 and float(x[3])>0 for x in rows):
        raise RuntimeError('residual output or first-layer gradient did not learn')
    counts={int(x.split()[2]) for x in log.splitlines() if x.startswith('SB_ZONE_MC48 ') and float(x.split()[6])>0}
    if counts!={0,1,2}:raise RuntimeError(f'missing MC48 group rows: {counts}')
    ckpts=sorted(base.rglob('*.bin'))
    if not ckpts:raise RuntimeError('checkpoint missing')
    ck=ckpts[-1];side=Path(str(ck)+'.hadam');mb=Path(str(ck)+'.mb')
    if not side.is_file() or not mb.is_file() or ck.stat().st_size!=mb.stat().st_size:
        raise RuntimeError('checkpoint/Muon/Adam sidecars missing or incorrect')
    n,off,step=hadam(side)
    ck_container='/res/'+str(ck.relative_to(CP/'results'))
    # This deliberately exercises the legacy optimizer-only restore. Complete
    # continuation is separately covered by check_full_resume.py.
    restored=run(a.name+'_restore',1536,[f'--base.load_model_path={ck_container}',
                                        '--base.allow_partial_resume=1'])
    pattern=rf'restored role Adam state .* \({n} weights, step {step}\)'
    if not re.search(pattern,restored):raise RuntimeError('Adam checkpoint was not restored with exact shape and step')
    if 'loaded optimizer momentum from ' not in restored:raise RuntimeError('Muon checkpoint was not restored')
    second=sorted((CP/'results'/(a.name+'_restore')).rglob('*.bin'))
    if not second:raise RuntimeError('restored checkpoint missing')
    n2,off2,step2=hadam(Path(str(second[-1])+'.hadam'))
    if (n2,off2)!=(n,off) or step2<=step:raise RuntimeError('restored Adam did not continue')
    print(f'PASS residual mixed-group CUDA graph, 48-step telemetry and exact Adam checkpoint restoration ({n} weights, {step}->{step2} steps)')

if __name__=='__main__':main()

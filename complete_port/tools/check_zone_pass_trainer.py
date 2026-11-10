#!/usr/bin/env python3
"""Exercise pass-refresh training in the existing trainer gate, not a parity run."""
import argparse, math, shlex, subprocess
from pathlib import Path
from check_zone_residual_trainer import hadam
CP=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--name',required=True);a=p.parse_args()
 base=CP/'results'/a.name
 if base.exists():raise RuntimeError('refusing to reuse gate output')
 empty=subprocess.check_output(['python3',str(CP/'tools/zone_empty_obs.py'),str(CP/'artifacts/verify_multi'),'--count','3'],text=True).strip()
 args=['--env.blob_dir=/art/verify_multi','--env.groups=3','--env.canonical=1','--env.zone_memory=1',
  '--env.zone_critic=1','--env.max_steps=8','--env.desync=0','--env.reward_scale=1',
  '--policy.hidden_size=64','--policy.num_layers=1','--policy.tanh_mean=0','--policy.tanh_squash=0',
  '--policy.beta=1','--policy.zone_mlp=1','--policy.zone_value_residual=0',
  '--policy.beta_cn_tau=12','--policy.beta_cn_tau_ahu=48','--policy.beta_conc_max=12',
  '--policy.value_shared=1','--policy.value_delay=1','--train.zone_critic=1','--train.valve_zone=1',
  '--train.headwise_ratio=1','--train.adv_norm=2','--train.role_adam=1','--train.rew_clamp=0',
  '--train.zone_pass_refresh=1','--train.horizon=64','--train.minibatch_size=256',
  '--train.replay_ratio=2','--train.total_timesteps=6144','--train.update_diagnostics=1',
  '--base.async=1','--base.eval_episodes=0','--base.cudagraphs=1','--base.checkpoint_optim=1',
  '--base.checkpoint_interval=1','--vec.total_agents=24',f'--base.log_dir=/res/{a.name}',
  f'--base.checkpoint_dir=/res/{a.name}']
 cmd=['docker','run','--rm','--gpus',f'device={a.gpu}','-v','/data/puffer-work:/work',
  '-v',f'{CP}/artifacts:/art','-v',f'{CP}/results:/res','-w','/work/plverify',
  '-e','SB_CREDIT=8','-e','SB_DETAIL_METRICS=1','-e','SB_ZONE_STATS=1','-e','SB_ZPASS_ASSERT=1',
  '-e','SB_ZONE_EMPTY_OBS='+empty,'pufferai/puffertank:5.0','timeout','180','./puffer_shared','train',*args]
 done=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=200)
 (CP/'results'/(a.name+'.log')).write_text('COMMAND '+shlex.join(cmd)+'\n'+done.stdout)
 if done.returncode:raise RuntimeError(f'train failed rc={done.returncode}:\n'+'\n'.join(done.stdout.splitlines()[-30:]))
 log=done.stdout
 for token in ['zone GAE, targets and normalization refreshed together per replay pass','SB_ZONE_MC48 ','SB_ZONE_PASS ','SB_ZONE_PASSSTAT ','SB_HEAD_FIRST ']:
  if token not in log:raise RuntimeError('missing '+token)
 stats=[x.split() for x in log.splitlines() if x.startswith('SB_ZONE_PASSSTAT ')]
 seen={}
 for x in stats:
  epoch,slot,ps=int(x[1]),int(x[2]),int(x[3]);mean,inv=float(x[4]),float(x[5]);count=int(x[6])
  if not math.isfinite(mean) or not math.isfinite(inv) or inv<=0 or count!=1:raise RuntimeError('invalid pass stats '+str(x))
  seen.setdefault((epoch,slot),set()).add(ps)
 if len(seen)<2 or {slot for _,slot in seen}!={0,1} or any(v!={0,1} for v in seen.values()):
  raise RuntimeError(f'passes/async slots incomplete {seen}')
 rows=[x.split() for x in log.splitlines() if x.startswith('SB_ZONE_PASS ')]
 if {(int(x[2]),int(x[3])) for x in rows}!={(p,g) for p in range(2) for g in range(3)}:raise RuntimeError('missing pass/group rows')
 if any(not all(math.isfinite(float(y)) for y in x[7:]) for x in rows):raise RuntimeError('nonfinite paired statistics')
 if not any(float(x[12])>0 for x in rows):raise RuntimeError('no observed within-pass critic drift')
 bad=[x for x in log.splitlines() if x.startswith('SB_ZONE_BAD ')]
 if not bad or any(float(x.split()[3])!=0 for x in bad):raise RuntimeError('invalid zone data')
 ckpts=sorted(base.rglob('*.bin'))
 if not ckpts:raise RuntimeError('no checkpoint')
 ck=ckpts[-1];mb=Path(str(ck)+'.mb');side=Path(str(ck)+'.hadam')
 if not side.is_file() or not mb.is_file() or mb.stat().st_size!=ck.stat().st_size:raise RuntimeError('missing optimizer sidecars')
 hadam(side)
 print('PASS async slot-isolated cached zone GAE with pass normalization, paired diagnostics, resets and optimizer sidecars')
if __name__=='__main__':main()

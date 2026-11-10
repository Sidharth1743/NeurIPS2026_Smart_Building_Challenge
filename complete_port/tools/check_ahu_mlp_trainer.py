#!/usr/bin/env python3
"""Mixed-group async graph and role-Adam checks inside the trainer gate."""
import argparse, math, re, shlex, struct, subprocess
from pathlib import Path
from check_zone_residual_trainer import hadam
import os
PL_ROOT = os.environ.get("SB_PL_ROOT", os.path.expanduser("~/.cache/sb/pufferlib"))
CP=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--name',required=True);a=p.parse_args()
 base=CP/'results'/a.name
 if base.exists():raise RuntimeError('refusing to reuse smoke output')
 empty=subprocess.check_output(['python3',str(CP/'tools/zone_empty_obs.py'),str(CP/'artifacts/verify_multi'),'--count','3'],text=True).strip()
 args=['--env.blob_dir=/art/verify_multi','--env.groups=3','--env.canonical=1','--env.zone_memory=1',
  '--env.zone_critic=1','--env.max_steps=8','--env.desync=0','--env.reward_scale=1',
  '--policy.hidden_size=64','--policy.num_layers=1','--policy.tanh_mean=0','--policy.tanh_squash=0',
  '--policy.beta=1','--policy.zone_mlp=1','--policy.ahu_mlp=1','--policy.zone_value_residual=0',
  '--policy.beta_cn_tau=12','--policy.beta_cn_tau_ahu=48','--policy.beta_conc_max=12',
  '--policy.value_shared=1','--policy.value_delay=1','--train.zone_critic=1','--train.valve_zone=1',
  '--train.headwise_ratio=1','--train.adv_norm=2','--train.role_adam=1','--train.rew_clamp=0',
  '--train.horizon=64','--train.minibatch_size=256','--train.replay_ratio=2','--train.total_timesteps=6144',
  '--train.update_diagnostics=1','--base.async=1','--base.eval_episodes=0','--base.cudagraphs=1',
  '--base.checkpoint_optim=1','--base.checkpoint_interval=1','--vec.total_agents=24',
  f'--base.log_dir=/res/{a.name}',f'--base.checkpoint_dir=/res/{a.name}']
 cmd=['docker','run','--rm','--gpus',f'device={a.gpu}','-v',PL_ROOT + ":/work",
  '-v',f'{CP}/artifacts:/art','-v',f'{CP}/results:/res','-w','/work/plverify',
  '-e','SB_CREDIT=8','-e','SB_DETAIL_METRICS=1','-e','SB_ZONE_STATS=1','-e','SB_ZONE_EMPTY_OBS='+empty,
  'pufferai/puffertank:5.0','timeout','180','./puffer_shared','train',*args]
 done=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=200)
 (CP/'results'/(a.name+'.log')).write_text('COMMAND '+shlex.join(cmd)+'\n'+done.stdout)
 if done.returncode:raise RuntimeError(f'trainer failed rc={done.returncode}:\n'+'\n'.join(done.stdout.splitlines()[-30:]))
 log=done.stdout
 for token in ['shared AHU MLP on','AHU MLP on role Adam','SB_AHU_MLP ','SB_AP_ACTION ','SB_AP_BAD ','SB_ZONE_ACTION ','SB_HEAD_FIRST ']:
  if token not in log:raise RuntimeError('missing '+token)
 bad=[l for l in log.splitlines() if l.startswith(('SB_ZONE_BAD ','SB_AP_BAD '))]
 if not bad or any(float(l.split()[3])!=0 for l in bad):raise RuntimeError('invalid diagnostic values')
 rows=[l.split() for l in log.splitlines() if l.startswith('SB_AHU_MLP ')]
 if not rows or not any(float(x[3])>0 and float(x[4])>0 and float(x[5])>0 for x in rows):
  raise RuntimeError('AHU branch weights/gradients did not train')
 action=[l.split() for l in log.splitlines() if l.startswith('SB_AP_ACTION ')]
 seen={(int(x[2]),int(x[5])) for x in action if float(x[6])>0}
 if seen!={(g,h) for g in range(3) for h in range(7)}:raise RuntimeError(f'missing group/head telemetry {seen}')
 if any(not all(math.isfinite(float(v)) for v in x[6:]) for x in action):raise RuntimeError('nonfinite AHU/plant diagnostic')
 ckpts=sorted(base.rglob('*.bin'))
 if not ckpts:raise RuntimeError('no checkpoint')
 ck=ckpts[-1];mb=Path(str(ck)+'.mb');side=Path(str(ck)+'.hadam')
 if not side.is_file() or not mb.is_file() or mb.stat().st_size!=ck.stat().st_size:raise RuntimeError('missing optimizer sidecars')
 n,off,step=hadam(side);new_n=32*(64+32)+8*32;expected=4*(64+68)+8*(64+32)+6*(64+16)+32*(64+68)+4*32+new_n
 if n!=expected:raise RuntimeError(f'Adam ownership wrong: {n} != {expected}')
 raw=side.read_bytes();m=struct.unpack_from(f'<{n}f',raw,32);v=struct.unpack_from(f'<{n}f',raw,32+4*n)
 if not any(x!=0 for x in m[-new_n:]) or not any(x>0 for x in v[-new_n:]):raise RuntimeError('AHU Adam moments did not update')
 print('PASS AHU MLP mixed-group async graph, action telemetry and role-Adam checkpoint')
if __name__=='__main__':main()

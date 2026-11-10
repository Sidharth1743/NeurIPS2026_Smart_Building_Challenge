#!/usr/bin/env python3
"""Execute the user-approved, fixed 30M-per-arm training-distribution study.

prepare records the protocol without GPU work; run requires the full gate.
All evaluations use the same immutable fp64 runtime as both training arms.
No score-based stopping, schedule adaptation, extra trials, or controller.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

CP=Path(__file__).resolve().parents[1]
OUT=CP/'results/study_r24'
PARENT=CP/'results/runs/r22/ckpt/0000000024985600.bin'
BATCH=960*256
STEPS=122*BATCH
POINTS=[n*BATCH for n in [20,40,60,80,100,122]]
ARMS={'a':('r24a_single30m','r22_five',5),'b':('r24b_multi30m','tr23_shipped30',30)}
EXPECTED_PARENT='5e458eb0426a9c1a4fbf62e5bc00bc3bb2a59413f4180d62648b0c00a12a2f61'
EXPECTED_BINARY='f659f0396213463fbefd3a89f1825041975e6eed99a548d24b7e7d2bfef82836'

def now():return datetime.now(timezone.utc).isoformat()

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,obj):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def common_flags():
    return ['--vec.total_agents=960','--env.canonical=1','--env.merge_weathers=1',
     '--env.rew_norm=0','--env.rew_center=0','--env.reward_scale=1',
     '--env.desync=1','--env.max_steps=0','--policy.hidden_size=256','--policy.num_layers=2',
     '--policy.init_logstd=-1','--policy.logstd_anneal=0','--policy.freeze_logstd=0',
     '--policy.tanh_mean=1','--policy.tanh_squash=1','--policy.squash_chain_floor=0.25',
     '--train.learning_rate=0.0001','--train.anneal_lr=0','--train.ent_coef=0',
     '--train.anneal_ent_coef=0','--train.gamma=0.99','--train.gae_lambda=0.95',
     '--train.clip_coef=0.2','--train.vf_coef=0.5','--train.vf_clip_coef=10',
     '--train.max_grad_norm=0.5','--train.replay_ratio=4','--train.minibatch_size=2048',
     '--train.horizon=256','--train.adv_norm=2','--train.shuffle=1','--train.rew_clamp=0',
     '--train.action_repeat=1','--train.ratio_per_dim=1',f'--train.total_timesteps={STEPS}',
     '--base.checkpoint_interval=4','--base.seed=73','--base.deterministic=0',
     '--base.cudagraphs=1',f'--base.load_model_path=/res/{PARENT.relative_to(CP/"results")}']

def prepare():
    assert digest(PARENT)==EXPECTED_PARENT
    runtime=CP/'results/campaign_r23/fixed_runtime'
    assert digest(runtime/'puffer')==EXPECTED_BINARY
    old_manifest=json.loads((runtime/'manifest.json').read_text())
    old_gate=CP.parent/old_manifest['gate']
    assert old_gate.read_text().rstrip().endswith('ALL CHECKS PASSED')
    assert digest(old_gate)==old_manifest['gate_sha256']
    for name,_,_ in ARMS.values():assert not (CP/'results/runs'/name).exists(),name
    OUT.mkdir(exist_ok=False)
    for sub in ['eval','reference','runner_logs','sources']: (OUT/sub).mkdir()
    shutil.copytree(runtime,OUT/'runtime')
    source_suffixes={'.c','.h','.cu','.cuh','.inc','.py','.sh','.diff','.ini','.md','.txt','.mk'}
    for folder in ['gpu','csim','patches','tools','config']:
        for source in (CP/folder).rglob('*'):
            if not source.is_file() or 'baked' in source.parts or '__pycache__' in source.parts:continue
            if source.suffix not in source_suffixes and source.name!='Makefile':continue
            target=OUT/'sources'/source.relative_to(CP);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    hashes={str(p.relative_to(OUT/'sources')):digest(p) for p in (OUT/'sources').rglob('*') if p.is_file()}
    save(OUT/'source_hashes.json',hashes)
    datasets={};blob_hashes={}
    for dataset in ['r22_five','tr23_shipped30','evalcore3','eval9']:
        files=sorted((CP/'artifacts/sets'/dataset).glob('*.sbb'))
        assert len(files)=={'r22_five':5,'tr23_shipped30':30,'evalcore3':15,'eval9':45}[dataset]
        datasets[dataset]=[]
        for path in files:
            stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size)
            if key not in blob_hashes:blob_hashes[key]=digest(path)
            datasets[dataset].append({'name':path.name,'bytes':stat.st_size,'sha256':blob_hashes[key]})
    assert {p['name'] for p in datasets['r22_five']}<={p['name'] for p in datasets['tr23_shipped30']}
    for panel in ['r22_five','evalcore3']:
        src=CP/f'results/campaign_r23/eval/r22_25m_fixed_{panel}.json'
        d=json.loads(src.read_text());assert d['complete']
        shutil.copy2(src,OUT/'reference'/f'parent_{panel}.json')
    plan={'approved_utc':now(),'hypothesis':'Broader scenario experience improves transfer relative to equally long single-building continuation.',
     'parent':str(PARENT.relative_to(CP)),'parent_sha256':EXPECTED_PARENT,'binary_sha256':EXPECTED_BINARY,
     'steps_per_arm':STEPS,'epochs':122,'batch_steps':BATCH,'checkpoint_interval_epochs':4,
     'scheduled_evaluation_steps':POINTS,'seed':73,'common_flags':common_flags(),
     'arms':{a:{'run':name,'dataset':dataset,'scenarios':n} for a,(name,dataset,n) in ARMS.items()},
     'primary':'Mean evalcore3 score over the last three scheduled checkpoints; lower is better.',
     'secondary':['final score','worst-building score','last-window variability','comfort/energy/carbon','all curve points','wall time and physical SPS'],
     'stopping':'Fixed duration; only user stop or numerical/implementation invalidity. No score-based stopping.',
     'inference':'One paired seed is exploratory. No leaderboard or final-test claim.',
     'datasets':datasets,'training_gpu':0,'evaluation_gpu':1,'post_training_evaluation_gpus':[0,1]}
    save(OUT/'protocol.json',plan)
    jobs=[{'id':0,'arm':'parent','step':0,'dataset':'eval9','checkpoint':str(PARENT.relative_to(CP/'results')),'status':'pending'}]
    for arm,(name,_,_) in ARMS.items():
        for step in POINTS:
            for dataset in ['r22_five','evalcore3']:
                jobs.append({'id':len(jobs),'arm':arm,'step':step,'dataset':dataset,
                    'checkpoint':f'runs/{name}/ckpt/{step:016d}.bin','status':'pending'})
        jobs.append({'id':len(jobs),'arm':arm,'step':STEPS,'dataset':'eval9',
                     'checkpoint':f'runs/{name}/ckpt/{STEPS:016d}.bin','status':'pending'})
    save(OUT/'jobs.json',jobs)
    save(OUT/'state.json',{'phase':'prepared','created_utc':now(),'training_complete':False})
    print(f'Prepared immutable protocol: {STEPS} steps/arm, {len(jobs)} full evaluation jobs.',flush=True)

def eval_job(index,gpu):
    job=json.loads((OUT/'jobs.json').read_text())[index]
    path=CP/'results'/job['checkpoint'];assert path.stat().st_size==PARENT.stat().st_size
    label=f'{job["arm"]}_{job["step"]:08d}_{job["dataset"]}'
    target=OUT/'eval'/f'{label}.json';meta=target.with_suffix('.meta.json')
    assert not target.exists() and not meta.exists(),str(target)
    env=dict(os.environ,PUFFERLIB=str(OUT/'runtime'),SB_BIN='puffer',SB_CONTAINER_NAME=f'sb-r24-eval-g{gpu}-j{index}')
    if job['dataset']=='eval9':
        env['SB_EVAL_SPLITS']='sets/evalcore3 sets/evalr24_rest_a sets/evalr24_rest_b'
    command=[str(CP/'tools/eval_policy.sh'),f'sets/{job["dataset"]}',str(target.relative_to(CP/'results')),
      f'/res/{job["checkpoint"]}','--env.canonical=1','--env.residual=0','--policy.tanh_squash=1',
      '--base.cudagraphs=-1','--train.horizon=256','--base.seed=73']
    m={'job':job,'gpu':gpu,'started_utc':now(),'checkpoint_sha256':digest(path),
       'binary_sha256':digest(OUT/'runtime/puffer'),'episode_steps':8640,'deterministic':True,'command':command}
    if 'SB_EVAL_SPLITS' in env:m['evaluation_parts']=env['SB_EVAL_SPLITS'].split()
    save(meta,m);started=time.monotonic()
    code=subprocess.call(command,env=env)
    m.update(exit_code=code,wall_seconds=time.monotonic()-started,ended_utc=now())
    if code==0:
        d=json.loads(target.read_text());assert d['complete']
        assert d['n_scenarios_done']==len(list((CP/'artifacts/sets'/job['dataset']).glob('*.sbb')))
        m['score']=d['score']
    save(meta,m);raise SystemExit(code)

def run(gate):
    assert gate.read_text().rstrip().endswith('ALL CHECKS PASSED'),str(gate)
    state=json.loads((OUT/'state.json').read_text());assert state['phase']=='prepared',state
    assert digest(OUT/'runtime/puffer')==EXPECTED_BINARY and digest(PARENT)==EXPECTED_PARENT
    hashes=json.loads((OUT/'source_hashes.json').read_text())
    for relative,h in hashes.items():assert digest(CP/relative)==h,f'Source changed after protocol: {relative}'
    state.update(phase='running',started_utc=now(),gate=str(gate),gate_sha256=digest(gate),arms={})
    save(OUT/'state.json',state)
    jobs=json.loads((OUT/'jobs.json').read_text());lock=threading.Lock();stop=threading.Event();training_done=threading.Event()
    children=set();children_lock=threading.Lock()
    def handle_stop(signum,frame):stop.set()
    signal.signal(signal.SIGTERM,handle_stop);signal.signal(signal.SIGINT,handle_stop)
    def spawn(cmd,log):
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with children_lock:children.add(p)
        return p
    def reap(p):
        with children_lock:children.discard(p)
    def worker(gpu):
        while not stop.is_set():
            job=None
            with lock:
                for candidate in jobs:
                    path=CP/'results'/candidate['checkpoint']
                    if candidate['status']=='pending' and path.is_file() and path.stat().st_size==PARENT.stat().st_size and time.time()-path.stat().st_mtime>2:
                        job=candidate;job.update(status='running',gpu=gpu,started_utc=now());save(OUT/'jobs.json',jobs);break
                if job is None and training_done.is_set() and not any(j['status']=='pending' for j in jobs):return
            if job is None:stop.wait(3);continue
            with (OUT/'runner_logs'/f'job_{job["id"]:02d}.log').open('xb') as log:
                cmd=[str(CP/'tools/with_gpu.sh'),str(gpu),'--',sys.executable,str(Path(__file__).resolve()),'eval-job','--id',str(job['id']),'--gpu',str(gpu)]
                p=spawn(cmd,log)
                while p.poll() is None and not stop.wait(2):pass
                if stop.is_set():return
                code=p.wait();reap(p)
            with lock:
                job.update(status='complete' if code==0 else 'failed',exit_code=code,ended_utc=now());save(OUT/'jobs.json',jobs)
            print('EVALUATED',job['arm'],job['step'],job['dataset'],job['status'],flush=True)
    threads=[threading.Thread(target=worker,args=(1,),daemon=True)];threads[0].start()
    failed=False
    try:
        for arm,(name,dataset,n) in ARMS.items():
            if stop.is_set():break
            state['arms'][arm]={'phase':'training','started_utc':now()};save(OUT/'state.json',state)
            command=[sys.executable,str(CP/'tools/run_controlled.py'),name,'--gpu','0','--runtime',str(OUT/'runtime'),'--',
                     f'--env.blob_dir=/art/sets/{dataset}',f'--env.groups={n}',*common_flags()]
            print('TRAINING',arm,name,STEPS,flush=True)
            with (OUT/'runner_logs'/f'train_{arm}.log').open('xb') as log:
                p=spawn(command,log)
                while p.poll() is None and not stop.wait(3):pass
                if stop.is_set():break
                code=p.wait();reap(p)
            state['arms'][arm].update(phase='complete' if code==0 else 'failed',exit_code=code,ended_utc=now());save(OUT/'state.json',state)
            if code!=0:failed=True;stop.set();break
            for step in POINTS:
                model=CP/f'results/runs/{name}/ckpt/{step:016d}.bin'
                assert model.stat().st_size==PARENT.stat().st_size,str(model)
            print('TRAINING COMPLETE',arm,flush=True)
        if not stop.is_set():
            training_done.set();state.update(training_complete=True,phase='evaluating');save(OUT/'state.json',state)
            threads.append(threading.Thread(target=worker,args=(0,),daemon=True));threads[-1].start()
            while any(t.is_alive() for t in threads) and not stop.wait(3):pass
    except BaseException as exc:
        failed=True;stop.set();state['error']=repr(exc)
        raise
    finally:
        if stop.is_set():
            with children_lock:
                for child in children:
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
            names=[f'sb-{v[0]}' for v in ARMS.values()]+[f'sb-r24-eval-g{g}-j{j["id"]}' for g in [0,1] for j in jobs]
            active=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            owned=[n for n in names if n in active]
            if owned:subprocess.run(['docker','stop','--timeout','2',*owned],check=False)
            state['phase']='invalid' if failed else 'stopped'
        else:state['phase']='complete' if all(j['status']=='complete' for j in jobs) else 'evaluation_incomplete'
        state['ended_utc']=now();save(OUT/'state.json',state)
    print('STUDY',state['phase'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','eval-job'])
    p.add_argument('--gate',type=Path);p.add_argument('--id',type=int);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.gate)
    else:eval_job(a.id,a.gpu)

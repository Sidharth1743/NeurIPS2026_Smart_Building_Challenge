#!/usr/bin/env python3
"""Full-episode deterministic evaluation with provenance and no output overwrite."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

CP=Path(__file__).resolve().parents[1]

def main():
    cut=sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name')
    p.add_argument('dataset')
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--binary',default='puffer')
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--graphs',type=int,choices=[0,1],default=1)
    p.add_argument('--reserved',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(sys.argv[1:cut]);extra=sys.argv[cut+1:]
    if not a.reserved:
        # Start the measurement only after the shipped queue grants the GPU.
        raise SystemExit(subprocess.call([str(CP/'tools/with_gpu.sh'),str(a.gpu),'--',
                         sys.executable,str(Path(__file__).resolve()),
                         *sys.argv[1:cut],'--reserved',*sys.argv[cut:]]))
    output=CP/'results/campaign_r23/eval'/f'{a.name}_{a.dataset}.json'
    if output.exists():raise SystemExit(f'Preserving existing output: {output}')
    checkpoint=a.checkpoint.resolve();relative=checkpoint.relative_to(CP/'results')
    runtime=a.runtime.resolve();binary=runtime/a.binary
    digest=lambda path:hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    command=[str(CP/'tools/eval_policy.sh'),
             f'sets/{a.dataset}',str(output.relative_to(CP/'results')),f'/res/{relative}',
             '--env.canonical=1','--env.residual=0','--policy.tanh_squash=1',
             f'--base.cudagraphs={1 if a.graphs else -1}',*extra]
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'gpu':a.gpu,
              'binary_sha256':digest(binary),'checkpoint_sha256':digest(checkpoint),
              'command':command,'graphs':bool(a.graphs),'episode_steps':8640}
    sidecar=output.with_suffix('.meta.json')
    assert not sidecar.exists(),sidecar
    sidecar.write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic()
    code=subprocess.call(command,env=dict(os.environ,PUFFERLIB=str(runtime),SB_BIN=a.binary))
    manifest.update(exit_code=code,wall_seconds=time.monotonic()-start)
    if code==0:
        d=json.loads(output.read_text());assert d['complete']
        manifest['score']=d['score']
    sidecar.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()

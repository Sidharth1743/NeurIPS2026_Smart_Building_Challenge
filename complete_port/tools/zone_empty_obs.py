#!/usr/bin/env python3
"""Read one common normalized empty-building value from a set of SBB blobs."""
import argparse
import math
from pathlib import Path
import struct

HDR=struct.Struct('<4sII5Q')
SEC=struct.Struct('<IIII5Q')
def empty_obs(path):
    with path.open('rb') as f:
        h=f.read(HDR.size)
        if len(h)!=HDR.size: raise ValueError(f'{path}: short header')
        magic,version,n,do,no,nl,total,_=HDR.unpack(h)
        if magic!=b'SBB1' or version!=1: raise ValueError(f'{path}: bad blob header')
        if path.stat().st_size!=total: raise ValueError(f'{path}: wrong blob size')
        f.seek(no); names=f.read(nl)
        f.seek(do); directory=f.read(SEC.size*n)
        for i in range(n):
            off,dtype,ndim,_,s0,s1,s2,s3,data=SEC.unpack_from(directory,i*SEC.size)
            end=names.index(b'\0',off)
            if names[off:end]!=b'obs_meta':continue
            if dtype!=0 or ndim!=1 or s0<3: raise ValueError(f'{path}: invalid obs_meta')
            f.seek(data+16);raw=f.read(8)
            if len(raw)!=8: raise ValueError(f'{path}: short obs_meta')
            c=struct.unpack('<d',raw)[0]
            if not math.isfinite(c) or c<0: raise ValueError(f'{path}: invalid occupancy normalization')
            # Match float32 observation conversion exactly.
            return struct.unpack('<f',struct.pack('<f',-c/(c+1)))[0]
        raise ValueError(f'{path}: missing obs_meta')

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--count',type=int);a=p.parse_args()
    paths=sorted(a.directory.glob('*.sbb'))
    if not paths or (a.count is not None and len(paths)!=a.count):raise SystemExit(f'{a.directory}: wrong scenario count {len(paths)}')
    values=[empty_obs(path) for path in paths]
    if any(x!=values[0] for x in values[1:]):raise SystemExit(f'{a.directory}: occupancy normalizations differ: {values}')
    print(f'{values[0]:.9g}')
if __name__=='__main__':main()

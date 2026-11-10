#!/usr/bin/env python3
"""Print the last value of every scalar tag in a TensorBoard event directory (no tensorboard needed).
usage: tbread.py <log_dir>/tb [substring ...]   (with substrings: the full series of every matching tag)"""
import struct,sys,glob,collections
def varint(b,i):
    r=s=0
    while True:
        c=b[i]; i+=1; r|=(c&0x7f)<<s; s+=7
        if c<0x80: return r,i
def fields(b):
    i=0
    while i<len(b):
        k,i=varint(b,i); f,w=k>>3,k&7
        if w==0: v,i=varint(b,i)
        elif w==1: v=b[i:i+8]; i+=8
        elif w==5: v=b[i:i+4]; i+=4
        else: l,i=varint(b,i); v=b[i:i+l]; i+=l
        yield f,v
d=collections.defaultdict(list)
for fn in glob.glob(sys.argv[1]+'/events.out*'):
    b=open(fn,'rb').read(); i=0
    while i<len(b):
        l=struct.unpack('<Q',b[i:i+8])[0]; ev=b[i+12:i+12+l]; i+=12+l+4
        step=0; summ=None
        for f,v in fields(ev):
            if f==2: step=v
            if f==5: summ=v
        if summ:
            for f,v in fields(summ):
                tag=val=None
                for g,x in fields(v):
                    if g==1: tag=x.decode()
                    if g==2: val=struct.unpack('<f',x)[0]
                d[tag].append((step,val))
pats=sys.argv[2:]
if pats:
    for t in sorted(d):
        if any(p in t for p in pats): print(t, ' '.join(f'{s/1e6:.1f}M:{v:.4g}' for s,v in d[t]))
    sys.exit()
for t in sorted(d): print(f"{t:32s} n={len(d[t]):3d} last_step={d[t][-1][0]} last={d[t][-1][1]:.5g}")

#!/usr/bin/env python3
"""Check actual deterministic Beta-mean dispatch on a recorded gate episode."""
from array import array
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
body = json.loads((root / 'score.json').read_text())
assert body['complete'] and body['n_scenarios_done'] == 1
rec = root / 'record'
meta = (rec / 'meta.txt').read_text().splitlines()
for required in ['agents 1', 'obs 1024', 'act 288', 'loc beta_mean_action', 'conc alpha_plus_beta', 'shapes alpha_beta_float64']:
    assert required in meta, (required, meta)
def floats(name):
    v = array('f')
    with (rec / name).open('rb') as f:
        v.fromfile(f, (rec / name).stat().st_size // v.itemsize)
    return v
obs, act, done = floats('obs.bin'), floats('loc.bin'), floats('done.bin')
shapes = array('d')
with (rec / 'shapes.bin').open('rb') as f:
    shapes.fromfile(f, (rec / 'shapes.bin').stat().st_size // shapes.itemsize)
mask = (rec / 'mask.bin').read_bytes()
assert len(mask) == 288 and len(obs) % 1024 == 0 and len(act) % 288 == 0
frames = len(done)
assert frames >= 8 and len(obs) == frames * 1024 and len(act) == frames * 288
assert len(shapes) == frames * 288 * 2
assert not any(done[:7]) and done[7] != 0
zones = [z for z in range(90) if obs[9*z] > .5 and mask[2*z+1]]
assert zones
maximum = max(abs(obs[(t+1)*1024+9*z+6]-act[t*288+2*z+1]) for t in range(7) for z in zones)
assert maximum < 2e-5, f'dispatched action differs from recorded Beta mean: {maximum}'
for t in range(8):
    for z in zones:
        h=2*z+1
        alpha,beta=shapes[(t*288+h)*2:(t*288+h)*2+2]
        assert alpha>0 and beta>0
        mean=max(1e-12,min(.99999994,alpha/(alpha+beta)))
        assert abs(act[t*288+h]-(2*mean-1))<2e-6
print(f'PASS deterministic mean dispatch, {len(zones)} active reheat heads, max error {maximum:g}')

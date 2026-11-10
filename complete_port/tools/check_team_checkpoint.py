#!/usr/bin/env python3
"""Check the trainer's separately checkpointed team-baseline optimizer state."""
import math
from pathlib import Path
import struct
import sys

path = Path(sys.argv[1])
data = path.read_bytes()
assert len(data) >= 8
width, steps = struct.unpack_from('=ii', data)
assert width > 0 and steps > 0 and len(data) == 8 + 12 * width
values = struct.unpack_from(f'={3 * width}f', data, 8)
assert all(math.isfinite(x) for x in values)
assert any(x != 0 for x in values[:width]), 'baseline weights did not train'
assert any(x != 0 for x in values[width:2 * width]), 'Adam first moment missing'
assert any(x > 0 for x in values[2 * width:]), 'Adam second moment missing'
print(f'PASS team baseline checkpoint: width={width}, Adam steps={steps}, finite nonzero state')

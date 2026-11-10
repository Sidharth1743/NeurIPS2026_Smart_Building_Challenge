#!/bin/bash
SB_ROOT="${SB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [ ! -d "$SB_ROOT/complete_port" ] && [ "$SB_ROOT" != "/" ]; do SB_ROOT="$(dirname "$SB_ROOT")"; done
# Bake every scenario in training_set.yaml into csim/baked/<b>__<w>.sbb (skipping
# ones that already carry the observation sections). Run from anywhere.
set -u
SK=$SB_ROOT/starter-kit
OUT=$SB_ROOT/complete_port/csim/baked
PY=$SK/.venv/bin/python
mkdir -p "$OUT"
cd "$SK" || exit 1
PAIRS=$($PY - <<'PYEOF'
import yaml
d = yaml.safe_load(open("smart_buildings/training_set.yaml"))
for s in d["scenarios"]:
    print(s["building"], s["weather"])
PYEOF
)
echo "$PAIRS" | xargs -P "${BAKE_JOBS:-3}" -n 2 bash -c '
  b="$0"; w="$1"
  SK=$SB_ROOT/starter-kit
  OUT=$SB_ROOT/complete_port/csim/baked
  PY=$SK/.venv/bin/python
  cd "$SK" || exit 1
  if [ -f "$OUT/${b}__${w}.sbb" ]; then
    if $PY - "$OUT/${b}__${w}.sbb" <<"PYEOF"
import sys, struct
p = sys.argv[1]
with open(p, "rb") as f:
    hdr = f.read(64)
    n = struct.unpack_from("<i", hdr, 8)[0]
    no, nl = struct.unpack_from("<qq", hdr, 16)
    f.seek(no); names = f.read(nl)
sys.exit(0 if (b"n_dev" in names and b"aux_time" in names) else 1)
PYEOF
    then echo "skip $b/$w (fresh)"; exit 0; fi
  fi
  echo "bake $b/$w"
  ulimit -v 9000000
  timeout 1800 $PY $SB_ROOT/complete_port/pysim/bake_scenario.py \
      --out "$OUT" --name "${b}__${w}" \
      "smart_buildings/training_scenarios/${b}/${w}/config.gin" || echo "FAIL $b/$w"
'
echo "bake_all done: $(ls $OUT/*.sbb | wc -l) blobs"

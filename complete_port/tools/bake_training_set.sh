#!/bin/bash
# Bake the 30 shipped training scenarios (starter-kit training_set.yaml) into artifacts/train/
# (one .sbb each, ~140 MB; this directory is the default multi-building training set).
# Idempotent; BAKE_JOBS (default 3) bakes in parallel.
set -e
. "$(dirname "$0")/common.sh"
OUT="$ART/train"; mkdir -p "$OUT"
PY="$SK/.venv/bin/python"
export CP SK ART OUT PY
( cd "$SK" && $PY - <<'PYEOF'
import yaml
for s in yaml.safe_load(open("smart_buildings/training_set.yaml"))["scenarios"]:
    print(s["building"], s["weather"])
PYEOF
) | xargs -P "${BAKE_JOBS:-3}" -n 2 bash -c '
  b="$0"; w="$1"; name="${b}__${w}"
  [ -f "$OUT/$name.sbb" ] && exit 0
  if [ -f "$ART/blobs/$name.sbb" ]; then ln -f "$ART/blobs/$name.sbb" "$OUT/$name.sbb" 2>/dev/null || cp "$ART/blobs/$name.sbb" "$OUT/$name.sbb"; exit 0; fi
  echo "bake $name"
  ulimit -v 16000000
  ( cd "$SK" && $PY "$CP/pysim/bake_scenario.py" --out "$OUT" --name "$name" "smart_buildings/training_scenarios/$b/$w/config.gin" >/dev/null )
'
echo "training set: $(ls "$OUT"/*.sbb | wc -l) blobs in $OUT"

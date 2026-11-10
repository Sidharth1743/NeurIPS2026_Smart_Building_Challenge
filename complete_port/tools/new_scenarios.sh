#!/bin/bash
# Build as many NEW buildings / scenarios as you like with the starter-kit's own generators and
# bake them for the GPU env - nothing in the starter-kit tree is modified (outputs go to artifacts/gen).
#   new_scenarios.sh --seeds "500 501 502" --weathers "fairbanks_1 phoenix_5" [--verify]
#     --seeds     building seeds for run_floor_generator (default 500)
#     --weathers  weather presets from smart_buildings/weather (default fairbanks_1)
#     --verify    also dump an oracle trace, a CPU multi-episode reference and run the GPU gates
# Result: artifacts/blobs/<building>__<weather>.sbb (what ./puffer train --env.blob=... loads).
# Limits of the GPU env (it says so at start-up when exceeded): <= 1024 observation values,
# <= 16 discrete and <= 256 continuous actions (build with SB_CONT_BIG=1 for <= 320), <= 256 AHUs.
set -e
. "$(dirname "$0")/common.sh"
SEEDS="500"; WEATHERS="fairbanks_1"; VERIFY=0
while [ $# -gt 0 ]; do case "$1" in --seeds) SEEDS="$2"; shift 2;; --weathers) WEATHERS="$2"; shift 2;; --verify) VERIFY=1; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
PY="$SK/.venv/bin/python"; G="$ART/gen"; mkdir -p "$G/buildings" "$G/scenarios"
ulimit -v 28000000
( cd "$SK" && $PY -m smart_buildings.run_floor_generator --seeds $SEEDS --no-pdf --output-dir "$G/buildings" >/dev/null )
BL=""; for s in $SEEDS; do BL="$BL $G/buildings/building_$(printf %04d $s).yaml"; done
( cd "$SK" && $PY -m smart_buildings.run_scenario_generator --buildings $BL --weathers $WEATHERS --output-dir "$G/scenarios" >/dev/null )
for s in $SEEDS; do for w in $WEATHERS; do
  b=building_$(printf %04d $s); name=${b}__${w}; cfg="$G/scenarios/$b/$w/config.gin"
  [ -f "$ART/blobs/$name.sbb" ] || ( cd "$SK" && $PY "$CP/pysim/bake_scenario.py" --out "$ART/blobs" --name "$name" "$cfg" | tail -1 )
  python3 - "$ART/blobs/$name.json" "$name" <<'PYEOF'
import json,sys
j=json.load(open(sys.argv[1]))
print("%s: grid %s, %d zones, obs %d, actions %d (%d cont / %d disc)" % (sys.argv[2], j['grid'], j['n_phys_zones'], j['n_field'], j['n_action'], j['n_cont'], j['n_disc']))
PYEOF
  if [ $VERIFY = 1 ]; then
    [ -f "$ART/traces/$name.step" ] || ( cd "$SK" && $PY "$CP/pysim/dump_step.py" "$cfg" "$ART/traces/$name.step" 20 >/dev/null )
    [ -f "$ART/refs/$name.ref" ] || "$CP/csim/test_episodes" "$ART/blobs/$name.sbb" "$ART/traces/$name.step" 2 "$ART/refs/$name.ref" 150 >/dev/null
    nc=$(python3 -c "import json;print(json.load(open('$ART/blobs/$name.json'))['n_cont'])")
    GATE=gate; [ "$nc" -gt 256 ] && GATE=gate_big
    [ -x "$ART/bin/$GATE" ] || { [ $GATE = gate_big ] && SB_CONT_BIG=1 "$CP/tools/build.sh" >/dev/null || "$CP/tools/build.sh" >/dev/null; }
    dock "B=artifacts/blobs/$name.sbb; T=artifacts/traces/$name.step; R=artifacts/refs/$name.ref; \
      artifacts/bin/$GATE \$B \$T 8 | grep -E 'reset obs|SB GPU ENV STEP'; \
      artifacts/bin/$GATE \$B \$T 4 \$R | grep -E 'MULTI-EPISODE'"
  fi
done; done

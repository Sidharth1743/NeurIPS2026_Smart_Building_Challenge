#!/bin/bash
# Regenerate everything the gates consume from the UNMODIFIED starter-kit: baked blobs,
# oracle step traces (30 steps; 40 for the non-train_1 shapes) and the CPU multi-episode
# references (csim/test_episodes).  Idempotent: existing files are kept.
set -e
. "$(dirname "$0")/common.sh"
PY="$SK/.venv/bin/python"
( cd "$CP/csim" && make test_episodes >/dev/null )
ulimit -v 16000000
for bw in $SHAPES; do
  b=${bw%/*}; w=${bw#*/}; name=${b}__${w}
  cfg="$SK/smart_buildings/training_scenarios/$b/$w/config.gin"
  [ -f "$ART/blobs/$name.sbb" ] || { echo "bake $name"; ( cd "$SK" && $PY "$CP/pysim/bake_scenario.py" --out "$ART/blobs" --name "$name" "$cfg" >/dev/null ); }
  steps=40; [ "$name" = building_train_1__fairbanks_1 ] && steps=30
  [ -f "$ART/traces/$name.step" ] || { echo "trace $name"; ( cd "$SK" && $PY "$CP/pysim/dump_step.py" "$cfg" "$ART/traces/$name.step" $steps >/dev/null ); }
  eps=300; [ "$name" = building_train_1__fairbanks_1 ] && eps=900
  [ -f "$ART/refs/$name.ref" ] || { echo "cpu reference $name"; "$CP/csim/test_episodes" "$ART/blobs/$name.sbb" "$ART/traces/$name.step" 3 "$ART/refs/$name.ref" $eps >/dev/null; }
done
# every weather station of two buildings (blob + 30-step oracle trace): the weather-merge gate
# (one group = one building with all its weathers; train_1's stations span several time zones)
mkdir -p "$ART/traces_w"
for b in building_train_1 building_train_8; do
  for wd in "$SK/smart_buildings/training_scenarios/$b"/*/; do
    w=$(basename "$wd"); name=${b}__${w}; cfg="$wd/config.gin"
    [ -f "$ART/blobs/$name.sbb" ] || { echo "bake $name"; ( cd "$SK" && $PY "$CP/pysim/bake_scenario.py" --out "$ART/blobs" --name "$name" "$cfg" >/dev/null ); }
    [ -f "$ART/traces_w/$name.step" ] || { echo "trace $name"; ( cd "$SK" && $PY "$CP/pysim/dump_step.py" "$cfg" "$ART/traces_w/$name.step" 30 >/dev/null ); }
  done
done
echo artifacts ok

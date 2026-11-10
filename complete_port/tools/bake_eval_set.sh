#!/bin/bash
# Bake the blobs of an evaluation set.
#
#   tools/bake_eval_set.sh <set list> [jobs]
#
# <set list> is a text file with one scenario per line, "<building>__<weather>.sbb", as shipped
# in data/scenarios/sets/. A scenario needs a config.gin, and the organiser ships configs only
# for the 30 training scenarios, so any pair a set adds is generated first with the organiser's
# own generator (a cross product of the buildings and the weather strips named in the list).
# Then each scenario is packed into a .sbb blob, which is what the GPU env reads.
#
# Blobs land in complete_port/artifacts/blobs; tools/bake_training_set.sh does the same for the
# 30 training scenarios. Idempotent: an existing blob is left alone.
set -euo pipefail
. "$(dirname "$0")/common.sh"

LIST="${1:?usage: bake_eval_set.sh <set list> [jobs]}"
JOBS="${2:-2}"
[ -f "$LIST" ] || { echo "no scenario list $LIST" >&2; exit 1; }
PY="$SK/.venv/bin/python"
[ -x "$PY" ] || { echo "no virtualenv at $SK/.venv: run ./run.sh setup" >&2; exit 1; }

NAMES=$(sed -e 's/[[:space:]]*$//' -e 's/\.sbb$//' "$LIST" | grep -v '^$' | sort -u)
N=$(printf '%s\n' "$NAMES" | wc -l)
echo "set: $N scenarios from $LIST"

# 0. a building that the starter kit does not ship. Two of the five buildings in the published
#    panels are generated ones (building_train_3 is floor-generator seed 1018, building_train_6 is
#    1012); their YAML configs ship in data/generated_buildings/, so install any missing one.
GEN_BUILDINGS="${SB_GEN_BUILDINGS:-$CP/../data/generated_buildings}"
for b in $(printf '%s\n' "$NAMES" | awk -F'__' '{print $1}' | sort -u); do
  if [ ! -f "$SK/smart_buildings/buildings/$b.yaml" ] && [ -f "$GEN_BUILDINGS/$b.yaml" ]; then
    cp "$GEN_BUILDINGS/$b.yaml" "$SK/smart_buildings/buildings/$b.yaml"
    echo "installed $b from data/generated_buildings (not one of the six the starter kit ships)"
  fi
done

# 1. config.gin for any pair the starter kit does not ship, one generator call per building
MISSING_BUILDINGS=$(printf '%s\n' "$NAMES" | while read -r name; do
  b="${name%%__*}"; w="${name#*__}"
  [ -f "$SK/smart_buildings/training_scenarios/$b/$w/config.gin" ] || echo "$b"
done | sort -u)

for b in $MISSING_BUILDINGS; do
  WEATHERS=$(printf '%s\n' "$NAMES" | awk -v b="$b" -F'__' '$1 == b {print $2}' | sort -u | tr '\n' ' ')
  echo "generate: $b with $(echo $WEATHERS | wc -w) weather strips"
  ( cd "$SK" && "$PY" -m smart_buildings.run_scenario_generator \
      --buildings "$b" --weathers $WEATHERS >/dev/null )
done

# 2. bake the blobs
export ART SK PY CP
printf '%s\n' "$NAMES" | xargs -P "$JOBS" -n 1 bash -c '
  name="$1"; b="${name%%__*}"; w="${name#*__}"
  [ -f "$ART/blobs/$name.sbb" ] && exit 0
  ( cd "$SK" && ulimit -v 14000000 2>/dev/null
    "$PY" "$CP/pysim/bake_scenario.py" --out "$ART/blobs" --name "$name" \
      "smart_buildings/training_scenarios/$b/$w/config.gin" >/dev/null ) \
    && echo "baked $name" || echo "FAILED $name"
' _

echo "blobs: $(ls "$ART"/blobs/*.sbb 2>/dev/null | wc -l) in $ART/blobs"

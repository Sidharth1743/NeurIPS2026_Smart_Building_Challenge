#!/bin/bash
# Batch-layout A/B: the same blobs, one envs-per-scenario count per run, so the layout is chosen from measured
# throughput instead of theory.  Each run is a short tools/run_train.sh smoke:
#   envs/scenario E -> --vec.total_agents = E * (#blobs) -> group = 5*E envs (one weather per 32-env slab iff E % 32 == 0)
# The table reports steady-state epoch SPS, us/env-step, ms/step, VRAM and the solver's value mode (the env's
# startup line).  The runs have different batch sizes (agents = E x blobs), so compare us/env-step.
#
#   tools/bench_layout.sh <set> <E> [<E> ...] [-- extra ./puffer train args]
#   tools/bench_layout.sh rot2b 48 32 64 --env.canonical=1 --env.residual=1 --env.guard_d0=1 --env.guard_d1=2
set -eu
. "$(dirname "$0")/common.sh"
SET="$1"; shift
COUNTS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do COUNTS+=("$1"); shift; done
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA="$*"
STEPS="${STEPS:-800000}"
D="$ART/sets/$SET"
[ -d "$D" ] || { echo "no set $D"; exit 2; }
NG=$(ls "$D"/*.sbb | wc -l)
NGRP=$(ls "$D"/*.sbb | sed 's/__.*//' | sort -u | wc -l)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$CP/results/perf/layout_${SET}_$STAMP.tsv"
say() { echo "$@"; }
printf "set\tenvs_per_scenario\ttotal_agents\tgroups\tmode\trpl\tsps\tus_per_env_step\tms_per_step\tvram_gb\tepochs\n" > "$OUT"
for E in "${COUNTS[@]}"; do
    A=$((E * NG)); NAME="layout_${SET}_e$E"
    say "=== $SET: $E envs/scenario x $NG scenarios = $A agents, $NGRP groups"
    if ! "$CP/tools/run_train.sh" "$NAME" --env.blob_dir="/art/sets/$SET" --env.groups="$NG" --vec.total_agents="$A" \
         --env.max_steps=256 --train.total_timesteps="$STEPS" --policy.zero_init_actions=1 \
         --train.anneal_lr=0 --train.learning_rate=0.0002 $EXTRA > "$CP/results/runs/$NAME.out" 2>&1; then
        say "  run failed (see results/runs/$NAME.out)"; continue
    fi
    LOG="$CP/results/runs/$NAME/train.log"; TB="$CP/results/runs/$NAME/tb"
    MODE=$(grep -o 'value mode [0-9]' "$LOG" | head -1 | grep -o '[0-9]' || true)
    RPL=$(grep -o '[0-9]* env(s)/lane' "$LOG" | head -1 | grep -o '^[0-9]*' || true)
    VRAM=$(grep -o 'VRAM: *[0-9.]*/24G' "$LOG" | tail -1 | grep -o '^VRAM: *[0-9.]*' | grep -o '[0-9.]*$' || true)
    SPS=$(python3 "$CP/tools/tbread.py" "$TB" SPS | sed 's/^SPS *//' | tr ' ' '\n' \
          | awk -F: 'NF==2{print $2}' | python3 -c '
import sys, statistics
v = [float(x) for x in sys.stdin if float(x) > 0]
if len(v) > 4: v = v[2:-1]        # drop startup epochs and the epoch cut off by the run ending
elif len(v) > 1: v = v[:-1]
print(f"{statistics.median(v):.0f}" if v else 0)')
    EPOCHS=$(python3 "$CP/tools/tbread.py" "$TB" SPS | tr -cd ' ' | wc -c)
    [ "${SPS:-0}" != "0" ] || { say "  no SPS series in $TB"; continue; }
    US=$(python3 -c "print(f'{1e6/float($SPS):.1f}')")
    MS=$(python3 -c "print(f'{$A*1e3/float($SPS):.1f}')")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$SET" "$E" "$A" "$NGRP" "${MODE:-?}" "${RPL:-?}" \
           "$SPS" "$US" "$MS" "${VRAM:-?}" "$EPOCHS" >> "$OUT"
    say "  E=$E agents=$A mode=$MODE rpl=$RPL SPS=$SPS VRAM=${VRAM}G -> $US us/env-step, $MS ms/step"
done
say
column -t -s $'\t' "$OUT" 2>/dev/null || cat "$OUT"
say "TSV: $OUT"

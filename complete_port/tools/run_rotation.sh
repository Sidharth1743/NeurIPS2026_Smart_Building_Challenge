#!/bin/bash
# Building-rotation training: more training buildings than one batch can hold in VRAM.  The run is a sequence of
# segments; segment i trains on blob set SETS[i mod n] (artifacts/sets/<set>) for STEPS steps, starting from the
# previous segment's final checkpoint (--base.load_model_path; optimizer state starts cold each segment).
# Each segment is a tools/run_train.sh run named <name>_sNN (results/runs/<name>_sNN/); the chain of final checkpoints
# is listed in results/runs/<name>.chain.
#   tools/run_rotation.sh <name> <steps per segment> <segments> <envs per scenario> "<set> <set> ..." [./puffer train args...]
#   START=<checkpoint.bin> FIRST=<segment index> to continue a chain.
# The first segment (without START) gets --policy.zero_init_actions=1 (the agent starts as the base controller).
set -e
. "$(dirname "$0")/common.sh"
NAME="$1"; STEPS="$2"; NSEG="$3"; EPS="$4"; read -r -a SETS <<< "$5"; shift 5
CHAIN="$CP/results/runs/$NAME.chain"
PREV="${START:-}"
for ((i = ${FIRST:-0}; i < NSEG; i++)); do
    SET="${SETS[$((i % ${#SETS[@]}))]}"
    NG=$(ls "$ART/sets/$SET" | grep -c '\.sbb$')
    AGENTS=$((EPS * NG))      # VRAM ~ sum over buildings of size x envs: a fixed count per scenario (48 fits every rot set)
    SEG=$(printf "%s_s%02d" "$NAME" "$i")
    INIT="--policy.zero_init_actions=1"
    [ -n "$PREV" ] && INIT="--policy.zero_init_actions=0 --base.load_model_path=/res/runs/${PREV#$CP/results/runs/}"
    "$CP/tools/run_train.sh" "$SEG" --env.blob_dir="/art/sets/$SET" --env.groups="$NG" --vec.total_agents="$AGENTS" \
        --train.total_timesteps="$STEPS" $INIT "$@" > "$CP/results/runs/$SEG.out" 2>&1
    LAST=$(ls "$CP/results/runs/$SEG/ckpt/" 2>/dev/null | sort | tail -1)
    if [ -z "$LAST" ]; then echo "segment $SEG produced no checkpoint (see results/runs/$SEG/train.log)"; exit 1; fi
    PREV="$CP/results/runs/$SEG/ckpt/$LAST"
    echo "$i $SET $PREV" >> "$CHAIN"
done
echo "final: $PREV"

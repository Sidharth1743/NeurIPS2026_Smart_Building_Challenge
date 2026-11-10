#!/bin/bash
# Pure-RL chain, honest split (canonical layout, no residual controller): trains from scratch in a chain of
# segments and scores every finished segment on a held-out building subset while the next segment trains.
#
#   tools/run_pure_chain.sh <name> <steps per segment> "<set> <set> ..." [extra ./puffer train args...]
#   START=<checkpoint.bin> FIRST=<segment index> to continue a chain midway.
#   EVAL_FOLLOW=0 skips the scores.
#
# HONEST HOLDOUT: the training sets are artifacts/sets/tr36_s1..s6 (36 buildings), the scores are
# artifacts/sets/evalcore3 (3 buildings / 15 scenarios, the bar for monitoring) and eval9a + eval9b (the
# 9 evaluation buildings / 45 scenarios, merged into one score).  Building 30000-30019 never appears
# anywhere; tools/check_split.py proves training and evaluation share no file, no (building, weather)
# name and (deep) no content, and refuses the sealed range.  It runs before every segment.
#
# Per segment, on GPU 1 (tools/eval_policy.sh, the exact fp64 binary): evalcore3 (held out, the honest
# signal, the bar for monitoring) and the segment's own training set (in-sample, no bar - a flat held-out
# curve next to a rising in-sample one is the r7 failure signature) while the next segment trains on
# GPU 0.  After the last segment the two halves of the evaluation split are scored at the same time -
# eval9a on GPU 0, eval9b on GPU 1 - and merged by tools/merge_eval_json.py.  Scores are printed against
# the pid_water bar (lower is better).
#
# Each segment tries environments-per-scenario from ENVS_HI (default 20, or $ENVS if set) down by
# ENVS_STEP (default 2) to ENVS_LO (default 8).  A count whose agent total is not divisible by 8
# (minibatch_size / horizon) is skipped: the trainer rejects it before any step.  A resource failure
# (graph capture, out of memory, convection launch) retries the next lower legal count.  A building
# that exceeds the action or observation limit stops the chain; fewer environments will not load it.
# The search starts again at ENVS_HI for every segment, so a later, smaller set can still use 32
# after an earlier set had to step down.
#
# Segment k trains on blob set SETS[k mod n] (artifacts/sets/<set>) for STEPS steps.  Whole buildings
# per segment (merge_weathers=1) so a building's weathers share one group.  Every segment:
#   env.rew_norm=1 reward_scale=1        per-scenario running reward normalization (learning signal only)
#   train.adv_norm=2                     standardize advantages over the WHOLE rollout, not per minibatch
#                                        (a minibatch is 8 agent rows of one scenario, so per-minibatch
#                                        statistics recentre each building against itself and erase the
#                                        cross-building advantage scale; mode 2 keeps it)
#   train.shuffle=1                      visit the agent blocks in a fresh permutation each update
#                                        (the plain trainer walks the batch in rollout order, which is
#                                        building by building)
#   train.rew_clamp=0                    do not clamp the learning reward to +/-1 sigma: the z-tail the
#                                        policy must avoid reaches GAE (the env logs live/rew_z_gt*_frac)
#   train.action_repeat=HOLD             held-action exploration (default HOLD=4)
#   policy.logstd_anneal=0 init_logstd=-1  constant sigma = 0.37: the r7 schedule annealed to 0.05
#                                        (entropy ~30 over ~219 active heads, no exploration left)
#   base.checkpoint_optim=1              checkpoints carry the optimizer momentum (.mb sidecar)
#   train.gamma=0.999 gae_lambda=0.98 learning_rate=1e-4 anneal_lr=0   (LR_GAMMA/LR_LAMBDA/LR env overrides)
# Segment k>1 starts from the previous segment's final checkpoint + momentum.
# Extra args are appended and win (later --key=value overrides earlier ones in the trainer's ini).
set -e
. "$(dirname "$0")/common.sh"
NAME="$1"; STEPS="$2"; read -r -a SETS <<< "$3"; shift 3
mkdir -p "$CP/results/runs"
exec > >(tee -a "$CP/results/runs/$NAME.chain.log") 2>&1
echo "=== $NAME chain start $(date -Iseconds) steps/segment=$STEPS sets=${SETS[*]} ==="
# 20 environments per scenario is the measured memory ceiling for the tr36 sets on the 3090: the
# environment (building state + LU work arrays) scales with envs x building size, 30 scenarios at
# 20 envs (600 agents) peak at 22.9/24 GB and at 24 envs (720 agents) 23.5/24 GB, while the
# 27-scenario set OOMs at 24 envs (648 agents).  Throughput is the same at 20 and 24 (3.9K SPS);
# only the lane count drops (100 vs 120 of the 160 a full group wants), so 20 is the safer start.
ENVS_HI="${ENVS_HI:-${ENVS:-20}}"
ENVS_LO="${ENVS_LO:-8}"
ENVS_STEP="${ENVS_STEP:-2}"
HOLD="${HOLD:-4}"           # action hold
LR="${LR:-0.0001}"
GAMMA="${GAMMA:-0.999}"
LAMBDA="${LAMBDA:-0.98}"
CORE3="${CORE3:-evalcore3}" # per-segment monitoring subset (3 buildings, 15 scenarios)
BAR_CORE3="${BAR_CORE3:-0.0508}"
EVAL9A="${EVAL9A:-eval9a}"  # final split: 5 buildings on GPU 0, 4 on GPU 1
EVAL9B="${EVAL9B:-eval9b}"
BAR_EVAL9="${BAR_EVAL9:-0.0555}"
PL="${PUFFERLIB:-${SB_PL_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/sb/pufferlib}/plverify}"
TRAIN_GPU="${TRAIN_GPU:-0}"  # training is pinned to one device (run_train.sh would otherwise ask for both)
CHAIN="$CP/results/runs/$NAME.chain"
PREV="${START:-}"
NSEG=${#SETS[@]}
# minibatch_size / horizon in config/smart_buildings.ini (2048 / 256). The trainer asserts this.
BATCH_DIV=8
FOLLOW_PID=""

resource_fail() {
    local f="$1"
    [ -f "$f" ] || return 1
    grep -qE 'cudaGraphInstantiate|out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaMalloc failed|conv launch:|operation failed due to a previous error during capture' "$f"
}

# The scorer is the default (fp64 solve) build under its own name, so a training variant never
# clobbers it.  "./puffer" is the same build; the fp32 solve is detected by its cuSOLVER symbol.
ensure_eval_bin() {
    if [ ! -x "$PL/puffer_f64" ] && [ -x "$PL/puffer" ]; then
        echo "[$(date +%H:%M:%S)] $PL/puffer_f64 missing: copying ./puffer (same default build)"
        cp -f "$PL/puffer" "$PL/puffer_f64"
    fi
    if [ ! -x "$PL/puffer_f64" ]; then
        echo "[$(date +%H:%M:%S)] building $PL/puffer_f64 (default fp64-solve build)"
        docker run --rm --gpus device=0 -v "$PL":/work -w /work "$IMAGE" bash -lc \
            './build.sh smart_buildings >/dev/null 2>&1 && cp -f ./puffer ./puffer_f64' \
            || { echo "could not build $PL/puffer_f64"; return 1; }
    fi
    if grep -aq 'lu_solve_t.*EEv5LuDevPf' "$PL/puffer_f64"; then
        echo "$PL/puffer_f64 is the fp32-solve variant: the score would not be the exact one"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] eval binary: $PL/puffer_f64 (fp64 solve)"
}

# Training must not contain evaluation.  check_split.py exits non-zero on a shared file, a shared
# (building, weather) or the sealed 30000-30019 range; --deep also hashes content.
leak_check() {
    if ! python3 "$CP/tools/check_split.py" --quiet; then
        echo "split leak: refusing to train (see complete_port/tools/check_split.py --deep)"
        exit 1
    fi
}

# One deterministic fp64 score. The model path is the host checkpoint; the container sees it under /res.
score_one() {
    local gpu="$1" set="$2" out="$3" ckpt="$4" rel
    rel="${ckpt#$CP/results/runs/}"
    SB_BIN=puffer_f64 SB_GPU="$gpu" "$CP/tools/with_gpu.sh" "$gpu" -- \
        "$CP/tools/eval_policy.sh" "$set" "$out" "/res/runs/$rel" \
        --env.canonical=1 --env.residual=0
}

# Print a written score against its bar (lower is better).
report_score() {
    local json="$1" bar="$2" label="$3"
    [ -f "$CP/results/$json" ] || { echo "  $label: no $json"; return 0; }
    python3 - "$CP/results/$json" "$bar" "$label" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
bar = float(sys.argv[2]); s = float(d["score"])
extra = "vs bar %.6f (%.2fx)" % (bar, s / bar) if bar > 0 else "(in-sample, no bar)"
print("  %-22s score %.6f %s  %d scenarios, complete %s"
      % (sys.argv[3], s, extra, d.get("n_scenarios_done", 0), d.get("complete")))
EOF
}

# Score one finished segment: the held-out monitoring subset (core3, the honest signal) and then the
# segment's own training set (in-sample: an in-sample curve rising while core3 stays flat is exactly
# the r7 failure mode, so the pair is the cheapest way to see it).  Both on the same GPU, in sequence.
score_line() {
    local gpu="$1" line="$2" idx segset ckpt out
    idx=$(echo "$line" | awk '{print $1}')
    segset=$(echo "$line" | awk '{print $2}')
    ckpt=$(echo "$line" | awk '{print $3}')
    out=$(printf 'eval/%s_s%02d_%s.json' "$NAME" "$idx" "$CORE3")
    echo "[$(date +%H:%M:%S)] score segment $idx: $CORE3 (held out) on GPU $gpu"
    score_one "$gpu" "sets/$CORE3" "$out" "$ckpt" \
        && report_score "$out" "$BAR_CORE3" "segment $idx held-out" \
        || echo "[$(date +%H:%M:%S)] $CORE3 score failed for segment $idx"
    out=$(printf 'eval/%s_s%02d_insample_%s.json' "$NAME" "$idx" "$segset")
    echo "[$(date +%H:%M:%S)] score segment $idx: $segset (in-sample) on GPU $gpu"
    score_one "$gpu" "sets/$segset" "$out" "$ckpt" \
        && report_score "$out" 0 "segment $idx in-sample" \
        || echo "[$(date +%H:%M:%S)] in-sample score failed for segment $idx"
}

follow_evals() {
    local seen=0 n=0 line a b last
    while true; do
        n=$(grep -c . "$CHAIN" 2>/dev/null) || n=0   # missing/empty chain file: 0, not "" (line 194)
        if [ -f "${CHAIN}.done" ]; then
            while [ "$seen" -lt "$n" ]; do
                a=$(sed -n "$((seen + 1))p" "$CHAIN")
                if [ $((seen + 1)) -lt "$n" ]; then
                    b=$(sed -n "$((seen + 2))p" "$CHAIN")
                    score_line 1 "$a" &
                    score_line 0 "$b" &
                    wait || true
                    seen=$((seen + 2))
                else
                    score_line 0 "$a"
                    seen=$((seen + 1))
                fi
            done
            last=$(awk 'NF{print $3}' "$CHAIN" | tail -1)
            echo "[$(date +%H:%M:%S)] final checkpoint $last"
            echo "[$(date +%H:%M:%S)] final: $EVAL9A (GPU 0) + $EVAL9B (GPU 1) at the same time"
            sleep 5
            score_one 0 "sets/$EVAL9A" "eval/${NAME}_final_${EVAL9A}.json" "$last" &
            score_one 1 "sets/$EVAL9B" "eval/${NAME}_final_${EVAL9B}.json" "$last" &
            wait || echo "a final half failed"
            if python3 "$CP/tools/merge_eval_json.py" "$CP/results/eval/${NAME}_final_eval9.json" \
                    "$CP/results/eval/${NAME}_final_${EVAL9A}.json" \
                    "$CP/results/eval/${NAME}_final_${EVAL9B}.json"; then
                report_score "eval/${NAME}_final_eval9.json" "$BAR_EVAL9" "eval9"
            else
                echo "merge of the two evaluation halves failed (one half is missing or incomplete)"
            fi
            echo "[$(date +%H:%M:%S)] scores finished"
            return 0
        fi
        if [ "$seen" -lt "$n" ]; then
            line=$(sed -n "$((seen + 1))p" "$CHAIN")
            score_line 1 "$line"
            seen=$((seen + 1))
        else
            sleep 10
        fi
    done
}

# Last value of the tags that say whether the run is honest and healthy.
seg_report() {
    local seg="$1" tb="$CP/results/runs/$1/tb" v
    [ -d "$tb" ] || return 0
    echo "[$(date +%H:%M:%S)] $seg final scalars:"
    for t in loss/entropy loss/clipfrac loss/explained_variance loss/adv_abs_mean norm/adv_mean \
             norm/adv_inv_std live/rew_z_gt1_frac live/rew_z_gt3_frac live/rew_z_gt5_frac \
             live/worst_dev_K loss/value; do
        v=$(python3 "$CP/tools/tbread.py" "$tb" "$t" 2>/dev/null | tail -1 | sed 's/.* //')
        [ -n "$v" ] && printf '    %-26s %s\n' "$t" "$v"
    done
}

ensure_eval_bin
leak_check
python3 "$CP/tools/check_split.py" --deep --quiet && echo "[$(date +%H:%M:%S)] split check (deep): clean" \
    || { echo "deep split check failed"; exit 1; }

if [ "${EVAL_FOLLOW:-1}" = 1 ]; then
    follow_evals &
    FOLLOW_PID=$!
    trap '[ -n "${FOLLOW_PID:-}" ] && kill "$FOLLOW_PID" 2>/dev/null || true' EXIT
fi

for ((i = ${FIRST:-0}; i < NSEG; i++)); do
    SET="${SETS[$((i % NSEG))]}"
    N=$(ls "$ART/sets/$SET" | grep -c '\.sbb$')     # scenarios in the set (one group per building after merge)
    SEG="${NAME}_s$(printf '%02d' "$i")"
    leak_check
    INIT="--policy.zero_init_actions=0"
    [ -n "$PREV" ] && INIT="--base.load_model_path=/res/runs/${PREV#$CP/results/runs/}"
    WON=""
    for ((E = ENVS_HI; E >= ENVS_LO; E -= ENVS_STEP)); do
        AGENTS=$((E * N))
        if (( AGENTS % BATCH_DIV != 0 )); then
            echo "[$(date +%H:%M:%S)] segment $SEG: skip ENVS=$E ($AGENTS agents is not divisible by $BATCH_DIV)"
            continue
        fi
        echo "[$(date +%H:%M:%S)] segment $SEG: set $SET, $N scenarios, ENVS=$E, $AGENTS agents, sigma 0.37 constant"
        rm -rf "$CP/results/runs/$SEG/ckpt"
        set +e
        SB_GPU="$TRAIN_GPU" "$CP/tools/run_train.sh" "$SEG" --env.blob_dir="/art/sets/$SET" --env.groups="$N" \
            --vec.total_agents="$AGENTS" --train.total_timesteps="$STEPS" \
            --env.canonical=1 --env.residual=0 --env.rew_norm=1 --env.reward_scale=1 \
            --train.adv_norm=2 --train.shuffle=1 --train.rew_clamp=0 \
            --train.action_repeat="$HOLD" --base.checkpoint_optim=1 \
            --policy.logstd_anneal=0 --policy.init_logstd=-1 \
            --train.gamma="$GAMMA" --train.gae_lambda="$LAMBDA" \
            --train.learning_rate="$LR" --train.anneal_lr=0 \
            $INIT "$@" > "$CP/results/runs/$SEG.out" 2>&1
        rc=$?
        set -e
        LOG="$CP/results/runs/$SEG/train.log"
        if [ -f "$LOG" ] && grep -q 'scenario needs' "$LOG"; then
            echo "segment $SEG: a building exceeds the action or observation limit (see $LOG). Fewer environments will not load it."
            exit 1
        fi
        LAST=$(find "$CP/results/runs/$SEG/ckpt" -name '*.bin' ! -name '*.mb' -size +0c -printf '%f\n' 2>/dev/null | sort | tail -1)
        if [ -n "$LAST" ] && ! resource_fail "$LOG" && ! resource_fail "$CP/results/runs/$SEG.out"; then
            WON="$E"
            break
        fi
        if resource_fail "$LOG" || resource_fail "$CP/results/runs/$SEG.out"; then
            echo "[$(date +%H:%M:%S)] segment $SEG: ENVS=$E did not fit (rc=$rc), trying $((E - ENVS_STEP))"
            continue
        fi
        echo "segment $SEG failed at ENVS=$E for a reason a smaller batch will not fix (rc=$rc, see results/runs/$SEG/train.log)"
        exit 1
    done
    if [ -z "$WON" ]; then
        echo "segment $SEG: no environment count from $ENVS_HI down to $ENVS_LO fitted"
        exit 1
    fi
    PREV="$CP/results/runs/$SEG/ckpt/$LAST"
    echo "$i $SET $PREV $WON" >> "$CHAIN"
    echo "[$(date +%H:%M:%S)] segment $SEG done at ENVS=$WON -> $PREV"
    seg_report "$SEG"
done
touch "$CP/results/runs/$NAME.chain.done"
echo "final: $PREV"
if [ -n "$FOLLOW_PID" ]; then wait "$FOLLOW_PID" || true; fi
if [ "${EVAL_FOLLOW:-1}" = 1 ]; then
    echo "[$(date +%H:%M:%S)] chain summary:"
    for ((i = 0; i < NSEG; i++)); do
        report_score "$(printf 'eval/%s_s%02d_%s.json' "$NAME" "$i" "$CORE3")" "$BAR_CORE3" "s$i held-out"
        report_score "$(printf 'eval/%s_s%02d_insample_%s.json' "$NAME" "$i" "${SETS[$((i % NSEG))]}")" 0 "s$i in-sample"
    done
    report_score "eval/${NAME}_final_eval9.json" "$BAR_EVAL9" "eval9"
fi

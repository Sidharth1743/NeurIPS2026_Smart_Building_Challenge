#!/bin/bash
# Deterministic full-episode evaluation of a policy on a blob set, one env per scenario, each env's FIRST episode
# from a fresh start (= a fresh gateway episode; env.desync = 0), through the GPU queue.
#   tools/eval_policy.sh <set dir under artifacts/> <out.json under results/> [model.bin|-] [extra ./puffer args]
# model "-" = no checkpoint (with --policy.zero_init_actions=1 and the residual env: the base controller alone).
# SB_BIN selects the container binary (default puffer; puffer_f64 for a score while ./puffer is the fp32 trainer).
# SB_GPU set: the caller already holds that device (tools/with_gpu.sh). Skip the global queue so a score on
# the 3060 can run while training holds the 3090.
# Writes results/<out.json>: per scenario mean per-step reward + reward terms, and "score" = the leaderboard metric.
set -euo pipefail
. "$(dirname "$0")/common.sh"
SET="$1"; OUT="$2"; MODEL="${3:--}"; shift 3 || shift $#
PL="${PUFFERLIB:-${SB_PL_ROOT:-/data/puffer-work}/plverify}"
BIN="${SB_BIN:-puffer}"
N=$(ls "$ART/$SET"/*.sbb | wc -l)
M=""; [ "$MODEL" != "-" ] && M="$MODEL"
mkdir -p "$(dirname "$CP/results/$OUT")"
[ ! -e "$CP/results/$OUT" ] || { echo "Preserving existing output: $OUT" >&2; exit 1; }
if [ -n "${SB_EVAL_SPLITS:-}" ]; then
  read -r -a PARTS <<< "$SB_EVAL_SPLITS"
  RESULTS=(); index=0
  for part in "${PARTS[@]}"; do
    part_out="${OUT%.json}.part${index}.json"
    SB_EVAL_SPLITS= SB_CONTAINER_NAME="${SB_CONTAINER_NAME:+${SB_CONTAINER_NAME}-p${index}}" \
      "$0" "$part" "$part_out" "$MODEL" "$@"
    RESULTS+=("$CP/results/$part_out"); index=$((index+1))
  done
  python3 "$CP/tools/merge_eval_json.py" "$CP/results/$OUT" "${RESULTS[@]}"
  exit 0
fi
RAWLOG="$CP/results/$OUT.log"
[ ! -e "$RAWLOG" ] || { echo "Preserving existing log: $RAWLOG" >&2; exit 1; }
EVAL=(docker run --rm "${DEVS[@]}" -e SB_EVAL_OUT="/res/$OUT" \
  -v "$PL":/work:ro -v "$ART":/art:ro -v "$CP/results":/res -w /work "$IMAGE" \
  "./$BIN" eval)
if [ -n "${SB_CONTAINER_NAME:-}" ]; then EVAL=(docker run --rm --name "$SB_CONTAINER_NAME" "${EVAL[@]:3}"); fi
[ -z "$M" ] || EVAL+=("$M")
EVAL+=(--headless "--env.blob_dir=/art/$SET" "--env.groups=$N" "--base.eval_agents=$N" "--base.eval_episodes=$N"
  --env.desync=0 --base.deterministic=1 --base.cudagraphs=-1 "$@")
rc=0
if [ -n "${SB_GPU:-}" ] && [ "${SB_GPU}" != all ]; then
  "${EVAL[@]}" >"$RAWLOG" 2>&1 || rc=$?
else
  "$CP/../research/tools/gpu.sh" main "eval-$(basename "$OUT")" -- "${EVAL[@]}" >"$RAWLOG" 2>&1 || rc=$?
fi
if [ "$rc" -ne 0 ]; then tail -30 "$RAWLOG" >&2; exit "$rc"; fi
if [ ! -s "$CP/results/$OUT" ]; then echo "Missing score output; full log: $RAWLOG" >&2; tail -15 "$RAWLOG" >&2; exit 1; fi
python3 -c "import json;d=json.load(open('$CP/results/$OUT'));print('$OUT: complete',d['complete'],'scenarios',d['n_scenarios_done'],'score %.6f'%d['score'])"

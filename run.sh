#!/usr/bin/env bash
# The entry point. Six commands, plus reproduce and paper.
#
#   ./run.sh setup       container, Python 3.12 oracle env, PufferLib at the pinned commit + our patch
#   ./run.sh data        bake the 30 shipped scenarios, then the oracle traces and CPU references
#   ./run.sh verify      35 bit-exactness gates; ends with ALL CHECKS PASSED
#   ./run.sh train       PPO, 30 scenarios, 960 agents (480 if 960 does not fit the card)
#   ./run.sh eval        deterministic fp64 month for every scenario in a set
#   ./run.sh score       the 90-month panel for one checkpoint, with per-step recordings
#   ./run.sh reproduce   data <set> + verify + the 90-month panel of the shipped checkpoint
#   ./run.sh paper       rebuild both PDFs
#   ./run.sh doctor      print what is installed and what is missing
#
# Environment overrides, all optional:
#   SB_PL_ROOT   where the PufferLib checkouts live          (default ~/.cache/sb/pufferlib)
#   PUFFERLIB    one patched checkout                        (default $SB_PL_ROOT/plverify)
#   SB_GPU       which device to use, e.g. 0                 (default all)
#   SB_IMAGE     container image                             (default pufferai/puffertank:5.0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP="$HERE/complete_port"
SK="$HERE/starter-kit"
PL_ROOT="${SB_PL_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/sb/pufferlib}"
PL="${PUFFERLIB:-$PL_ROOT/plverify}"
IMAGE="${SB_IMAGE:-pufferai/puffertank:5.0}"
KIT_SHA="$(sed -n 's/^- commit: `\(.*\)`.*/\1/p' "$HERE/STARTER_KIT.md" 2>/dev/null | head -1)"
export SB_CP="$CP"
export SB_PL_ROOT="$PL_ROOT"

say() { printf '\n== %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
retry() {  # retry <attempts> <command...>
  local n="$1" i=1; shift
  while true; do
    "$@" && return 0
    [ "$i" -ge "$n" ] && return 1
    echo "  attempt $i failed, retrying in 5 s" >&2
    sleep 5; i=$((i + 1))
  done
}

find_run() {  # find_run <run-name or folder>
  local want="$1" hit
  hit="$(find "$HERE/experiments" -maxdepth 3 -type d -name "$want" 2>/dev/null | head -1)"
  [ -n "$hit" ] || hit="$(find "$HERE/experiments" -maxdepth 4 -type d -name "$want*" 2>/dev/null | head -1)"
  [ -n "$hit" ] || { echo "no experiment matches '$want'; see experiments/README.md" >&2; exit 1; }
  printf '%s\n' "$hit"
}

make_set() {  # make_set <set name> -> prints the directory of hard links
  local name="$1" list="$HERE/data/scenarios/sets/$1.txt" out="$CP/artifacts/sets/$1"
  [ -f "$list" ] || { echo "no scenario list $list" >&2; exit 1; }
  mkdir -p "$out"
  while read -r blob; do
    [ -n "$blob" ] || continue
    [ -f "$CP/artifacts/blobs/$blob" ] || { echo "missing blob $blob: run ./run.sh data" >&2; exit 1; }
    ln -f "$CP/artifacts/blobs/$blob" "$out/$blob" 2>/dev/null || cp "$CP/artifacts/blobs/$blob" "$out/$blob"
  done < "$list"
  printf '%s\n' "$out"
}

cmd="${1:-help}"; shift || true

case "$cmd" in
setup)
  need docker
  while [ $# -gt 0 ]; do
    case "$1" in
      --image) IMAGE="$2"; shift 2 ;;
      --pl-root) PL_ROOT="$2"; PL="${PUFFERLIB:-$PL_ROOT/plverify}"; export SB_PL_ROOT="$PL_ROOT"; shift 2 ;;
      *) echo "setup: unknown option $1 (supported: --image <image>, --pl-root <dir>)" >&2; exit 2 ;;
    esac
  done
  say "container"
  docker pull "$IMAGE"
  say "starter kit at $KIT_SHA"
  if [ ! -d "$SK/.git" ] && [ ! -f "$SK/README.md" ]; then
    retry 3 git -c http.version=HTTP/1.1 clone https://github.com/smart-buildings-2026/starter-kit.git "$SK"
  fi
  if [ -d "$SK/.git" ] && [ -n "$KIT_SHA" ]; then
    git -C "$SK" fetch --depth 1 origin "$KIT_SHA" 2>/dev/null || true
    git -C "$SK" checkout -q "$KIT_SHA" 2>/dev/null || echo "  note: could not check out $KIT_SHA, staying on the current commit"
  fi
  say "python 3.12 oracle environment (numba has no 3.13 wheels)"
  ( cd "$SK" && uv sync --extra dev --python 3.12 )
  say "pufferlib at the pinned base commit, under $PL_ROOT"
  mkdir -p "$PL_ROOT" 2>/dev/null || {
    echo "cannot create $PL_ROOT. Set SB_PL_ROOT to a directory you can write, for example" >&2
    echo "  export SB_PL_ROOT=\$HOME/pufferlib" >&2
    exit 1; }
  BASE="$(cat "$CP/patches/pufferlib_base_commit.txt")"
  if [ ! -d "$PL_ROOT/plclean/.git" ]; then
    # HTTP/1.1 and a shallow, blob-less clone: the full history is 100+ MB and the HTTP/2 stream
    # stalls behind some proxies.
    retry 3 git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none \
      https://github.com/PufferAI/PufferLib.git "$PL_ROOT/plclean" || {
        echo "could not clone PufferLib; set SB_PL_ROOT to a directory that already holds plclean" >&2
        exit 1; }
  fi
  retry 3 git -C "$PL_ROOT/plclean" -c http.version=HTTP/1.1 fetch -q --depth 1 origin "$BASE" 2>/dev/null || true
  if [ ! -d "$PL" ]; then
    git -C "$PL_ROOT/plclean" worktree add --detach "$PL" "$BASE"
  fi
  # Idempotent: apply the patch only if the checkout does not already carry it.
  if ( cd "$PL" && git apply --check "$CP/patches/pufferlib_core.diff" ) 2>/dev/null; then
    "$CP/tools/deploy_to_pufferlib.sh" "$PL" --apply-patch
  else
    echo "  checkout already carries the patch; redeploying the sources only"
    "$CP/tools/deploy_to_pufferlib.sh" "$PL"
  fi
  say "viewer environment (pygame, numpy, tensorboard: three gates and tools/watch.sh use it)"
  VPY="$CP/viewer/.venv/bin/python"
  if [ ! -x "$VPY" ]; then
    if command -v uv >/dev/null 2>&1; then
      ( cd "$CP/viewer" \
        && retry 3 uv venv --python 3.12 .venv \
        && retry 3 uv pip install --python .venv/bin/python pygame numpy tensorboard )
    else
      python3 -m venv "$CP/viewer/.venv" \
        && retry 3 "$CP/viewer/.venv/bin/pip" -q install pygame numpy tensorboard
    fi
  fi
  [ -x "$VPY" ] || echo "  note: the viewer environment could not be built; the viewer gates will fail"
  say "done; next: ./run.sh data"
  ;;
data)
  need docker
  say "30 shipped scenarios -> complete_port/artifacts/train (4.6 GB)"
  "$CP/tools/bake_training_set.sh"
  say "oracle traces and CPU references"
  "$CP/tools/make_artifacts.sh"
  if [ $# -ge 1 ]; then
    say "an evaluation set as well: $1"
    "$CP/tools/bake_eval_set.sh" "$HERE/data/scenarios/sets/$1.txt" "${2:-2}"
  fi
  say "done; next: ./run.sh verify"
  ;;
verify)
  need docker
  say "verify_all --trainer; this builds both variants and runs 35 gates"
  PUFFERLIB="$PL" "$CP/tools/verify_all.sh" --trainer "$@"
  ;;
train)
  need docker
  [ -d "$PL" ] || { echo "no checkout at $PL: run ./run.sh setup" >&2; exit 1; }
  # the beta head and the zone MLP live in the shared canonical policy build, not in the plain one
  for arg in "$@"; do
    case "$arg" in
      --policy.beta=1*|--policy.zone_mlp=1*) export SB_TRAIN_BIN="${SB_TRAIN_BIN:-puffer_shared}" ;;
    esac
  done
  say "PPO on the 30 shipped scenarios${SB_TRAIN_BIN:+ ($SB_TRAIN_BIN)}"
  PUFFERLIB="$PL" "$CP/tools/train.sh" "$@"
  ;;
eval)
  need docker
  [ -x "$PL/puffer_f64" ] || [ -x "$PL/puffer" ] || { echo "no trainer binary in $PL: run ./run.sh setup and ./run.sh verify first" >&2; exit 1; }
  SET="${1:-eval5_heldout18}"; OUTJSON="${2:-eval_$(date +%Y%m%d_%H%M%S).json}"
  CKPT="${3:--}"; RUNSRC="${4:-}"
  DIR="$(make_set "$SET")"
  # The binary builds the network from its flags, so a trained checkpoint needs the recipe it was
  # trained with: without them the loader rejects the weight-file size.
  FLAGS=()
  if [ -n "$RUNSRC" ]; then
    RUNSRC="$(find_run "$RUNSRC")"
    mapfile -t FLAGS < <(python3 "$HERE/complete_port/tools/decode_flags.py" "$RUNSRC")
  fi
  # the canonical shared policy (beta head, zone MLP) is a separate build; use it when present
  [ -x "$PL/puffer_shared" ] && export SB_BIN="${SB_BIN:-puffer_shared}"
  # the container mounts complete_port/results at /res and nothing else, so the weights are copied in
  MODEL="$CKPT"
  if [ "$CKPT" != "-" ]; then
    [ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
    mkdir -p "$CP/results"
    cp "$CKPT" "$CP/results/_eval_weights.bin"
    MODEL="/res/_eval_weights.bin"
  fi
  say "eval $SET ($(ls "$DIR"/*.sbb | wc -l) scenarios) -> results/$OUTJSON"
  PUFFERLIB="$PL" "$CP/tools/eval_policy.sh" "sets/$SET" "$OUTJSON" "$MODEL" "${FLAGS[@]}" "${@:5}"
  ;;
score)
  need docker
  RUN="${1:?usage: ./run.sh score <run> <step> [set]}"
  STEP="${2:?usage: ./run.sh score <run> <step> [set]}"
  SET="${3:-eval5_heldout18}"
  [ -x "$PL/puffer_shared" ] || { echo "no $PL/puffer_shared: run ./run.sh verify, which builds the shared-policy binary the recorder needs" >&2; exit 1; }
  DIRRUN="$(find_run "$RUN")"
  BIN="$DIRRUN/ckpt/$(printf '%016d' "$STEP").bin"
  [ -f "$BIN" ] || { echo "no checkpoint $BIN" >&2; exit 1; }
  make_set "$SET" >/dev/null
  BYTES=$(stat -c %s "$BIN")
  OUT="$CP/results/score_$(basename "$DIRRUN")_$STEP"
  say "score $DIRRUN @ $STEP on $SET -> $OUT"
  python3 "$CP/tools/score_eval90.py" \
    --run "$DIRRUN" --runtime "$PL" --set "artifacts/sets/$SET" \
    --out "$OUT" --gpu "${SB_GPU:-0}" \
    --ckpt-bytes "$BYTES" --checkpoints "$STEP=$BIN" "${@:4}"
  ;;
reproduce)
  SET="${1:-eval5_heldout18}"
  say "data, including the $SET set"
  "$0" data "$SET" "${2:-2}"
  say "verify, which also builds the shared-policy binary the scorer records with"
  "$0" verify
  say "the $SET panel of the shipped Final checkpoint; the paper reports 0.0165 at 90 months"
  "$0" score 19_final_r139_30_scenario_200m 199065600 "$SET"
  ;;
paper)
  need tectonic
  if [ $# -ge 1 ]; then
    say "compiling $*"
    ( cd "$HERE/paper" && for f in "$@"; do tectonic -X compile "$f"; done )
  else
    say "both PDFs"
    ( cd "$HERE/paper" && tectonic -X compile neurips_3_method.tex && tectonic -X compile neurips_3_method_appendix.tex )
  fi
  ;;
doctor)
  say "commands"
  for c in docker git uv tectonic python3 nvidia-smi; do
    printf '%-10s %s\n' "$c" "$(command -v "$c" || echo MISSING)"
  done
  say "container"; docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "$IMAGE present" || echo "$IMAGE not pulled"
  say "checkout"; [ -d "$PL" ] && echo "$PL" || echo "$PL missing"
  say "oracle env"; [ -x "$SK/.venv/bin/python" ] && echo "$SK/.venv present" || echo "starter-kit/.venv missing"
  say "artifacts"; ls "$CP/artifacts/train"/*.sbb 2>/dev/null | wc -l | xargs echo "train blobs:"
  say "gpu"; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  ;;
help|-h|--help|"")
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  ;;
*)
  echo "unknown command: $cmd" >&2
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 1
  ;;
esac

#!/bin/bash
# Run any tool pinned to one GPU, serialised with other jobs on the same device (P2: the 3060 is a
# second experiment slot, so the "one job at a time" rule becomes "one job per GPU").
#
#   tools/with_gpu.sh 1 -- tools/eval_policy.sh train results/eval.json
#   PUFFERLIB=/path/to/pufferlib SB_F32=1 tools/with_gpu.sh 0 -- tools/train.sh
#   tools/with_gpu.sh            # list devices, their memory and the locks held
#
# Every tool honours SB_GPU (common.sh: `--gpus device=N` instead of `--gpus all`); this wrapper sets
# it for the child and takes the per-device flock, so two jobs can never land on the same GPU by
# accident.  The lock is held for the whole command; a second job on the same GPU waits its turn (it
# does not fail).  Put `VAR=value` before this script, not after `--`: the lock is taken with
# `flock <file> <command>`, which execs the command directly (no shell to parse assignments).
set -e

if [ $# -eq 0 ]; then
  ( . "$(dirname "$0")/common.sh"
    echo "SB_GPU unset -> ${DEVS[*]}" ) 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  for f in /tmp/sb_gpu_*.lock; do
    [ -e "$f" ] || continue
    if flock -n "$f" true 2>/dev/null; then echo "$f: free"; else echo "$f: busy"; fi
  done
  exit 0
fi

G="$1"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "with_gpu.sh: nothing to run after '--'"; exit 2; }
case "$G" in
  all|"") LOCK=/tmp/sb_gpu_all.lock ;;
  *[!0-9]*) echo "with_gpu.sh: GPU must be an index or 'all', got '$G'"; exit 2 ;;
  *) LOCK="/tmp/sb_gpu_$G.lock" ;;
esac
export SB_GPU="$G"
. "$(dirname "$0")/common.sh"      # after SB_GPU is set: DEVS must match the lock we take
if ! flock -n "$LOCK" true 2>/dev/null; then
  echo "with_gpu.sh: GPU $G is busy, waiting for $LOCK ..." >&2
fi
exec flock "$LOCK" "$@"

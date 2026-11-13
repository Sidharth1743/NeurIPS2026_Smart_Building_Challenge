#!/bin/bash
# Start the approved 50M CleanRL-SAC run (or a named one) on GPU 0.
#   SAC/tools/run_50m.sh [run_name] [config]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
RUN="${1:-r01_cleanrl_sac_s75}"
CFG="${2:-$SAC/config/sac50m.json}"
mkdir -p "$SAC/results/$RUN"
[ -f "$SAC/results/$RUN/runner.pid" ] && kill -0 "$(cat "$SAC/results/$RUN/runner.pid")" 2>/dev/null && {
  echo "run $RUN already alive (pid $(cat "$SAC/results/$RUN/runner.pid"))"; exit 1; }
mkdir -p "$SAC/run/torchinductor" "$SAC/run/triton"
export TORCHINDUCTOR_CACHE_DIR="$SAC/run/torchinductor"
export TRITON_CACHE_DIR="$SAC/run/triton"
cd "$SAC"
nohup .venv/bin/python train/train_sac.py --run "$RUN" --out "results/$RUN" --gpu 0 --config "$CFG" \
  >> "results/$RUN/runner.log" 2>&1 &
echo $! > "results/$RUN/runner.pid"
echo "START $RUN pid $(cat "results/$RUN/runner.pid") log $SAC/results/$RUN/train.log"

#!/bin/bash
# Score the pre-registered saves of a run on the 10-month fp64 grid.
#   SAC/tools/score_all.sh r01_cleanrl_sac_s75 [gpu]
# Skips saves that are not there yet and any month already scored. Never
# overwrites an existing score file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
RUN="${1:?run name}"
GPU="${2:-0}"
R="$SAC/results/$RUN"
mkdir -p "$R/scored"
# 4*2,506,752 .. 20*2,506,752
for k in 4 8 12 16 20; do
  STEP=$((k * 2506752))
  P=$(printf "%s/ckpt/%012d.pt" "$R" "$STEP")
  O="$R/scored/$STEP.json"
  if [ ! -f "$P" ]; then echo "waiting: $P"; continue; fi
  if [ -f "$O" ]; then echo "have: $O"; continue; fi
  "$SAC/.venv/bin/python" "$HERE/score_months.py" --ckpt "$P" --out "$O" --gpu "$GPU" --tag "s$STEP"
done

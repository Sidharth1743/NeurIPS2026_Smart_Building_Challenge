#!/bin/bash
# One TensorBoard on :6007 that shows EVERY SAC run, automatically.
#
#   SAC/tools/tb_view.sh            # build/refresh the links and (re)start :6007 if needed
#   SAC/tools/tb_view.sh --watch    # the same, then refresh the links every 60 s
#
# Each run appears as results/tb_all/<run>, pointing at tb_fixed when the run has
# one (r01, written on the vector-step axis) and tb otherwise (r02+, already on
# the env-step axis). TensorBoard picks up a new subdirectory on its own.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
VIEW="$SAC/results/tb_all"
PORT="${TB_PORT:-6007}"
LINK() {
  mkdir -p "$VIEW"
  for d in "$SAC"/results/*/; do
    run="$(basename "$d")"
    [ -d "$d/tb" ] || continue
    src="$d/tb"
    [ -d "$d/tb_fixed" ] && src="$d/tb_fixed"
    ln -sfn "$src" "$VIEW/$run"
  done
}
LINK
if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  setsid nohup "$SAC/.venv/bin/tensorboard" --logdir "$VIEW" --port "$PORT" --bind_all \
    > "$SAC/results/tb_server.log" 2>&1 < /dev/null &
  echo "tensorboard started on :$PORT (log results/tb_server.log)"
else
  echo "tensorboard already listening on :$PORT"
fi
if [ "${1:-}" = "--watch" ]; then
  while :; do sleep 60; LINK; done
fi
echo "runs: $(ls "$VIEW" | tr '\n' ' ')"

#!/bin/bash
# TensorBoard for a trainer run:  tb.sh [logdir]   (default: $PUFFERLIB/logs/smart_buildings/tb)
# Open http://localhost:6006.  tools/tbread.py <logdir> prints the latest scalars without a browser.
. "$(dirname "$0")/common.sh"
LOGDIR="${1:-${PUFFERLIB:-${SB_PL_ROOT:-/data/puffer-work}/plverify}/logs/smart_buildings/tb}"
exec "$CP/viewer/.venv/bin/tensorboard" --logdir "$LOGDIR" --port "${TB_PORT:-6006}"

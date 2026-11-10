#!/bin/bash
# Watch the GPU env run: scripted policy in the container, viewer on the host.
#   watch.sh <scenario> [--policy replay|mid|random] [--steps N] [--fps F] [--envs N]
#                       [--every K] [--terminal]
# <scenario> is a name in artifacts/blobs (e.g. building_train_1__fairbanks_1).
#   replay = the oracle trace's recorded actions, looped (needs artifacts/traces/<scenario>.step)
#   mid    = middle of every continuous range, discrete at minimum   (no trace needed)
#   random = uniform random per env                                   (no trace needed)
# --fps paces the simulation in steps/s (default 20; 0 = as fast as the GPU goes), --every K
# writes a snapshot every Kth step, --envs sets the batch size (the viewer shows env 0).
# SB_CONT_BIG=1 selects the 320-action build (see README). To watch a trainer run or a
# checkpoint instead, see README "Viewer".
set -e
. "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { sed -n 2,13p "$0"; exit 2; }
NAME="${1%.sbb}"; NAME="$(basename "$NAME")"; shift
POLICY=replay; STEPS=1000000; FPS=20; ENVS=8; EVERY=1; TERM_FLAG=""
while [ $# -gt 0 ]; do case "$1" in
  --policy) POLICY="$2"; shift 2;; --steps) STEPS="$2"; shift 2;; --fps) FPS="$2"; shift 2;;
  --envs) ENVS="$2"; shift 2;; --every) EVERY="$2"; shift 2;; --terminal) TERM_FLAG="--terminal"; shift;;
  *) echo "unknown option $1"; exit 2;; esac; done
BLOB="$ART/blobs/$NAME.sbb"; [ -f "$BLOB" ] || { echo "no blob $BLOB (tools/new_scenarios.sh / make_artifacts.sh)"; exit 1; }
BIN=gate; [ -n "${SB_CONT_BIG:-}" ] && BIN=gate_big
[ -x "$ART/bin/$BIN" ] || "$CP/tools/build.sh"
TRACE="/cp/artifacts/traces/$NAME.step"
if [ ! -f "$ART/traces/$NAME.step" ]; then
  [ "$POLICY" = replay ] && { echo "policy replay needs $ART/traces/$NAME.step; use --policy mid|random"; exit 1; }
  TRACE=-
fi
SNAP="$CP/results/live.sbs"; mkdir -p "$CP/results"; rm -f "$SNAP" "$SNAP.tmp"
CN="sb_play_$$"
trap 'docker kill "$CN" >/dev/null 2>&1 || true' EXIT INT TERM
docker run --rm "${DEVS[@]}" --name "$CN" -v "$CP":/cp -w /cp -e SB_SNAPSHOT=/cp/results/live.sbs -e SB_SNAP_EVERY="$EVERY" \
  "$IMAGE" bash -lc "/cp/artifacts/bin/$BIN /cp/artifacts/blobs/$NAME.sbb $TRACE $ENVS play $POLICY $STEPS $FPS" >"$CP/results/play.log" 2>&1 &
VENV="$CP/viewer/.venv/bin/python"
[ -x "$VENV" ] || { echo "viewer env missing: uv venv --python 3.12 viewer/.venv && uv pip install --python viewer/.venv/bin/python pygame numpy tensorboard"; exit 1; }
"$VENV" "$CP/viewer/sb_viewer.py" "$SNAP" $TERM_FLAG --wait 120

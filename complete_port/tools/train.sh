#!/bin/bash
# Train PPO on the GPU env (TensorBoard logging on).
#   PUFFERLIB=/path/to/patched/checkout tools/train.sh [<scenario-name>] [extra ./puffer train args]
# Default = multi-building: every .sbb in artifacts/train (the 30 shipped scenarios, bake them with
# tools/bake_training_set.sh), 32 envs each; the weathers of a building share a group (6 groups x 160 envs). With a scenario name
# (a blob in artifacts/blobs) it trains on that single scenario with 256 envs.
# The checkout comes from tools/deploy_to_pufferlib.sh <checkout> --apply-patch; ./puffer is built
# here if missing. Output also goes to results/train_<time>.log.
# Metrics (tools/tb.sh): env/* per-episode means, live/* and scenario/<name>/* step-level means every
# epoch, loss/*, perf/*, util/*.  SB_SNAPSHOT=<file> additionally writes a viewer snapshot.
set -e
. "$(dirname "$0")/common.sh"
PL="${PUFFERLIB:?set PUFFERLIB to a patched PufferLib checkout}"; PL="$(cd "$PL" && pwd)"
if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
  NAME="$(basename "${1%.sbb}")"; shift
  [ -f "$ART/blobs/$NAME.sbb" ] || { echo "no blob $ART/blobs/$NAME.sbb"; exit 1; }
  SET="--env.blob=/art/blobs/$NAME.sbb --vec.total_agents=256"
else
  [ "$(ls "$ART/train"/*.sbb 2>/dev/null | wc -l)" -ge 1 ] || { echo "no blobs in $ART/train: run tools/bake_training_set.sh"; exit 1; }
  SET="--env.blob_dir=/art/train"
fi
SNAPENV=(); [ -n "${SB_SNAPSHOT:-}" ] && SNAPENV=(-e SB_SNAPSHOT="$SB_SNAPSHOT" -e SB_SNAP_EVERY="${SB_SNAP_EVERY:-1}")
# SB_F32=1: training-only fp32 solve (the build.sh case).  The variant is read from the *binary*
# (`lu_solve_t...P f/P d` in the mangled solve names), not from a marker: anything that rebuilds
# ./puffer (verify_all, a manual ./build.sh) invalidates a marker, and the first A/B of this flag
# silently compared two fp64 binaries for exactly that reason.
VARIANT=f64; [ -n "${SB_F32:-}" ] && { SNAPENV+=(-e SB_F32=1); VARIANT=f32; }
mkdir -p "$CP/results"; LOG="$CP/results/train_$(date +%Y%m%d_%H%M%S).log"
echo "TensorBoard:  tools/tb.sh $PL/logs/smart_buildings/tb      log: $LOG"
docker run --rm "${DEVS[@]}" "${SNAPENV[@]}" -v "$PL":/work -v "$ART":/art -v "$CP/results":/res -w /work "$IMAGE" bash -lc \
  "have=none; [ -x ./puffer ] && { have=f64; grep -aq 'lu_solve_t.*EEv5LuDevPf' ./puffer && have=f32; }; \
   [ \"\$have\" = $VARIANT ] || ./build.sh smart_buildings; \
   grep -aq 'lu_solve_t.*EEv5LuDevPf' ./puffer && echo 'variant: fp32 solve' || echo 'variant: fp64 solve'; \
   ./puffer train $SET $*" 2>&1 | tr '\r' '\n' | tee "$LOG"

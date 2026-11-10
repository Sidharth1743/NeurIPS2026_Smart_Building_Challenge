#!/bin/bash
# Package a trained checkpoint as a submission folder (policy code + vendored base + weights + config).
#   tools/export_submission.sh <checkpoint.bin> <out dir> [res_scale] [res_dscale] [hidden] [layers]
# Comfort guard: GUARD_D0 / GUARD_D1 / GUARD_HOT in the environment (the env.guard_* the checkpoint was trained with;
# default off).
set -e
CK="$1"; OUT="$2"; RS="${3:-0.5}"; RD="${4:-1.5}"; H="${5:-256}"; L="${6:-2}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT/model"
cp "$CP/submission/residual_policy.py" "$CP/submission/pidw_base.py" "$CP/submission/LICENSE.starter-kit" "$OUT/"
cp "$CK" "$OUT/model/weights.bin"
NF=$(( $(stat -c %s "$CK") / 4 ))
OD=$(( (NF - H * 1024 - H - L * 3 * H * H) / (H + 1) ))
cat > "$OUT/model/config.json" <<J
{"weights": "weights.bin", "hidden": $H, "layers": $L, "obs_size": 1024, "out_dim": $OD,
 "residual": true, "res_scale": $RS, "res_dscale": $RD,
 "guard_d0": ${GUARD_D0:-1.0}, "guard_d1": ${GUARD_D1:-0.0}, "guard_hot": ${GUARD_HOT:-0}, "source_checkpoint": "$(realpath "$CK")"}
J
echo "exported $CK -> $OUT (floats $NF, out_dim $OD)"

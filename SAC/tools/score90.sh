#!/bin/bash
# Score checkpoints on the 90-scenario held-out set (5 buildings x 18 weathers),
# one building per process run, exactly like results/study_r139_eval90.
#   SAC/tools/score90.sh <run> <step> [<step> ...]      (GPU 1 by default)
# Writes results/<run>/scored90/<step>.json  (per building)  and  <step>.mean.json
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$SAC/.." && pwd)"
RUN="${1:?run name}"; shift
GPU="${SB_GPU:-1}"
SRC="$ROOT/complete_port/artifacts/sets/eval5_heldout18"
BASE="$SAC/artifacts/eval90"
mkdir -p "$BASE"
for b in 1 2 3 6 8; do
  d="$BASE/building_train_$b"; mkdir -p "$d"
  for f in "$SRC"/building_train_${b}__*.sbb; do
    ln -sfn "../../../../complete_port/artifacts/blobs/$(basename "$f")" "$d/$(basename "$f")"
  done
done
OUT="$SAC/results/$RUN/scored90"; mkdir -p "$OUT"
for step in "$@"; do
  CK="$SAC/results/$RUN/ckpt/$(printf '%012d' "$step").pt"
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  parts=()
  for b in 1 2 3 6 8; do
    O="$OUT/${step}_b${b}.json"
    [ -f "$O" ] && { echo "have $O"; parts+=("$O"); continue; }
    echo "== $RUN @$step building_train_$b"
    REC=()
    [ -n "${SB_RECORD_DIR:-}" ] && REC=(--record "$SAC/results/$RUN/record90/$step/building_train_$b")
    "$SAC/.venv/bin/python" "$HERE/score_set.py" --ckpt "$CK" --set "$BASE/building_train_$b" \
      --out "$O" --gpu "$GPU" --tag "s${step}b${b}" "${REC[@]}"
    parts+=("$O")
  done
  "$SAC/.venv/bin/python" - "$step" "$OUT" "${parts[@]}" <<'PY'
import json, sys
step, out, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
sc = {}
for p in parts:
    d = json.load(open(p))
    sc.update(d["scenarios"])
n = len(sc)
mean = sum(abs(v["mean_reward"]) for v in sc.values()) / n
res = {"step": int(step), "score": mean, "n_scenarios": n,
       "scenarios": {k: abs(v["mean_reward"]) for k, v in sc.items()},
       "terms": {t: sum(v[t] for v in sc.values()) / n
                 for t in ("productivity", "energy_cost", "carbon")},
       "per_building": {p.split("_b")[-1].split(".")[0]: json.load(open(p))["score"] for p in parts}}
json.dump(res, open(f"{out}/{step}.mean.json", "w"), indent=2)
print(f"### {step}  90-set score {mean:.6f}  over {n} scenarios")
PY
done

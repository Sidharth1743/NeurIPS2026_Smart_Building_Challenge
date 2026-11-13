#!/bin/bash
# Score checkpoints on the 54-scenario held-out set (train_4/5/7 x 18 weathers),
# one building per process run, the same protocol as tools/score90.sh.
#   SB_GPU=0 SAC/tools/score54.sh <run> <step> [<step> ...]
# Writes results/<run>/scored54/<step>.json (per building) and <step>.mean.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$SAC/.." && pwd)"
RUN="${1:?run name}"; shift
GPU="${SB_GPU:-0}"
SRC="$ROOT/complete_port/artifacts/sets/eval3_heldout18"
BASE="$SAC/artifacts/eval54"
BUILDINGS=(4 5 7)
mkdir -p "$BASE"
for b in "${BUILDINGS[@]}"; do
  d="$BASE/building_train_$b"; mkdir -p "$d"
  for f in "$SRC"/building_train_${b}__*.sbb; do
    ln -sfn "../../../../complete_port/artifacts/blobs/$(basename "$f")" "$d/$(basename "$f")"
  done
done
OUT="$SAC/results/$RUN/scored54"; mkdir -p "$OUT"
for step in "$@"; do
  CK="$SAC/results/$RUN/ckpt/$(printf '%012d' "$step").pt"
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  parts=()
  for b in "${BUILDINGS[@]}"; do
    O="$OUT/${step}_b${b}.json"
    [ -f "$O" ] && { echo "have $O"; parts+=("$O"); continue; }
    echo "== $RUN @$step building_train_$b"
    REC=()
    [ -n "${SB_RECORD_DIR:-}" ] && REC=(--record "$SAC/results/$RUN/record54/$step/building_train_$b")
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
print(f"### {step}  54-set score {mean:.6f}  over {n} scenarios")
PY
done

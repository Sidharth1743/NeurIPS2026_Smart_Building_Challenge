#!/bin/bash
# Score checkpoints on the 90- or 54-scenario held-out set, with per-building jobs run
# CONCURRENTLY (SB_JOBS, default 3). Same per-building protocol and the same mean JSON as
# tools/score90.sh / tools/score54.sh; only the scheduling changes.
#
#   SB_GPU=0 SB_JOBS=3 SB_RECORD_DIR=1 tools/score_par.sh r14_structured_200m_s75 90 49972800 119934720
#   SB_GPU=1 SB_JOBS=2 SB_RECORD_DIR=1 tools/score_par.sh r14_structured_200m_s75 54 159912960
#
# Jobs already finished are skipped, so a chain can be resumed by re-running the same command.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAC="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$SAC/.." && pwd)"
RUN="${1:?run name}"; KIND="${2:?90 or 54}"; shift 2
GPU="${SB_GPU:-0}"
JOBS="${SB_JOBS:-3}"
case "$KIND" in
  90) SRC="$ROOT/complete_port/artifacts/sets/eval5_heldout18"; BASE="$SAC/artifacts/eval90"; OUT="$SAC/results/$RUN/scored90"; REC_ROOT="$SAC/results/$RUN/record90"; BUILDINGS=(1 2 3 6 8) ;;
  54) SRC="$ROOT/complete_port/artifacts/sets/eval3_heldout18"; BASE="$SAC/artifacts/eval54"; OUT="$SAC/results/$RUN/scored54"; REC_ROOT="$SAC/results/$RUN/record54"; BUILDINGS=(4 5 7) ;;
  *) echo "set must be 90 or 54"; exit 2 ;;
esac
mkdir -p "$OUT"
for b in "${BUILDINGS[@]}"; do
  d="$BASE/building_train_$b"; mkdir -p "$d"
  for f in "$SRC"/building_train_${b}__*.sbb; do
    ln -sfn "../../../../complete_port/artifacts/blobs/$(basename "$f")" "$d/$(basename "$f")"
  done
done

running=0
for step in "$@"; do
  CK="$SAC/results/$RUN/ckpt/$(printf '%012d' "$step").pt"
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  for b in "${BUILDINGS[@]}"; do
    O="$OUT/${step}_b${b}.json"
    [ -f "$O" ] && { echo "have $O"; continue; }
    REC=()
    [ -n "${SB_RECORD_DIR:-}" ] && REC=(--record "$REC_ROOT/$step/building_train_$b")
    echo "== $RUN @$step building_train_$b (gpu $GPU)"
    "$SAC/.venv/bin/python" "$HERE/score_set.py" --ckpt "$CK" --set "$BASE/building_train_$b" \
      --out "$O" --gpu "$GPU" --tag "s${step}b${b}" "${REC[@]}" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n || true; running=$((running - 1)); fi
  done
done
wait || true

for step in "$@"; do
  parts=()
  for b in "${BUILDINGS[@]}"; do
    O="$OUT/${step}_b${b}.json"; [ -f "$O" ] && parts+=("$O")
  done
  [ "${#parts[@]}" -eq "${#BUILDINGS[@]}" ] || { echo "incomplete $step: ${#parts[@]}/${#BUILDINGS[@]}"; continue; }
  "$SAC/.venv/bin/python" - "$step" "$OUT" "${parts[@]}" <<'PY'
import json, sys
step, out, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
sc = {}
for p in parts:
    sc.update(json.load(open(p))["scenarios"])
n = len(sc)
mean = sum(abs(v["mean_reward"]) for v in sc.values()) / n
res = {"step": int(step), "score": mean, "n_scenarios": n,
       "scenarios": {k: abs(v["mean_reward"]) for k, v in sc.items()},
       "terms": {t: sum(v[t] for v in sc.values()) / n
                 for t in ("productivity", "energy_cost", "carbon")},
       "per_building": {p.split("_b")[-1].split(".")[0]: json.load(open(p))["score"] for p in parts}}
json.dump(res, open(f"{out}/{step}.mean.json", "w"), indent=2)
print(f"### {step}  {n}-set score {mean:.6f}  over {n} scenarios")
PY
done

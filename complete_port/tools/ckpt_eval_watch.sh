#!/bin/bash
# Watch a chain run's checkpoints and evaluate each landed checkpoint (every EVAL_EVERY-th bin,
# lexicographic order) on a fixed eval set, pinned to GPU 1 (the RTX 3060), fenced per-GPU by
# tools/with_gpu.sh.  Writes, under results/runs/<name>/:
#   eval/<name>_<step>.json   full per-scenario eval result (deterministic, fresh first episodes)
#   eval_ladder.csv           step, score, mean_reward, prod, energy, carbon
#   eval_tb/events.out.tfevents.watch.*   TensorBoard scalars (eval/score, ...) indexed by train step
# Exits when the chain's <name>.chain.done marker exists AND no unevaluated checkpoint remains.
#   SB_GPU=1 tools/ckpt_eval_watch.sh <run_name> [eval_set=sets/dev12] [eval_every=4]
set -u
. "$(dirname "$0")/common.sh"
NAME="${1:?run name}"; SET="${2:-sets/dev12}"; EVERY="${3:-4}"
R="$CP/results/runs/$NAME"
EVDIR="$R/eval"; DONE="$R/eval_done"; TB="$R/eval_tb"; CSV="$R/eval_ladder.csv"
mkdir -p "$EVDIR" "$DONE" "$TB"
[ -f "$CSV" ] || echo "step,score,mean_reward,prod,energy,carbon" > "$CSV"
PVENVS="$CP/viewer/.venv/bin/python"
TBEV="$TB/events.out.tfevents.watch.$$"
# sampled index: every (EVERY+1)-th checkpoint in the stable lexicographic order
SAMPLE_EVERY=$((EVERY + 1))

write_tb() {  # write_tb <step> <json>
  "$PVENVS" - "$1" "$2" "$TBEV" <<'PYEOF'
import json, sys
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
step, path, tbfile = int(sys.argv[1]), sys.argv[2], sys.argv[3]
d = json.load(open(path))
sc = d.get("scenarios", {})
def mean(k):
    xs = [v[k] for v in sc.values() if k in v]
    return sum(xs) / len(xs) if xs else 0.0
vals = {
    "eval/score": d.get("score", 0.0),
    "eval/mean_reward": mean("mean_reward"),
    "eval/productivity": mean("productivity"),
    "eval/energy_cost": mean("energy_cost"),
    "eval/carbon": mean("carbon"),
}
def scalar(tag, v):
    s = Summary(value=[Summary.Value(tag=tag, simple_value=v)])
    return Event(step=step, summary=s).SerializeToString()
with open(tbfile, "ab") as f:
    for t, v in vals.items():
        f.write(scalar(t, v))
PYEOF
}

append_csv() {  # append_csv <step> <json>
  "$PVENVS" - "$1" "$2" "$CSV" <<'PYEOF'
import json, sys, csv
path, step, csvp = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
sc = d.get("scenarios", {})
def mean(k):
    xs = [v[k] for v in sc.values() if k in v]
    return sum(xs) / len(xs) if xs else 0.0
with open(csvp, "a", newline="") as f:
    csv.writer(f).writerow([str(step), f"{d.get('score', 0):.6f}", f"{mean('mean_reward'):.6f}",
                            f"{mean('productivity'):.6f}", f"{mean('energy_cost'):.6f}",
                            f"{mean('carbon'):.6f}"])
PYEOF
}

while true; do
  mapfile -t BINS < <(ls "$R/ckpt" 2>/dev/null | grep '\.bin$' | sort)
  WORKED=0
  idx=0
  for f in "${BINS[@]}"; do
    idx=$((idx + 1))
    [ -f "$DONE/$f" ] && continue
    if [ $((idx % SAMPLE_EVERY)) -ne 0 ]; then
      touch "$DONE/$f.skip"      # not sampled: mark seen so the loop is never busy
      continue
    fi
    [ -f "$DONE/$f" ] && continue
    STEP="${f%.bin}"
    J="$EVDIR/${NAME}_${STEP}.json"
    echo "[$(date +%H:%M:%S)] eval $f (sampled idx $idx) on $SET"
    if [ ! -f "$J" ]; then
      SB_GPU=1 "$CP/tools/with_gpu.sh" 1 -- "$CP/tools/eval_policy.sh" "$SET" \
          "eval/ladder_${NAME}_${STEP}.json" "/res/runs/$NAME/ckpt/$f" >> "$R/eval_watch.log" 2>&1 \
        || { echo "eval failed for $f (see $R/eval_watch.log)"; }
      mv "$CP/results/eval/ladder_${NAME}_${STEP}.json" "$J" 2>/dev/null || true
    fi
    touch "$DONE/$f"             # sampled (evaluated or failed-but-attempted): done
    if [ -f "$J" ]; then
      append_csv "$((10#$STEP))" "$J"
      write_tb "$((10#$STEP))" "$J"
      WORKED=1
    fi
  done
  ALL_DONE=1
  for f in "${BINS[@]}"; do [ -f "$DONE/$f" ] || { ALL_DONE=0; break; }; done
  if [ "$ALL_DONE" = 1 ] && [ -f "$CP/results/runs/$NAME.chain.done" ]; then
    echo "[$(date +%H:%M:%S)] chain complete and all sampled checkpoints evaluated; watcher exits"
    exit 0
  fi
  [ "$WORKED" = 0 ] && sleep 90
done

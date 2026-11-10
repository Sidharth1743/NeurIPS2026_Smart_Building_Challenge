#!/bin/bash
# Time the GPU step (the gate's `phase ms/step`, the same number docs/HISTORY.md reports) for scenario
# blobs at several batch sizes, inside the CUDA container.  One run per (blob, nenv); the oracle trace
# in artifacts/traces supplies the actions, and its length sets the step count.
#
#   tools/bench.sh                                  # the six shipped shapes + one blob of each big set
#   tools/bench.sh --blobs "a.sbb b.sbb" --nenv "32 160" [--graph] [--bin gate_f32]
#   tools/bench.sh --nsys building_train_2__atlanta_3.sbb 160
#
# --graph adds SB_GRAPH=1 (the trainer's CUDA-graph path).  --bin picks the gate binary
# (default: gate, or gate_big when the blob's action block is wider than 256).  Result:
# results/perf/bench_<date>.tsv (+.md).
set -u
. "$(dirname "$0")/common.sh"
OUTDIR="$CP/results/perf"; mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_BLOBS="artifacts/train/building_train_1__fairbanks_1.sbb
artifacts/train/building_train_2__atlanta_3.sbb
artifacts/train/building_train_4__atlanta_5.sbb
artifacts/train/building_train_5__atlanta_6.sbb
artifacts/train/building_train_7__atlanta_8.sbb
artifacts/train/building_train_8__atlanta_1.sbb
artifacts/sets/diverseB35/building_30008__atlanta_7.sbb
artifacts/sets/diverseB35/building_30009__atlanta_7.sbb
artifacts/sets/diverseC35/building_30004__atlanta_7.sbb
artifacts/sets/diverseC35/building_30025__atlanta_7.sbb"
BLOBS="$DEFAULT_BLOBS"; NENVS="32 64 160 256"; GRAPH=0; NSYS=""; BIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --blobs) BLOBS="$2"; shift 2;;
    --nenv) NENVS="$2"; shift 2;;
    --graph) GRAPH=1; shift;;
    --nsys) NSYS="$2"; shift;;
    --bin) BIN="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
# the gate to use for one blob: --bin overrides; otherwise gate_big for wide action blocks
gate_for() { if [ -n "$BIN" ]; then echo "artifacts/bin/$BIN"; elif [ "$1" -gt 256 ]; then echo artifacts/bin/gate_big; else echo artifacts/bin/gate; fi; }
DARGS=(); [ "$GRAPH" = 1 ] && DARGS=(-e SB_GRAPH=1)
TSV="$OUTDIR/bench_$STAMP.tsv"
printf "blob\tnenv\tms_per_step\tus_per_env_step\tsteps\tgraph\tgate\n" > "$TSV"
run_one() {  # $1 = blob rel to CP, $2 = nenv
  local B="$1" N="$2" NAME TR T GATE NC STEPS LOG RAW
  NAME="$(basename "${B%.sbb}")"; TR="artifacts/traces/$NAME.step"
  [ -f "$CP/$B" ] || { echo "SKIP $NAME: no blob"; return 1; }
  [ -f "$CP/$TR" ] || { echo "SKIP $NAME: no trace $TR"; return 1; }
  NC="$(python3 -c "import struct;print(struct.unpack('<7i',open('$CP/$TR','rb').read(28))[2])")"
  STEPS="$(python3 -c "import struct;print(struct.unpack('<7i',open('$CP/$TR','rb').read(28))[4])")"
  GATE="$(gate_for "$NC")"
  LOG="$OUTDIR/.bench_${NAME}_$N.log"; RAW="$LOG"
  docker run --rm "${DEVS[@]}" "${DARGS[@]}" -v "$CP":/cp -w /cp "$IMAGE" bash -lc \
    "$GATE $B $TR $N" > "$RAW" 2>&1
  local MS
  MS="$(grep -o 'phase ms/step: [0-9.]*' "$RAW" | tail -1 | grep -o '[0-9.]*$')"
  if [ -z "$MS" ]; then echo "FAIL $NAME nenv=$N:"; tail -3 "$RAW"; return 1; fi
  python3 - "$TSV" "$NAME" "$N" "$MS" "$STEPS" "$GRAPH" "${GATE##*/}" <<'PY'
import sys
tsv, blob, nenv, ms, steps, graph, gate = sys.argv[1:8]
ms = float(ms); n = int(nenv)
with open(tsv, "a") as f:
    f.write(f"{blob}\t{n}\t{ms:.2f}\t{ms*1000/n:.1f}\t{steps}\t{graph}\t{gate}\n")
PY
  printf '  %-40s nenv=%-4s %7.2f ms  %7.1f us/env-step\n' "$NAME" "$N" "$MS" \
    "$(python3 -c "print(float('$MS')*1000/$N)")"
  rm -f "$RAW"
}
for B in $BLOBS; do for N in $NENVS; do run_one "$B" "$N"; done; done
python3 - "$TSV" > "${TSV%.tsv}.md" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
print("# GPU step timing (`gate` phase ms/step)\n")
print("| blob | envs | ms/step | us/env-step | env-steps/s | steps | graph | gate |")
print("|---|--:|--:|--:|--:|--:|--:|---|")
for r in rows:
    us = float(r["us_per_env_step"])
    print(f"| {r['blob']} | {r['nenv']} | {float(r['ms_per_step']):.2f} | {us:.1f} | {1e6/us:,.0f} |"
          f" {r['steps']} | {r['graph']} | {r['gate']} |")
PY
echo; echo "TSV: $TSV"; echo "MD:  ${TSV%.tsv}.md"
if [ -n "$NSYS" ]; then
  set -- $NSYS; NB="$1"; NN="$2"
  NAME="$(basename "${NB%.sbb}")"; TR="artifacts/traces/$NAME.step"
  NC="$(python3 -c "import struct;print(struct.unpack('<7i',open('$CP/$TR','rb').read(28))[2])")"
  GATE="$(gate_for "$NC")"
  echo "--- nsys profile: $NAME nenv=$NN ($GATE)"
  docker run --rm "${DEVS[@]}" --privileged -v "$CP":/cp -w /cp "$IMAGE" bash -lc \
    "nsys profile -o /cp/results/perf/nsys_${NAME}_${NN} --force-overwrite true $GATE $NB $TR $NN >/dev/null 2>&1; \
     nsys stats --report cuda_gpu_kern_sum --format table /cp/results/perf/nsys_${NAME}_${NN}.nsys-rep 2>/dev/null | tail -45"
fi

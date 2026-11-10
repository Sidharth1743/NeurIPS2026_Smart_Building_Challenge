#!/bin/bash
# One named training run through the GPU queue: clean TensorBoard dir, ./puffer train with the given args, then the
# run's TensorBoard logs (cleared inside the queued job, so queued runs never touch a running one's), newest checkpoints and the log are archived under results/runs/<name>/.
#   tools/run_train.sh <name> [./puffer train args...]
set -e
. "$(dirname "$0")/common.sh"
NAME="$1"; shift
PL="${PUFFERLIB:-${SB_PL_ROOT:-/data/puffer-work}/plverify}"
R="$CP/results/runs/$NAME"; mkdir -p "$R"
echo "$*" > "$R/args.txt"
touch "$R/.start"
# SB_F32=1: training-only fp32 solve. Same check as tools/train.sh: the variant is read from the
# built binary, and a mismatch rebuilds, so a previous exact-build ./puffer is not reused by accident.
VARIANT=f64; F32ENV=(); [ -n "${SB_F32:-}" ] && { F32ENV=(-e SB_F32=1); VARIANT=f32; }
"$CP/../research/tools/gpu.sh" main "train-$NAME" -- docker run --rm "${DEVS[@]}" "${F32ENV[@]}" -e SB_SNAPSHOT=/res/live.sbs -e SB_SNAP_EVERY=1 \
  -v "$PL":/work -v "$ART":/art -v "$CP/results":/res -w /work "$IMAGE" bash -lc \
  "have=none; [ -x ./puffer ] && { have=f64; grep -aq 'lu_solve_t.*EEv5LuDevPf' ./puffer && have=f32; }; \
   [ \"\$have\" = $VARIANT ] || ./build.sh smart_buildings; \
   grep -aq 'lu_solve_t.*EEv5LuDevPf' ./puffer && echo 'variant: fp32 solve' || echo 'variant: fp64 solve'; \
   rm -rf logs/smart_buildings/tb; mkdir -p logs/smart_buildings/tb /res/runs/$NAME/tb /res/runs/$NAME/ckpt; \
   chmod 777 logs/smart_buildings/tb /res/runs/$NAME/tb /res/runs/$NAME/ckpt; \
   touch /res/runs/$NAME/.start; \
   ( while true; do sleep 20; \
       cp -a logs/smart_buildings/tb/. /res/runs/$NAME/tb/ 2>/dev/null || true; \
       find checkpoints/smart_buildings \( -name '*.bin' -o -name '*.bin.mb' \) -newer /res/runs/$NAME/.start -size +0c -print0 2>/dev/null \
         | while IFS= read -r -d '' f; do \
             dest=/res/runs/$NAME/ckpt/\$(basename \"\$f\"); \
             s1=\$(stat -c%s \"\$f\" 2>/dev/null || echo 0); sleep 2; \
             s2=\$(stat -c%s \"\$f\" 2>/dev/null || echo 0); \
             [ \"\$s1\" = \"\$s2\" ] && [ \"\$s1\" -gt 0 ] || continue; \
             [ -f \"\$dest\" ] && [ \"\$(stat -c%s \"\$dest\")\" = \"\$s2\" ] && continue; \
             cp -f \"\$f\" \"\$dest\"; \
           done; \
       chmod -R a+rX /res/runs/$NAME/tb /res/runs/$NAME/ckpt 2>/dev/null || true; \
     done ) >/tmp/sb-train-copy.log 2>&1 & TBCP=\$!; \
   stdbuf -oL -eL ./puffer train $* 2>&1 | tr '\r' '\n' > /res/runs/$NAME/train.log; \
   rc=\${PIPESTATUS[0]}; \
   kill \$TBCP 2>/dev/null || true; wait \$TBCP 2>/dev/null || true; \
   cp -a logs/smart_buildings/tb/. /res/runs/$NAME/tb/; \
   find checkpoints/smart_buildings \( -name '*.bin' -o -name '*.bin.mb' \) -newer /res/runs/$NAME/.start -size +0c -exec cp -f {} /res/runs/$NAME/ckpt/ \;; \
   chmod -R a+rwX /res/runs/$NAME; exit \$rc"
ls "$R/ckpt" | tail -3

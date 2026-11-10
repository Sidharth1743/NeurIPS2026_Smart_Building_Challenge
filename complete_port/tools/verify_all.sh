#!/bin/bash
# End-to-end verification of the GPU port from complete_port alone (+ the unmodified
# starter-kit as oracle).  Writes results/verify_<timestamp>.log and exits non-zero on
# any failure.
#   verify_all.sh [--trainer]   --trainer also rebuilds ./puffer in a fresh PufferLib
#                               worktree (base commit + patch + deploy) and smoke-trains.
set -u
. "$(dirname "$0")/common.sh"
LOG="$CP/results/verify_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$CP/results"
RESUME_LOG="${SB_VERIFY_RESUME:-}"
FAIL=0
say() { echo "$@" | tee -a "$LOG"; }
chk() { # chk <description> <command...>
  d="$1"; shift
  if [ -n "$RESUME_LOG" ] && grep -Fxq "PASS  $d" "$RESUME_LOG"; then return 0; fi
  out=$("$@" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then say "PASS  $d"; else say "FAIL  $d"; echo "$out" | tail -5 | tee -a "$LOG"; FAIL=1; fi
}
if [ -n "$RESUME_LOG" ]; then
  # Resume only after all trainer checks and the weather-merge checks completed.
  # Earlier checkpoints and failures cannot be silently accepted.
  [ "${1:-}" = "--trainer" ] && [ -f "$RESUME_LOG" ] &&
  grep -Fxq 'PASS  trainer: team valve credit, reset, diagnostics and separate Adam state through CUDA graph' "$RESUME_LOG" &&
  grep -Fxq 'PASS  weather merge: building_train_8, all stations in one group, 16 and 32 envs each: merged == solo bit for bit, obs == oracle' "$RESUME_LOG" &&
  ! grep -q '^FAIL  ' "$RESUME_LOG" || { say "FAIL  invalid verification resume log"; exit 1; }
  cat "$RESUME_LOG" >>"$LOG"
  say "continuing verification from $RESUME_LOG on GPU ${SB_GPU:-all}"
else
  "$CP/tools/make_artifacts.sh" >>"$LOG" 2>&1 || { say "FAIL  make_artifacts"; exit 1; }
  "$CP/tools/build.sh" >>"$LOG" 2>&1 || { say "FAIL  build"; exit 1; }
  SB_LU_EXACT=1 "$CP/tools/build.sh" >>"$LOG" 2>&1 || { say "FAIL  build (exact)"; exit 1; }
fi
# Run trainer checks first so a broken actor fails before the long simulator checks.
if [ "${1:-}" = "--trainer" ] && [ -z "$RESUME_LOG" ]; then
  PL=${SB_PL_ROOT:-/data/puffer-work}/plverify
  if [ ! -d "$PL" ]; then git -C ${SB_PL_ROOT:-/data/puffer-work}/plclean worktree add --detach "$PL" "$(cat "$CP/patches/pufferlib_base_commit.txt")" >>"$LOG" 2>&1; fi
  ( cd "$PL" && git checkout -q -- . && git clean -fdq 2>/dev/null; true )
  "$CP/tools/deploy_to_pufferlib.sh" "$PL" --apply-patch >>"$LOG" 2>&1 || { say "FAIL  deploy"; FAIL=1; }
  chk "trainer: ./build.sh smart_buildings from base commit + patch + complete_port" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -w /work/plverify "$IMAGE" bash -lc "./build.sh smart_buildings 2>&1 | grep -q 'Built: ./puffer'"
  chk "trainer: actual mixed-head PPO kernels match finite-difference reference (masked heads, squash, clipping, ratio)" \
      python3 "$CP/tools/check_ppo_math.py" --checkout "$PL" \
      --out "$CP/results/ppo_math_$(date +%Y%m%d_%H%M%S)" --gpu "${SB_GPU:-0}"
  chk "shared policy: optional build from canonical sources" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -w /work/plverify "$IMAGE" bash -lc \
      "NVCC_EXTRA=-DSB_SHARED_POLICY ./build.sh smart_buildings puffer_shared > /tmp/shared_build.log 2>&1 || { tail -30 /tmp/shared_build.log; exit 1; }"
  RANDOM_START_RUN="verify_random_start_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: random starts, seven oracle clocks, full episodes and CUDA graph resets" \
      "$SK/.venv/bin/python" "$CP/tools/check_random_start.py" \
      --name "$RANDOM_START_RUN" --gpu "${SB_GPU:-0}"
  chk "trainer: complete asynchronous checkpoint resumes optimizer, RNG, simulator and pending rollout" \
      python3 "$CP/tools/check_full_resume.py" --gpu "${SB_GPU:-0}" \
      --name "verify_full_resume_$(date +%Y%m%d_%H%M%S)" --fixture "$CP/results/$RANDOM_START_RUN"
  chk "trainer: zone-state diagnostics masks, occupancy, reset, critic and spatial spread" \
      python3 "$CP/tools/check_zone_diagnostics.py" --out "$CP/results/zone_math_$(date +%Y%m%d_%H%M%S)" --gpu "${SB_GPU:-0}"
  chk "trainer: raw conditional Beta action graph and likelihood" \
      python3 "$CP/tools/check_conditional_beta.py" --checkout "$PL" \
      --out "$CP/results/conditional_math_$(date +%Y%m%d_%H%M%S)" --gpu "${SB_GPU:-0}"
  CB_RUN="verify_conditional_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: mean/concentration Beta asynchronous graph, episode reset and deterministic dispatch" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/results":/res -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "timeout 180 ./puffer_shared train --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.canonical=1 --env.zone_memory=1 --env.zone_critic=1 --env.max_steps=8 --env.desync=0 --env.reward_scale=1 --policy.hidden_size=64 --policy.num_layers=1 --policy.tanh_mean=0 --policy.tanh_squash=0 --policy.beta=1 --policy.beta_conditional=1 --policy.beta_mean_precision=1 --policy.beta_cn_tau=12 --policy.beta_cn_tau_ahu=48 --policy.beta_conc_max=12 --policy.value_shared=1 --policy.value_delay=1 --train.zone_critic=1 --train.valve_zone=1 --train.adv_norm=2 --train.role_adam=1 --train.horizon=16 --train.minibatch_size=256 --train.replay_ratio=1 --train.total_timesteps=1024 --train.update_diagnostics=1 --base.async=1 --base.eval_episodes=0 --base.cudagraphs=1 --base.checkpoint_interval=1 --base.log_dir=/res/$CB_RUN --base.checkpoint_dir=/res/$CB_RUN --vec.total_agents=16 > /res/$CB_RUN.log 2>&1 || { tail -25 /res/$CB_RUN.log; exit 1; }; grep -q 'raw-sample conditional Beta' /res/$CB_RUN.log && grep -q 'Beta independent mean and concentration' /res/$CB_RUN.log && ck=\$(find /res/$CB_RUN -name '*.bin' ! -name '*.mb' -print -quit) && test -n \"\$ck\" && python3 /cp/tools/tbread.py /res/$CB_RUN/smart_buildings/tb > /res/$CB_RUN.tb.txt && grep -q 'norm/nonfinite_ratio_frac .*last=0\$' /res/$CB_RUN.tb.txt && grep -q 'diag/reward_row_grad2_zone' /res/$CB_RUN.tb.txt && grep -q 'diag/entropy_row_grad2_zone' /res/$CB_RUN.tb.txt && mkdir /res/$CB_RUN/record && SB_RECORD=/res/$CB_RUN/record SB_EVAL_OUT=/res/$CB_RUN/score.json timeout 120 ./puffer_shared eval \"\$ck\" --headless --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.canonical=1 --env.zone_memory=1 --env.zone_critic=1 --env.max_steps=8 --env.desync=0 --env.reward_scale=1 --policy.hidden_size=64 --policy.num_layers=1 --policy.tanh_mean=0 --policy.tanh_squash=0 --policy.beta=1 --policy.beta_conditional=0 --policy.beta_mean_precision=1 --policy.beta_conc_max=12 --policy.value_shared=1 --policy.value_delay=1 --train.zone_critic=1 --base.eval_agents=1 --base.eval_episodes=1 --base.deterministic=1 --base.cudagraphs=-1 --train.horizon=16 > /res/$CB_RUN.eval.log 2>&1 && grep -q 'Beta independent mean and concentration' /res/$CB_RUN.eval.log && python3 /cp/tools/check_mean_record.py /res/$CB_RUN"
  chk "shared policy: derivatives, permutation equivariance, masked outputs, CUDA graph" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -w /work/plverify "$IMAGE" ./puffer_shared policycheck
  rm -f "$CP/results/verify_tr.sbs"
  chk "trainer: 64 agents train with 100-step episodes (episode_length 100), TensorBoard + live snapshot written" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/results":/res -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "rm -rf logs/smart_buildings/tb; SB_SNAPSHOT=/res/verify_tr.sbs SB_SNAP_EVERY=50 timeout 100 ./puffer train --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.max_steps=100 --base.cudagraphs=-1 --vec.total_agents=64 --train.total_timesteps=64000 > /tmp/t.log 2>&1; rc=\$?; tr -d '\\r' < /tmp/t.log | grep SPS | tail -1; [ \$rc -eq 0 ] || [ \$rc -eq 124 ] || exit 1; python3 /cp/tools/tbread.py logs/smart_buildings/tb > /tmp/tb.txt; grep -q 'env/episode_length .*last=100\$' /tmp/tb.txt && grep -q live/reward /tmp/tb.txt && grep -q loss/policy /tmp/tb.txt && [ -s /res/verify_tr.sbs ]" 
  mkdir -p "$ART/verify_multi"; rm -f "$ART/verify_multi"/*.sbb; for b in building_train_1__fairbanks_1 building_train_1__phoenix_5 building_train_2__atlanta_3; do ln -f "$ART/blobs/$b.sbb" "$ART/verify_multi/$b.sbb" 2>/dev/null || cp "$ART/blobs/$b.sbb" "$ART/verify_multi/"; done   # hard links: a symlink target is not mounted in the container
  chk "trainer: 3 scenarios in one batch (blob_dir; train_1's two weathers merge: 2 groups), per-scenario TensorBoard tags" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "rm -rf logs/smart_buildings/tb; timeout 100 ./puffer train --env.blob_dir=/art/verify_multi --env.groups=3 --env.max_steps=100 --base.cudagraphs=1 --vec.total_agents=96 --train.total_timesteps=96000 > /tmp/t.log 2>&1; rc=\$?; [ \$rc -eq 0 ] || [ \$rc -eq 124 ] || exit 1; grep -q 'group 1/2, weather 1' /tmp/t.log && python3 /cp/tools/tbread.py logs/smart_buildings/tb > /tmp/tb.txt; grep -q 'scenario/building_train_1__fairbanks_1/reward' /tmp/tb.txt && grep -q 'scenario/building_train_1__phoenix_5/reward' /tmp/tb.txt && grep -q 'scenario/building_train_2__atlanta_3/comfort_dev_K' /tmp/tb.txt && grep -q 'env/episode_length .*last=100\$' /tmp/tb.txt"
  ZVR_RUN="verify_zone_residual_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: nonlinear zone value residual, 48-step telemetry and Adam checkpoint restoration" \
      python3 "$CP/tools/check_zone_residual_trainer.py" --name "$ZVR_RUN" --gpu "${SB_GPU:-0}"
  ZPASS_RUN="verify_zone_pass_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: zone GAE per-pass cache, paired stats, mixed-group async graph" \
      python3 "$CP/tools/check_zone_pass_trainer.py" --name "$ZPASS_RUN" --gpu "${SB_GPU:-0}"
  AHU_RUN="verify_ahu_mlp_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: nonlinear AHU actor, action telemetry and role Adam in mixed-group async graph" \
      python3 "$CP/tools/check_ahu_mlp_trainer.py" --name "$AHU_RUN" --gpu "${SB_GPU:-0}"
  SHARED_RUN="verify_shared_$(date +%Y%m%d_%H%M%S)"
  chk "shared policy: multi-building graph rollout/backward, exact chain and frozen widths smoke" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/results":/res -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "timeout 100 ./puffer_shared train --env.blob_dir=/art/verify_multi --env.groups=3 --env.canonical=1 --env.residual=0 --env.comfort_shape=0 --env.guard_d1=0 --env.max_steps=100 --env.reward_scale=0.1 --policy.zero_init_actions=0 --policy.hidden_size=64 --policy.num_layers=1 --policy.init_logstd=-0.7 --policy.freeze_logstd=1 --policy.logstd_anneal=0 --policy.tanh_squash=1 --policy.squash_chain_floor=0 --base.checkpoint_optim=0 --base.eval_episodes=0 --base.cudagraphs=1 --base.log_dir=/res/$SHARED_RUN --base.checkpoint_dir=/res/$SHARED_RUN --vec.total_agents=96 --train.total_timesteps=49152 > /res/$SHARED_RUN.log 2>&1 && python3 /cp/tools/tbread.py /res/$SHARED_RUN/smart_buildings/tb | grep -q 'norm/nonfinite_ratio_frac .*last=0\$'"
  ZMLP_RUN="verify_zone_mlp_$(date +%Y%m%d_%H%M%S)"
  chk "shared policy: global valve credit, diagnostics and nonlinear Beta actor train through CUDA graph" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/results":/res -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "SB_ROLESTEP=1 SB_CREDIT=64 SB_VALVE_DIAG=1 SB_ZONE_CREDIT_AUDIT=64 SB_DETAIL_METRICS=1 timeout 120 ./puffer_shared train --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.canonical=1 --env.zone_memory=1 --env.zone_critic=1 --env.zone_night_share=1 --env.max_steps=100 --policy.hidden_size=64 --policy.num_layers=1 --policy.tanh_mean=0 --policy.beta=1 --policy.zone_mlp=1 --policy.value_shared=1 --policy.value_delay=1 --train.zone_critic=1 --train.valve_zone=1 --train.valve_global=1 --train.adv_norm=2 --train.role_adam=1 --train.update_diagnostics=1 --base.eval_episodes=0 --base.cudagraphs=1 --base.checkpoint_optim=1 --base.checkpoint_interval=1 --base.log_dir=/res/$ZMLP_RUN --base.checkpoint_dir=/res/$ZMLP_RUN --vec.total_agents=64 --train.total_timesteps=32768 > /res/$ZMLP_RUN.log 2>&1 && grep -q 'zone MLP on role Adam' /res/$ZMLP_RUN.log && grep -q 'valves use the building advantage alone' /res/$ZMLP_RUN.log && grep -q 'empty-building fuel billed' /res/$ZMLP_RUN.log && awk '/^SB_ZONE_CREDIT / && (\$4+\$8)>0 && (\$7+\$11)<0.0001*(\$4+\$8) {ok=1} END {exit !ok}' /res/$ZMLP_RUN.log && awk '/^SB_VALVE_CREDIT / && \$4 > 0 && \$17 > 0 {ok=1} END {exit !ok}' /res/$ZMLP_RUN.log && python3 /cp/tools/tbread.py /res/$ZMLP_RUN/smart_buildings/tb | grep -q 'scenario/building_train_1__fairbanks_1/weekend_day/gas_kw' && awk '/^SB_ZONE_MLP / && \$5 > 0 && \$4 > 0 {ok=1} END {exit !ok}' /res/$ZMLP_RUN.log && mkdir -p /res/$ZMLP_RUN/record && ck=\$(find /res/$ZMLP_RUN -name '*.bin' ! -name '*.mb' | head -1) && test -n \"\$ck\" && SB_RECORD=/res/$ZMLP_RUN/record SB_EVAL_OUT=/res/$ZMLP_RUN/score.json timeout 120 ./puffer_shared eval \"\$ck\" --headless --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.canonical=1 --env.zone_memory=1 --env.zone_critic=1 --env.max_steps=100 --policy.hidden_size=64 --policy.num_layers=1 --policy.tanh_mean=0 --policy.beta=1 --policy.zone_mlp=1 --policy.value_shared=1 --policy.value_delay=1 --train.zone_critic=1 --base.eval_agents=1 --base.eval_episodes=1 --base.deterministic=1 --base.cudagraphs=-1 > /res/$ZMLP_RUN.eval.log 2>&1 && test -s /res/$ZMLP_RUN/record/zone_feat.bin && test -s /res/$ZMLP_RUN/score.json"
  TEAM_RUN="verify_team_baseline_$(date +%Y%m%d_%H%M%S)"
  chk "trainer: team valve credit, reset, diagnostics and separate Adam state through CUDA graph" \
      docker run --rm "${DEVS[@]}" -v ${SB_PL_ROOT:-/data/puffer-work}:/work -v "$ART":/art -v "$CP/results":/res -v "$CP/tools":/cp/tools -w /work/plverify "$IMAGE" bash -lc \
      "SB_CREDIT=16 SB_VALVE_DIAG=1 timeout 180 ./puffer_shared train --env.blob=/art/blobs/building_train_1__fairbanks_1.sbb --env.canonical=1 --env.zone_memory=1 --env.zone_critic=1 --env.max_steps=8 --env.desync=0 --env.reward_scale=1 --policy.hidden_size=64 --policy.num_layers=1 --policy.tanh_mean=0 --policy.tanh_squash=0 --policy.beta=1 --policy.beta_cn_tau=12 --policy.beta_cn_tau_ahu=48 --policy.beta_conc_max=12 --policy.value_shared=1 --policy.value_delay=1 --train.zone_critic=1 --train.valve_zone=1 --train.valve_team_baseline=1 --train.adv_norm=2 --train.role_adam=1 --train.rew_clamp=0 --train.horizon=16 --train.minibatch_size=256 --train.replay_ratio=2 --train.total_timesteps=2048 --train.update_diagnostics=1 --base.async=1 --base.eval_episodes=0 --base.cudagraphs=1 --base.checkpoint_optim=1 --base.checkpoint_interval=1 --base.log_dir=/res/$TEAM_RUN --base.checkpoint_dir=/res/$TEAM_RUN --vec.total_agents=16 > /res/$TEAM_RUN.log 2>&1 || { tail -40 /res/$TEAM_RUN.log; exit 1; }; grep -q 'valve team baseline on' /res/$TEAM_RUN.log && awk '/^SB_TEAM / && \$4>0 {ok=1} END {exit !ok}' /res/$TEAM_RUN.log && ck=\$(find /res/$TEAM_RUN -name '*.bin.team' -print -quit) && test -n \"\$ck\" && python3 /cp/tools/check_team_checkpoint.py \"\$ck\" && python3 /cp/tools/tbread.py /res/$TEAM_RUN/smart_buildings/tb | grep -q 'norm/nonfinite_ratio_frac .*last=0\$'"
fi
[ "$FAIL" -eq 0 ] || { say "SOME CHECKS FAILED"; exit 1; }
G=/cp/artifacts/bin; A=/cp/artifacts
for bw in $SHAPES; do
  name=$(echo $bw | tr / _ | sed 's/_/__/'); name=${bw%/*}__${bw#*/}
  B=$A/blobs/$name.sbb; T=$A/traces/$name.step; R=$A/refs/$name.ref
  chk "$name: trace gate (oracle) + reset observation, 8 envs" dock "$G/gate $B $T 8 | grep -q 'SB GPU ENV STEP: OK'"
  chk "$name: multi-episode vs CPU env, host reset" dock "$G/gate $B $T 4 $R | grep -q 'MULTI-EPISODE: OK'"
  chk "$name: multi-episode vs CPU env, in-step restart" dock "SB_INSTEP=1 $G/gate $B $T 4 $R | grep -q 'MULTI-EPISODE: OK'"
  chk "$name: trace gate through a CUDA graph" dock "SB_GRAPH=1 $G/gate $B $T 8 | grep -q 'SB GPU ENV STEP: OK'"
  chk "$name: device solve == sb_lu_solve bit for bit (5, 40 envs)" dock "$G/lutest $B 5 2 | grep -q BIT-IDENTICAL && $G/lutest $B 40 2 | grep -q BIT-IDENTICAL"
  chk "$name: convection == sb_conv_apply bit for bit" dock "$G/convtest $B 100 0.1 4 | grep -q OK"
done
name=building_train_1__fairbanks_1; B=$A/blobs/$name.sbb; T=$A/traces/$name.step
chk "train_1: trace gate at 1, 64 and 256 envs" dock "for n in 1 64 256; do $G/gate $B $T \$n | grep -q 'SB GPU ENV STEP: OK' || exit 1; done"
chk "train_1: bit-exact-solve build (SB_LU_EXACT) passes the oracle gate and the multi-episode diff" dock "$G/gate_exact $B $T 64 | grep -q 'SB GPU ENV STEP: OK' && $G/gate_exact $B $T 4 $A/refs/building_train_1__fairbanks_1.ref | grep -q 'MULTI-EPISODE: OK'"
chk "train_1: convection extremes p=1e-4,0.5,1.0" dock "for p in 0.0001 0.5 1.0; do $G/convtest $B 100 \$p 4 | grep -q OK || exit 1; done"
# Training-only fp32 solve (SB_F32): not a gate build (the gate prints FAIL by design), but it must
# build, complete the trace gate and keep the drift bounded - a broken fp32 solve would diverge.
if [ -n "$RESUME_LOG" ] && grep -q '^PASS  train_1: training-only fp32 build (SB_F32)' "$RESUME_LOG"; then
  : # Already checked in the initial segment.
elif SB_F32=1 "$CP/tools/build.sh" >>"$LOG" 2>&1; then
  DRIFT=$(dock "$G/gate_f32 $B $T 64" 2>&1 | awk -F'field=' '/worst:/{split($2,a," "); print a[1]}')
  if awk -v d="$DRIFT" 'BEGIN{exit !(d+0 > 0 && d+0 < 0.05)}'; then
    say "PASS  train_1: training-only fp32 build (SB_F32) runs the gate, drift bounded (worst field $DRIFT K)"
  else
    say "FAIL  train_1: SB_F32 drift '$DRIFT' missing or out of bound"; FAIL=1
  fi
else
  say "FAIL  build (SB_F32)"; FAIL=1
fi
if [ -z "$RESUME_LOG" ]; then dock "$G/gate $B $T 128 | grep phase" | tee -a "$LOG"; fi
# new buildings from the generators (not shipped): bake + oracle trace + multi-episode, incl. a building with
# rooms too big for the shared-memory convection and one that needs the wide (320) action block
chk "generated buildings 0502/0511 (new floor plans, new weather): bake, oracle gate, multi-episode" \
    bash -c "'$CP/tools/new_scenarios.sh' --seeds '502 511' --weathers phoenix_5 --verify > /tmp/newsc.log 2>&1; [ \$(grep -c 'SB GPU ENV STEP: OK' /tmp/newsc.log) -eq 2 ] && [ \$(grep -c 'MULTI-EPISODE: OK' /tmp/newsc.log) -eq 2 ]"
chk "generated 4-floor building 0500 (278 continuous actions, 120 AHU-coupled zones): SB_CONT_BIG oracle gate" \
    bash -c "SB_CONT_BIG=1 '$CP/tools/build.sh' >/dev/null 2>&1; '$CP/tools/new_scenarios.sh' --seeds 500 --weathers fairbanks_1 --verify > /tmp/newsc2.log 2>&1; grep -q 'SB GPU ENV STEP: OK' /tmp/newsc2.log"
# multi-scenario batches: the per-scenario state list is current, and scenarios stepped solo, interleaved and
# concurrently (own streams) give bit-identical observations, rewards and temperature fields
chk "multi-scenario: sb_group_state.inc matches the sources" python3 "$CP/tools/gen_group_state.py" --check
B1="$A/blobs/building_train_1__fairbanks_1.sbb $A/traces/building_train_1__fairbanks_1.step"
B2="$A/blobs/building_train_2__atlanta_3.sbb $A/traces/building_train_2__atlanta_3.step"
B4="$A/blobs/building_train_4__atlanta_5.sbb $A/traces/building_train_4__atlanta_5.step"
B8="$A/blobs/building_train_8__atlanta_1.sbb $A/traces/building_train_8__atlanta_1.step"
chk "multi-scenario: train_1+train_2, solo vs interleaved vs concurrent streams (bit-identical)" \
    dock "$G/gate groups $B1 $B2 32 20 | grep -q 'GROUPS: OK' && SB_GT_PAR=1 $G/gate groups $B1 $B2 32 20 | grep -q 'GROUPS: OK' && SB_GT_PAR=1 $G/gate groups $B4 $B8 16 20 | grep -q 'GROUPS: OK'"
# weathers of one building merged into one group (per-env weather tables, per-lane / per-slab solver
# values): every env must give exactly the bits of its weather stepped alone, and the oracle's observations
for b in building_train_1 building_train_8; do
  WARGS=""; for t in "$ART"/traces_w/${b}__*.step; do nm=$(basename "$t" .step); WARGS="$WARGS $A/blobs/$nm.sbb $A/traces_w/$nm.step"; done
  chk "weather merge: $b, all stations in one group, 16 and 32 envs each: merged == solo bit for bit, obs == oracle" \
      dock "$G/gate wmerge 16 20 $WARGS | grep -q 'WMERGE: OK' && $G/gate wmerge 32 20 $WARGS | grep -q 'WMERGE: OK'"
done
# per-env episode restart (desynchronised episodes): restarting one env leaves every other env bit-identical,
# and the restarted env matches the whole-group restart (which the multi-episode gate checks against the CPU env)
chk "per-env restart: env 0 alone == group restart, other envs untouched (train_1, train_2)" \
    dock "$G/gate resetiso $A/blobs/building_train_1__fairbanks_1.sbb $A/traces/building_train_1__fairbanks_1.step 40 25 10 | grep -q 'RESETISO: OK' && $G/gate resetiso $A/blobs/building_train_2__atlanta_3.sbb $A/traces/building_train_2__atlanta_3.step 40 25 10 | grep -q 'RESETISO: OK'"
# fused multi-group solve (SB_FUSE=1 path): six buildings in one persistent solve == each stepped alone
MARGS=""; for bw in $SHAPES; do nm=${bw%/*}__${bw#*/}; MARGS="$MARGS $A/blobs/$nm.sbb $A/traces/$nm.step"; done
chk "fused multi-group solve: 6 buildings x 40 envs, fused == solo bit for bit (CUDA graph)" \
    dock "SB_GRAPH=1 $G/gate mgtest 40 12 $MARGS | grep -q 'MGTEST: OK'"
# viewer: scripted play -> snapshot -> terminal frame + headless window screenshot
VPY="$CP/viewer/.venv/bin/python"
if [ -x "$VPY" ]; then
  rm -f "$CP/results/verify_snap.sbs"
  chk "viewer: play replay writes a snapshot; terminal frame shows step 40/8640" \
      bash -c "docker run --rm ${DEVS[*]} -v '$CP':/cp -w /cp -e SB_SNAPSHOT=/cp/results/verify_snap.sbs -e SB_SNAP_EVERY=40 '$IMAGE' bash -lc '$G/gate $A/blobs/$name.sbb $T 4 play replay 40 0' >/dev/null 2>&1 && '$VPY' '$CP/viewer/sb_viewer.py' '$CP/results/verify_snap.sbs' --terminal --once | grep -q 'step 40/8640'"
  chk "viewer: pygame window renders the snapshot (headless screenshot > 20 kB)" \
      bash -c "'$VPY' '$CP/viewer/sb_viewer.py' '$CP/results/verify_snap.sbs' --screenshot '$CP/results/verify_view.png' >/dev/null 2>&1 && [ \$(stat -c %s '$CP/results/verify_view.png') -gt 20000 ]"
else
  say "FAIL  viewer env missing (uv venv --python 3.12 viewer/.venv && uv pip install --python viewer/.venv/bin/python pygame numpy tensorboard)"; FAIL=1
fi

[ $FAIL -eq 0 ] && say "ALL CHECKS PASSED" || say "SOME CHECKS FAILED"
exit $FAIL

# tools

Every script here is called by `../../run.sh`, and every one can also be run directly with its own
arguments. `run.sh <command> ...` forwards your arguments to the tool, so both forms accept the same
things; the tool is listed under each command.

Conventions that apply to all of them:

| variable | meaning | default |
|---|---|---|
| `SB_GPU` | device index, e.g. `0`; unset means all | all |
| `SB_IMAGE` | container image | `pufferai/puffertank:5.0` |
| `SB_PL_ROOT` | directory holding the PufferLib checkouts | `~/.cache/sb/pufferlib` |
| `PUFFERLIB` | one patched checkout | `$SB_PL_ROOT/plverify` |
| `SB_CP` | this `complete_port/` | derived from the script path |

## Build

| tool | usage | notes |
|---|---|---|
| `build.sh` | `[SB_LU_EXACT=1] [SB_F32=1] [SB_CONT_BIG=1] tools/build.sh` | builds the device test programs into `artifacts/bin`: `gate`, `convtest`, and with the flags `gate_exact`, `lutest`, `gate_f32`, `gate_big`. `SB_F32` and `SB_LU_EXACT` are mutually exclusive |
| `deploy_to_pufferlib.sh` | `deploy_to_pufferlib.sh <checkout> [--apply-patch]` | copies `ocean/smart_buildings`, `config/smart_buildings.ini` and `patches/src` into a checkout; `--apply-patch` also applies `patches/pufferlib_core.diff` |

## Gates

| tool | usage | notes |
|---|---|---|
| `verify_all.sh` | `verify_all.sh [--trainer]` | 35 gate groups, 67 checks, ends `ALL CHECKS PASSED`. `--trainer` also rebuilds the trainer in a fresh worktree and smoke-trains. `SB_VERIFY_RESUME=<log>` skips checks already recorded as PASS |
| `make_artifacts.sh` | `tools/make_artifacts.sh` | bakes the six gate shapes, their oracle traces and the CPU references. Idempotent |
| `check_ppo_math.py` | `--checkout <dir> --out <dir> --gpu <n>` | finite-difference reference for the mixed-head PPO kernels |
| `check_random_start.py` | `--gpu <n> --name <run>` | seven calendar starts, full episodes, graph resets |
| `check_full_resume.py` | `[--checkout <dir>] [--gpu <n>]` | asynchronous complete-checkpoint resume |
| `check_zone_*_trainer.py`, `check_ahu_mlp_trainer.py` | `--name <run> --gpu <n>` | zone diagnostics, zone pass cache, AHU actor |
| `check_mean_record.py`, `check_team_checkpoint.py`, `check_conditional_beta.py`, `check_split.py` | `--help` | helpers the gates call |

## Train

| tool | usage | notes |
|---|---|---|
| `train.sh` | `[<scenario>] [puffer args...]` | no scenario = every blob in `artifacts/train` (30 scenarios). `SB_TRAIN_BIN=puffer_shared` selects the shared canonical policy and builds it if missing. `SB_F32=1` selects the training-only fp32 solve. `SB_SNAPSHOT=<file> SB_SNAP_EVERY=<n>` writes viewer snapshots |
| `run_controlled.py` | `<name> --gpu <n> --runtime <checkout> [--binary puffer_shared] -- <puffer args...>` | one isolated run under `results/runs/<name>`, fresh, with `args.txt`, `manifest.json`, `train.log`, `status.json`. It owns the checkpoint and log paths. Forwards `SB_CREDIT`, `SB_VALVE_DIAG`, `SB_DETAIL_METRICS`, `SB_ROLESTEP`, `SB_ZONE_STATS`, `SB_ZONE_EMPTY_OBS`, `SB_ZONE_CREDIT_AUDIT`, `SB_ZONE_MLP`, `SB_TRAJ`, `SB_F32` |
| `run_train.sh` | `<name> [puffer args...]` | the older single-run wrapper |
| `run_pure_chain.sh` | `<name> <steps> "<set> <set> ..." [puffer args...]` | segments across blob sets; `START=` and `FIRST=` continue a chain, `EVAL_FOLLOW=0` skips scoring |
| `run_rotation.sh` | `<name> <steps> <segments> <envs> "<set> ..." [puffer args...]` | the same, rotating the set each segment |
| `with_gpu.sh` | `with_gpu.sh <index> -- <command...>` | takes the per-device lock and runs the command. `with_gpu.sh` alone lists devices and the locks held |

## Evaluate and score

| tool | usage | flags |
|---|---|---|
| `eval_policy.sh` | `<set dir under artifacts/> <out.json under results/> [model|-] [puffer args...]` | `SB_BIN` selects the binary (default `puffer`; the shared policy build is `puffer_shared`), `SB_GPU` pins the device, `SB_EVAL_FORCE=1` lets a retry replace a failed attempt's log, `SB_EVAL_SPLITS` scores a set in parts |
| `score_eval90.py` | `--run <run dir> --runtime <checkout> --set <set dir> --out <dir> --gpu <n> --checkpoints <step>=<path> [...]` | `--ckpt-bytes` when the weight file is not the default size, `--set` relative to `complete_port/`. A relative `--out` lands under `results/`. Writes per-step recordings |
| `merge_eval_json.py` | `<out.json> <part.json> [...]` | merges the parts of a split panel |
| `score_cpu_pid.py` | `<list file> <out dir> [workers] [steps]` | the organiser `pid_water` baseline on CPU, one month per scenario |
| `decode_flags.py` | `<run dir>` | prints the decode-relevant flags of a run, from `args.txt` or `flags.txt`; this is what `./run.sh eval` uses |
| `export_submission.sh` | `<checkpoint.bin> <out dir> [res_scale] [res_dscale] [hidden] [layers]` | writes a self-contained submission folder |
| `ckpt_eval_watch.sh` | `--help` | watches checkpoints and scores each ladder step |

## Data

| tool | usage | flags |
|---|---|---|
| `bake_training_set.sh` | `tools/bake_training_set.sh` | the 30 shipped scenarios into `artifacts/train`. `BAKE_JOBS` workers (default 3) |
| `bake_eval_set.sh` | `<set list> [jobs]` | any scenario list from `data/scenarios/sets/`; generates missing configs with the organiser generator, installs generated buildings the starter kit lacks, then bakes. `SB_GEN_BUILDINGS` overrides where those YAMLs are read from |
| `new_scenarios.sh` | `--seeds "500 501" --weathers "fairbanks_1 phoenix_5" [--verify]` | new buildings from the floor generator; `--verify` also bakes and gates them |
| `bench.sh`, `bench_layout.sh`, `roofline.py` | `--help` | throughput measurements |
| `gen_group_state.py` | `tools/gen_group_state.py` | regenerates `gpu/sb_group_state.inc` from the sources; the gate checks it |

## Inspect

| tool | usage | notes |
|---|---|---|
| `tbread.py` | `<tb dir> [tag substring ...]` | scalars without TensorBoard: latest values, or the full series for the matching tags |
| `live_tb_from_logs.py`, `timeline.py`, `run_stats.py` | `--help` | read a run's log: live scalars, phase timeline, steps/s and memory |
| `tb.sh` | `[log dir]` | TensorBoard on port 6006 |
| `watch.sh` | `<scenario> [--policy replay\|mid\|random] [--steps N] [--fps F] [--envs N] [--every K] [--terminal]` | live viewer from a simulator snapshot |

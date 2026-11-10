> **This page is the port's development notes.** It describes `complete_port/` as it was
> built, and it names files of the working tree (`results/`, `AGENTS.md`, `plan.md`) that are
> not part of this repository. Start at [`../README.md`](../README.md); the runs are in
> [`../experiments/`](../experiments/README.md) and the gate record in
> [`../docs/VERIFICATION.md`](../docs/VERIFICATION.md).

# complete_port: Smart Buildings simulator on the GPU (PufferLib 5.0)

GPU port of the starter-kit simulator, checked against the unmodified Python simulator (the oracle). One
RTX 3090 simulates ~13K env-steps/s (`building_train_1`, 256 envs, exact build) versus ~40/s for the Python simulator on
one core, and the trainer runs all 30 shipped scenarios (960 agents) at **~17.4K steps/s on the exact build, ~23.4K
with the opt-in `SB_F32` solve** - or ~35.7K with a second job on the 3060 (~26.7K on the exact build; see "Two GPUs, two jobs").
This is the default way to simulate, train, and build new buildings in this workspace.

## Quick start

Needs: Docker with GPU access and the image `pufferai/puffertank:5.0`; `starter-kit/.venv` (Python 3.12, see `AGENTS.md`);
for training, a PufferLib 5.0 git checkout. Run everything from `complete_port/`.

```bash
tools/verify_all.sh --trainer      # builds everything, runs the checks, ends with "ALL CHECKS PASSED" (~18 min)
tools/new_scenarios.sh --seeds "500 501" --weathers "fairbanks_1 phoenix_5" --verify   # new buildings -> artifacts/blobs/*.sbb
```

Train. **Default = multi-building**: all 30 shipped scenarios in one batch, 32 envs each (960 agents). The 5 weathers of
each building share one group, so 6 groups of 160 envs: **~17.4K steps/s exact, ~23.4K with `SB_F32`** (~16-18 GB VRAM;
measured from each trainer's final `Steps`/`Uptime`: 983,040 steps in 56.5 s / 42.0 s, see `docs/HISTORY.md`):

```bash
tools/bake_training_set.sh                                           # once: 30 blobs -> artifacts/train/ (~1 min)
tools/deploy_to_pufferlib.sh /path/to/pufferlib --apply-patch        # once: checkout at patches/pufferlib_base_commit.txt (omit --apply-patch if patched)
PUFFERLIB=/path/to/pufferlib tools/train.sh                          # multi-building, defaults from config/smart_buildings.ini
PUFFERLIB=/path/to/pufferlib tools/train.sh building_train_1__fairbanks_1   # one scenario, 256 envs
SB_F32=1 PUFFERLIB=/path/to/pufferlib tools/train.sh                 # training-only fp32 solve: 1.34x (23.4K vs 17.4K steps/s, 960 agents)
```

**PPO settings that matter here** (`config/smart_buildings.ini`, see `docs/HISTORY.md` Session 9): the trainer is built fp32
(bf16 log-prob mismatch over ~250 heads saturates PPO's clip), `ratio_per_dim = 1` (per-head ratio), `tanh_mean = 1`,
`init_logstd = -1`, `desync = 1` (random first-episode cut, per-env restarts), `reward_scale = 0.1` (trainer signal only).

**The policy learns only by acting.** Do not put a competition baseline into the action: no residual on `pid_water`, no action head initialized so the policy starts as that controller, no comfort guard, no comfort shaping. Those baselines are scores to beat. The public target is 0.00603 on the Kaggle leaderboard, and the policy has to generalize past the scenarios it trained on. The experiment log is `Experiments.md` at the workspace root.

**Two GPUs, two jobs.** The second card is a second *experiment slot*, not a shard of the same batch: one job per GPU,
each run at full speed. Re-measured 2026-10-16 (both jobs launched together, same fp32 binary, 1.2M timesteps each,
quoted from each trainer's final `Steps`/`Uptime`): the 3060 ran the **full 960-agent batch** (its blob dir has 3
buildings -> 3 groups x 320 envs, 10.5-11.1 of its 12 GB) at **74.1 s = 13.3K steps/s**, while the 3090 (6 groups x
160) ran the same 983,040 steps at **43.9 s = 22.4K steps/s** - only **+4.6%** over its solo 42.0 s (reproducing the
+4.7% of the first measurement). **The pair sustains ~35.7K steps/s aggregate = 2.05x the 3090's exact-build solo
(17.4K) and 1.52x its fp32 solo (23.4K).** The 3060 is ~0.57 of a 3090 slot because the step is latency-bound (its
per-group step is 1.35-1.47x the 3090's at 160 envs, not the 2.6x its memory roofline predicts), and it holds
120-128 W of its 140 W cap doing it. With the **exact build** the 3060 needs the smaller batch: fp64 at 960 agents
fails in `cudaGraphInstantiate` at epoch 2, so use `--env.groups=3 --vec.total_agents=480` - measured at **9.0K
steps/s** for a 26.7K pair aggregate (**1.53x** the exact solo; training VRAM 6.3-6.6 of 12 GB, and the one-off
graph-capture peak is why the 960-agent layout does not fit). Every tool honours `SB_GPU=<index>` (`common.sh` turns it into
`--gpus device=N`), and
`tools/with_gpu.sh <index> -- <command>` also takes the per-device flock, so two jobs can never land on the same device
by accident (it passes its environment through, so put `VAR=value` on the wrapper, not after `--`):

**Read the trainer's own clock, not the `SPS` field.** The dashboard/TensorBoard `SPS` counts *issued* steps against
CPU-side phase times (the env phase reads `0 ms` because the step is enqueued and the CPU runs ahead of the GPU until a
buffer fills). It is wrong in every regime: four consecutive frames of the fp64 run behind these numbers read
7.8K / 205.7K / 10.7K / 209.1K while the true rate was 17.4K steps/s (the 3060 once claimed 117K against a real ~13K).
Quote the final frame's `Steps`/`Uptime` pair, or the wall for a fixed `total_timesteps`; note `total_epochs =
total_timesteps / (total_agents * horizon)` rounds down (a 960-agent `--train.total_timesteps=2400000` run is 9 epochs =
2,211,840 steps).

```bash
tools/with_gpu.sh                                   # devices, memory, which per-GPU locks are held
tools/with_gpu.sh 1 -- tools/eval_policy.sh train results/eval.json    # evals on the 3060 while training goes on
PUFFERLIB=/path/to/pufferlib SB_F32=1 tools/with_gpu.sh 0 -- tools/train.sh            # 3090: full batch (23.4K steps/s)
PUFFERLIB=/path/to/pufferlib SB_F32=1 tools/with_gpu.sh 1 -- tools/train.sh \
    --env.blob_dir=/art/3060                                                           # 3060 fp32: full batch fits too (13.3K)
PUFFERLIB=/path/to/pufferlib tools/with_gpu.sh 1 -- tools/train.sh \
    --env.blob_dir=/art/3060 --env.groups=3 --vec.total_agents=480                    # 3060 exact: 480 agents (960 does not fit)
```

`tools/train.sh [scenario] [puffer args]` runs `./puffer train` in the container with `--base.cudagraphs=1` from the ini
(`--env.blob_dir`/`--env.blob`, `--env.groups`, `--vec.total_agents`, `--train.*` override it). `env.max_steps=N` shortens episodes
for smoke runs. Checkpoints go to `<checkout>/checkpoints/smart_buildings/` every `checkpoint_interval` epochs. The trainer's own
final evaluation is off (`eval_episodes = 0`; its default waits for 10000 whole episodes).

### Complete r116-style checkpoint continuation

For the supported single-GPU, asynchronous r116 shared-policy configuration, `base.full_checkpoint=-1`
automatically writes a complete `<step>.bin.full` after the ordinary weights, Muon and role-Adam files.
`base.full_checkpoint=1` requires support and fails if the configuration is unsupported. The `.full` file
is the commit marker: it includes the prefetched rollout, optimizer state, recurrent and exploration state,
simulator state, random streams, calendar state and episode memory. These files can be large. A crash before
the marker leaves the preceding committed checkpoint as the recovery point. The unfinished epoch is replayed.

After explicitly approving a continuation, use the **same frozen executable, physical GPU, CUDA environment,
scenario blobs, flags, total training budget and annealing schedule**. Pass the committed `.bin` path as
`--base.load_model_path`, or use `--base.load_model_path=latest` with the original `base.checkpoint_dir` and
`base.run_id`: this selects the highest numbered committed `.full` in that run. A run-directory lock prevents
two complete-checkpoint processes from writing concurrently. Incomplete next-checkpoint components are
preserved with `.orphan.<pid>.<serial>` names. Changed or damaged components and unsupported configurations abort.
Logs retain their original step numbers; elapsed time and logging cadence depend on the restart.

Historical checkpoints without `.full`, including files created by older running binaries, do not contain the
state needed for exact continuation. The default refuses such training resumes. `base.allow_partial_resume=1`
explicitly opts into the old optimizer-only behavior when full checkpointing is disabled or unsupported; it is
not an exact continuation. Neither checkpoint creation nor this mechanism grants approval to launch a resume.

**How a multi-scenario batch works.** A group is one building with one or more of its weathers. At start-up the env compares
the blobs: scenarios whose blobs differ only in the per-weather sections (LU values, convection coefficient, ambient series,
calendars, see `sb_weather_sections` in `gpu/smart_buildings.cu`) are merged into one group, checked byte for byte, keyed on nothing
but the blob contents, so new buildings and stations merge automatically (`env.merge_weathers = 0` gives one group per scenario).
A group has one geometry and one solver pattern; every env reads its own weather's values (`envw`), and its results are bit-identical
to that weather run alone (`gate wmerge`). Groups step concurrently on their own CUDA streams, forked from and joined into the
rollout stream (graph-capturable). Group state is swapped by byte image of the variables listed in `gpu/sb_group_state.inc`
(generated by `tools/gen_group_state.py`; `verify_all.sh` checks it is current and that solo, interleaved and concurrent groups are
bit-identical). Cost: VRAM grows with the number of buildings (plus ~0.12 GB of solver values per extra weather) and the envs;
speed with the total envs. **32 envs per scenario is the efficient size**: each weather then fills whole 32-lane warp slabs of the
solve. `blob_dir` is ordered building-first, so a prefix of any length spans all buildings (`--env.groups=N` = number of scenarios,
`--vec.total_agents`). Observation/action layout differs per building (padded to 1024 / 256+16 with an action mask).

### Random calendar starts

`env.random_start=1` independently selects one of seven offsets, each 288 five-minute steps apart, for every
environment and every reset, including the initial reset. For the shipped timezone-naive 06:00 starts these are
seven consecutive days at 06:00. A timezone-aware scenario crossing daylight saving may have a shifted wall time.
It disables `desync`'s initial truncation;
normal episodes have the scenario's full length. `env.max_steps` can still shorten episodes for verification.
`env.random_start_seed` controls a separate reproducible calendar RNG; it is not automatically set by `base.seed`.
Starts remain at the same time of day, so without desync the environments share a clock hour and finish together.
The selected dates shift the weather window as well as the weekday. They are training inputs, not a replacement
for the competition's original fixed-start scenario.

The flag requires separately baked blobs with six extra days of continuous weather and calendars. For each
scenario, from `starter-kit/`, use its normal scenario name in a new output directory:

```bash
.venv/bin/python ../complete_port/pysim/bake_scenario.py --out ../complete_port/artifacts/random_start \
  --name building_train_1__fairbanks_1 --random-start-days 7 \
  smart_buildings/training_scenarios/building_train_1/fairbanks_1/config.gin
```

For multiple scenarios, bake each into that separate directory and point `env.blob_dir` there. Existing blobs
are rejected when `random_start=1`. Rebuild the trainer from the current sources before using the flag; older
frozen binaries do not contain it. Source weather must cover the extended window. TensorBoard's
`scenario/.../start_day_N/initial_draws`, `later_draws`, and completed-episode components use `N=0..6` as
**days after the original scenario start**, not necessarily a weekday name. The flag is incompatible with
`prestart_explore`, `plant_pair` and `comfort_shape`.

## Viewer and TensorBoard

**See it run** (window with floor-plan temperature map, zone table, reward/energy/comfort values, AHUs; or a live terminal table):

```bash
tools/watch.sh building_train_1__fairbanks_1 --policy replay            # pygame window
tools/watch.sh building_train_1__fairbanks_1 --policy random --terminal  # values in the terminal
```

Policies: `replay` (the oracle trace's actions, looped), `mid`, `random` (`mid`/`random` need no trace). `--fps` paces the sim
(default 20 steps/s, 0 = max), `--envs`, `--every K`. Window keys: SPACE freeze display, M map mode (temperature / band
deviation), L labels, `[` `]` `-` `=` colour range, S screenshot, Q quit. The viewer runs on the host from `viewer/.venv`
(pygame, numpy, tensorboard; create with `uv venv --python 3.12 viewer/.venv && uv pip install --python viewer/.venv/bin/python
pygame numpy tensorboard`).

The env writes snapshots, the viewer only reads them, so the same viewer attaches to anything that sets `SB_SNAPSHOT=<file>`
(optionally `SB_SNAP_ENV` = which env, default 0, and `SB_SNAP_EVERY` = every Kth step): a training run (`tools/train.sh` passes it
through; with graphs on a snapshot is written once per epoch, with `--base.cudagraphs=-1` every Kth step) or a trained checkpoint:

```bash
SB_SNAPSHOT=results/live.sbs ... ./puffer eval <checkpoint.bin> --headless --env.blob=... --base.cudagraphs=-1 --base.eval_agents=32
viewer/.venv/bin/python viewer/sb_viewer.py results/live.sbs [--terminal]
```

**Metrics** (TensorBoard is on by default, `[tb] enabled` in the ini): `tools/tb.sh [logdir]` -> http://localhost:6006
(`tools/tbread.py <logdir>` prints the latest scalars without a browser). Tags: `env/*` per-episode means over finished
episodes (`perf`, `episode_return`, `episode_length`, `comfort_dev_K`, `worst_dev_K`, `out_of_band_frac`, `productivity`,
`energy_cost_norm`, `carbon_norm`, `elec_kwh`, `gas_kwh`, `n`); `live/*` step-level means over all envs and steps of the epoch
(`reward`, comfort, energy, carbon, `elec_kw`, `gas_kw`), available long before the first 8640-step episode ends; `loss/*`, `perf/*`,
`util/*`, `SPS`. The diagnostics are read from the reward kernel's intermediates and never feed back into the reward.

## Layout

| path | what |
|---|---|
| `gpu/` | canonical GPU sources: `smart_buildings.cu` (env + parity gate), `sb_gpu_kernels.cuh`, `sb_gpu_lusolve.cuh` (solver), `sb_gpu_convect2.inc`, `sb_gpu_airside.inc`, tests `sb_gpu_lutest.cu`, `sb_gpu_convtest.cu`; `gpu/legacy/` = early probes (unused) |
| `csim/` | CPU C core: the bit-exact reference; `make test_episodes` builds the CPU multi-episode reference |
| `pysim/` | Python tooling on `starter-kit/.venv`: `bake_scenario.py` (scenario -> `.sbb` blob), `dump_step.py` (oracle trace), checks |
| `ocean/smart_buildings/` | deploy tree PufferLib compiles (symlinks into `gpu/`, `csim/`) |
| `patches/` | `pufferlib_core.diff` (mixed action heads, mask hook, `build.sh` case, TensorBoard for smart_buildings, `puf_live_log` hook) against `pufferlib_base_commit.txt`; `src/tblog.h` |
| `config/smart_buildings.ini` | trainer config |
| `viewer/` | `sb_viewer.py` (pygame + terminal snapshot viewer), `.venv` |
| `tools/` | `verify_all.sh`, `new_scenarios.sh`, `make_artifacts.sh`, `build.sh`, `deploy_to_pufferlib.sh`, `watch.sh`, `train.sh`, `tb.sh`, `tbread.py`, `with_gpu.sh` (pin/serialise a job on one GPU); measurement: `roofline.py` (offline cost model), `bench.sh` (gate sweep, `--nsys` kernel summary), `bench_layout.sh` (trainer layout A/B), `timeline.py` (per-step critical path + stream overlap from an nsys trace), `run_stats.py` (summarize runs) |
| `artifacts/` | generated, git-ignored: `blobs/`, `traces/`, `refs/`, `bin/`, `gen/` |
| `results/` | `verify_*.log` |
| `docs/HISTORY.md` | full development log (milestones, measurements, rejected ideas) |
| `pufferlib_cpu_env.h`, `plan.md`, `pufferl_mixed_heads.diff` | historical (CPU env, original plan, early diff) |

Edit only `gpu/*` and `csim/*`. Never edit the PufferLib checkout or `artifacts/` by hand.

## Build flags

| command | result |
|---|---|
| `tools/build.sh` (default) | `gate` (parity gate, fast solve) and `convtest` |
| `SB_LU_EXACT=1 tools/build.sh` | `gate_exact` and `lutest`: solve bit-identical to the CPU core (verification only; 15-35% slower) |
| `SB_CONT_BIG=1 tools/build.sh` | 320 continuous action heads instead of 256 (use `-DSB_CONT_BIG` for the trainer build too) |
| `SB_F32=1 tools/build.sh` | `gate_f32`: **training-only fp32 solve** (`SB_SOLVE_F32`).  Same source, fp32 `work`/`zout` + fp32 solve arithmetic; coefficients and everything outside the solve stay double.  ~1.15-1.27x per step (measured, `results/perf/P3_fp32_solve_20261016.md`), drift vs the oracle bounded at ~1e-4 (obs) and never gated.  The default build is unchanged and stays the one used for evaluation and the submission. |

Gate usage (inside the container): `artifacts/bin/gate <blob.sbb> <trace.step> <nenv> [reference.ref]`; also
`gate wmerge <nper> <steps> <blob> <trace> ...` (weathers of one building in one group == each alone, bit for bit, + oracle obs),
`gate mgtest <nper> <steps> <blob> <trace> ...` (fused multi-group solve == solo, with timing), `gate groups ...`. Runtime variables:
`play <replay|mid|random> [steps] [fps]` after the nenv argument (viewer feed, see above), `SB_GRAPH=1` replay through a CUDA graph, `SB_INSTEP=1` in-step episode restart, `SB_VERBOSE=1`, `SB_OLDCONV=1` original
convection kernel, `SB_CONV_PER_ENV=1`, `SB_CONV_GLOBAL=1`. Env (trainer and gate): `SB_FUSE=1` one fused solve for all groups (bit-identical,
measured slower), `SB_LU_BLOCKS` solver blocks (164), `SB_LU_TPB` solver threads per block (128), `SB_FORCE_MW=1|2` solver value mode
(A/B timing only), `SB_LU_CHUNK2=1` twice the terms per flag wait (fp32 build only).  All measured neutral or worse on the 3090
(1.15-1.27x comes from `SB_F32`, nothing else; see the P3 report).

## Performance (RTX 3090, default build)

`building_train_1`, gate timing (one scenario):

| envs | ms/step | env-steps/s |
|--:|--:|--:|
| 64 | 10.1 | 6.4K |
| 128 | 12.5 | 10.3K |
| 256 | 19.1 | 13.4K |
| 512 | 38.3 | 13.4K |

Trainer, all 30 shipped scenarios (`tools/train.sh`, 6 building groups): 960 agents (default) **17.4K steps/s, ~17.6 GB** with
the default (bit-exact) build and **23.4K steps/s, 16.5 GB** with the training-only fp32 solve (`SB_F32=1`) - 1.34x, measured
2026-10-16 on an idle machine from each trainer's final `Steps`/`Uptime` (983,040 steps in 56.5 s / 42.0 s;
`results/perf/P3_fp32_solve_20261016.md`; the first A/B's 10.5K/14.4K readouts carried docker/build time and the older-SPS
trap, the ratio stands). A second job on the 3060 adds 13.3K steps/s (fp32, full batch, +4.6% on the 3090) - see
"Two GPUs, two jobs". Cost scales with control volumes and nonzeros of the factors
(the six shipped buildings at 256 envs: 13.9-39.6 ms/step, train_2 the largest). The wall is **memory bandwidth**: the solve
streams 8 B of state per term (`nnzL+nnzU+nnzZ`) per env-step, so its DRAM floor (state/936 GB/s) is 2.4x the FP64-pipe
floor (terms/278e3 us) - every generated building is DRAM-bound. Measured steps land at 1.41-1.69x that floor (62-70% of
peak bandwidth) for the big buildings, where the solve is most of the step; for the smallest (train_1 @256) the step is
solve 7.9 ms + field side 5.5 ms + sequencer/air side/reward ~2 ms on the main stream, with `convect2_k` (10.5 ms) hidden
on stream2 (nsys, fp32 build - P3 report).

**Batch layout** (measured, see `results/perf/P0_P1_measurements_20261016.md`): with >=160 envs per group and every weather a
multiple of 32 envs the layout is worth <10% - 32, 48 and 64 envs/scenario measured 4360/4136/4284 SPS on identical blobs.
32 envs/scenario (one 32-column slab per weather, value mode 2, no padding, ~24-29 MB/agent VRAM) matches the SPS of the
48/weather layout with 2/3 of the agents and 3/4 of the VRAM (rot2 5714 vs 5591 SPS; rot1 6821 vs 6856 SPS). Below ~160 envs
per group the cost explodes (32 envs, 1 env/lane: 5.7x the floor; 160 envs, 5/lane: 1.66x) and the 4-8 envs/scenario `diverse*`
sets run at 2.2x the floor. Every group prints its layout at startup (`envs, ld, env(s)/lane, value mode`) and warns when it is
in the slow regime - with `SB_FORCE_MW=1|2` the value mode can be forced for A/B timing.

## What "correct" means (all checked by `verify_all.sh`)

- Oracle trace gate, six shipped shapes: observations 0 differing values, reset observation exact, temperature field
  <= 3.2e-11 K, reward <= 7.5e-9.
- The air-side correction is folded into the solve (forward, sensed values `R z`, q, backward from `z + Z q`; the oracle does a
  second solve): same round-off class as the FMA solve; full 8640-step episode 0 differing observations, reward <= 9.1e-13.
- Weathers merged into one group, and groups in one fused solve, are bit-identical to each scenario run alone (`gate wmerge`,
  `gate mgtest`).
- Multi-episode diff against the CPU env (host reset and in-step restart): observations exact, reward <= ~1e-12 per step.
- The solve replays the baked SuperLU factors (no cuDSS) and is deterministic. The default build differs from the CPU replay by
  ~2e-14 (full 8640-step episode: 0 differing observations, reward <= 9e-13); `SB_LU_EXACT` is bit-identical.
- Convection (random stream + swaps) is bit-exact against the CPU core; the step is CUDA-graph capturable.

## New buildings

`tools/new_scenarios.sh --seeds "<seeds>" --weathers "<presets>" [--verify]` runs the starter-kit generators
(`run_floor_generator`, `run_scenario_generator`) into `artifacts/gen/` (the starter-kit is not modified) and bakes each
building x weather into `artifacts/blobs/<building>__<weather>.sbb`. `--verify` adds an oracle trace, a CPU reference and the
gates. Tested up to 1.15M control volumes / 120 zones / 278 continuous actions.
Limits (the env reports them at start-up): 1024 observation values, 256 continuous actions (320 with `SB_CONT_BIG=1`),
16 discrete, 256 AHUs, 65 535 cells per room. A scenario group is one blob; several groups can share a batch (see Quick start). Shrink `width`/`height` in a generated building YAML for faster, shipped-sized buildings.

## Shipped scenarios

```bash
cd ../starter-kit
.venv/bin/python ../complete_port/pysim/bake_scenario.py --out ../complete_port/artifacts/blobs --name <b>__<w> \
    smart_buildings/training_scenarios/<b>/<w>/config.gin
.venv/bin/python ../complete_port/pysim/dump_step.py smart_buildings/training_scenarios/<b>/<w>/config.gin <out.step> 30
```

## Troubleshooting

- `scenario needs N continuous ...` -> build/train with `SB_CONT_BIG=1`, or shrink the building.
- `BUILD FAILED: <bin>` from `tools/build.sh` -> compile error; run the printed nvcc line, never trust a stale binary.
- Trainer build `fatal error: tblog.h` -> deploy with `tools/deploy_to_pufferlib.sh` (installs `patches/src/tblog.h`).
- Gate numbers differing between identical runs -> must not happen (the step is deterministic); treat as a bug.

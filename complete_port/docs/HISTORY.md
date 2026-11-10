# complete_port — Smart Buildings → PufferLib 5.0 (native, GPU-batched)

Port of the `smart_buildings` simulator to a PufferLib 5.0 native env, gated on
trajectory parity against the unmodified Python starter kit (the oracle).

Canonical sources live here and are SYNCed into `/data/puffer-work/plclean`
(the PufferLib 5.0 checkout, branch `5.0`).

## Reproduce / verify everything from this directory

`complete_port/` is self-contained (plus the unmodified `starter-kit/` as the oracle and a PufferLib 5.0
checkout for the trainer):

| path | what |
|---|---|
| `gpu/` | canonical GPU sources: `smart_buildings.cu` (env + parity gate), `sb_gpu_kernels.cuh`, `sb_gpu_lusolve.cuh`, `sb_gpu_lutest.cu`, `sb_gpu_convtest.cu`; `gpu/legacy/` = early standalone probes (cuDSS era, unused) |
| `ocean/smart_buildings/` | the deploy tree PufferLib compiles (symlinks into `gpu/` and `csim/`, plus `smart_buildings.h`, `test_env_cpu.c`) |
| `patches/` | `pufferlib_core.diff` (mixed heads, mask hook, `build.sh` case) against `pufferlib_base_commit.txt`, and `src/tblog.h` |
| `config/smart_buildings.ini` | trainer config (128 agents) |
| `tools/` | `make_artifacts.sh` (bake blobs, oracle traces, CPU references into `artifacts/`), `build.sh`, `deploy_to_pufferlib.sh`, `verify_all.sh` |
| `artifacts/` | generated, git-ignored |

`tools/verify_all.sh --trainer` regenerates the artifacts from the starter-kit, builds the gate/lutest/convtest,
runs 53 checks (oracle gate, multi-episode host reset and in-step restart, CUDA-graph replay, solve and convection
bit-exactness for all six shapes, scaling runs, the `SB_LU_FAST` build, weather merge, multi-scenario batches solo
vs interleaved vs concurrent streams, the fused multi-group solve, the `SB_F32` drift bound, the viewer) and finally
builds `./puffer` in a fresh PufferLib worktree from base commit + patch + `ocean/smart_buildings`, then smoke-trains
with episode restarts.  Last run: ALL CHECKS PASSED (53/53, `results/verify_20261016_182836.log`).

## Layout (older notes)

| path | role |
|---|---|
| `plan.md` | the agreed plan (milestones M0–M5, gates L0–L4) |
| `pysim/` | Python tooling; runs on `starter-kit/.venv` (3.12), imports starter-kit unchanged |
| `pysim/sbb.py` | SBB1 self-describing binary blob (write/read); `python sbb.py` self-tests |
| `pysim/bake_scenario.py` | bakes one scenario → `<name>.sbb` + `<name>.json` |
| `pysim/bake_env.py` | M0b extension: HVAC, env contract, reward, schedule, exogenous |
| `pysim/check_l0.py` | L0 gate: re-parse the blob, compare to a freshly-loaded live scenario |
| `pysim/probe_*.py` | ground-truth dumps used to design the bake (no assumptions) |
| `csim/` | standalone C core (no CUDA): blob loader, FDM step, HVAC, reward, obs |
| `csim/baked/` | generated blobs (regenerate with `bake_scenario.py`; not committed) |
| `ocean/smart_buildings/` | PufferLib env wrapper (CPU `.h` + GPU `.cu`) |
| `config/`, `patches/` | PufferLib config + core patches |

## Run (from `starter-kit/`)

```bash
# bake one scenario (writes ../complete_port/csim/baked/<b>__<w>.sbb)
.venv/bin/python ../complete_port/pysim/bake_scenario.py \
    --out ../complete_port/csim/baked --name <b>__<w> \
    smart_buildings/training_scenarios/<b>/<w>/config.gin

# L0 gate for it
.venv/bin/python ../complete_port/pysim/check_l0.py \
    ../complete_port/csim/baked/<b>__<w>.sbb \
    smart_buildings/training_scenarios/<b>/<w>/config.gin
```

## Ground-truth findings (measured, not assumed)

- **The thermal system is contractive, not chaotic.** A 1e-12 K perturbation of
  the field grows to only 1.7e-12 K after 300 steps; two fresh identical envs
  are bit-identical. So an accurate-but-different solver is acceptable: swapping
  only the SuperLU fill-reducing ordering drifts the 300-step field by 1.4e-10 K.
  This frees the C core from replaying SuperLU's equilibrated factorization.
- **A is SPD.** `_conduction` is symmetric and strictly diagonally dominant, so
  the C core factorizes `A[perm][:,perm]` with a plain Cholesky (no pivoting,
  no equilibration). The blob carries `perm`/`perm_inv`.
- **numba's `np.random` == numpy legacy `RandomState` (MT19937)** for
  `random()`/`randint()`; the convection kernel is therefore portable exactly.
  Occupancy uses `RandomState` too and is exogenous, but is **ported** (not
  baked) so the RNG consumption order matches the original step loop.
- `occupancy_normalization_constant` is **0.0** at runtime (evaluate's
  `_occupancy_normalization_constant = 125.0` is the documented no-op), so the
  `num_occupants` observation is the raw count.
- Observation normalizer matches **exactly** on the unprefixed
  `measurement_name` (else identity); 51 auxiliary features bypass `id_map`.

## Gate status

| gate | scope | status |
|---|---|---|
| **M0 / L0** | bake + blob == live scenario (geometry, solver, convection, HVAC, env contract, reward, schedule, weather, occupancy params) | **PASSED** for all 6 shipped building shapes |
| **M1 / solver** | C replays the baked SuperLU L/U/perms exactly | **PASSED** — max|x−x_ref| = 5.5e-17, 8 ms/solve, nnz = 3.13M |
| **M1b / FDM** | C FDM step (forcing + LU solve + Woodbury air-side + interior mass + ambient exposure) vs a dumped oracle trace | **PASSED** — ~1e-12 K per step (state seeded per step) |
| **M3a / reward** | C reward (`SetpointEnergyCarbonRegretFunction`: productivity regret, elec/gas cost + carbon) vs the env's returned reward | **PASSED** — ~7e-9 absolute (≈1e-7 relative) |
| **M3b / observation** | C observation (per-field observable + normalizer + 51 aux features) vs the env's returned obs, **including the reset obs** | **PASSED** — **bit-exact (0.000e+00)** for all 6 shipped shapes, 30 steps + reset |
| **M1c / L1** | C FDM **+ convection** (MT19937 + room swaps) closed-loop vs the oracle, 30 steps | **PASSED** — max T error 1.27e-11 K (≈4e-13/step), mass 2.3e-13 |
| **M2 / step** | full `env.step` from actions alone (decode → HVAC → FDM → occupancy) vs the oracle, all 6 shapes | **PASSED** — T ~1e-11, air-side 0, decode 0, occupancy 0 |
| **M4 / PufferLib CPU env** | env driven from the oracle action stream vs the oracle's obs/reward; padding + mask; smoke mixed-head PPO | **PASSED** — obs **bit-exact**, reward ≤7.5e-9, all 6 shapes; `build.sh smart_buildings` green; smoke trains |
| **M5a / batched solve** | `sb_lu_solve_multi` (G RHS, L/U streamed once) vs G sequential `sb_lu_solve` | **PASSED** — **bit-identical** (0 / 1 189 800 differ) |
| **M5b / batched FDM** | `sb_fdm_step_batch` vs G sequential `sb_fdm_step`, G = 1, 4, 8, 16 × 20 steps | **PASSED** — **bit-identical** at every G; ≈1.5× per env-step (latency-bound scalar kernels are the floor) |
| **M5c / batched step** | `sb_sim_step_batch` (HVAC + one batched FDM + convection + reward) vs G sequential steps, G = 1, 4, 8 × 10 steps | **PASSED** — temperature, observation and reward all **bit-identical** |
| **M5d / device solve** | cuSPARSE SpSM replay of the baked factors vs `solref_x` | **NOT VERIFIED** — CSR conversion + analysis work; `SpSM_solve` returns wrong values with an async CUDA error |

The C thermodynamic engine (temperature field) is now ported and verified against
the oracle.

### M2 — GREEN

`csim/sb_sim.c` (+ `sb_occup.c`) has the HVAC devices and the full
`execute_step_sim`. The actions-only gate `test_step` runs the C step from the
hybrid action alone (decode -> devices -> step) and compares the temperature
field to the oracle:

| building (n_cvs, k) | steps | max T error | decode | air_g | air_T | boiler | occ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train_1 (205k, 39) | 200 | 3.3e-11 | 0 | 0 | 4e-13 | 0 | 0 |
| train_8 (124k, 30) | 60 | 7.8e-12 | 0 | 0 | 3e-13 | 0 | 0 |
| train_2 (263k, 85) | 60 | 1.5e-11 | 0 | 0 | 7e-13 | 0 | 0 |

So the whole thermodynamic + HVAC + occupancy pipeline matches the Python
oracle from **actions alone**, to ~1e-11 K (≈2e-13/step, ~2e-9 over a full
8640-step episode). The hybrid->native decode matches exactly (numpy computes
the setpoint map in **float32** for a float32 action).

#### Bake contents + the two intricate behaviours

The blob already carries everything the HVAC step needs:
`act_owner_kind`/`act_owner_index`/`act_actuator` (per-action routing),
`vav_zone_offsets`/`vav_zone_idx`, `zone_area`, and per-step
`sched_comfort`/`sched_heat_sp`/`sched_cool_sp` + `occ_hour`/`occ_workday`
(calendar logic stays in Python).

Two intricate behaviours found by reading the source, to port exactly:

1. **Boiler tank ramp is ACTIVE** (`heating_rate=0.5`, `cooling_rate=0.1` in the
   shipped config). `Boiler._set_current_temperature` is *read-count-sensitive*:
   every read of `supply_water_temperature_sensor` advances `_current_temperature`
   by `rate * last_step_duration / 60` (capped at the setpoint) and rewrites
   `_step_tank_temperature_change`. `last_step_duration = observation_ts -
   action_ts`. During `execute_step_sim` the sensor is read once per coupled VAV
   plus once for the return-water calc, and once more in the observation.
2. **Occupancy advances once per step.** `average_zone_occupancy` memoises on
   `(start,end)`; the reward at step k asks for `[ts_{k+1}, ts_{k+2})`, which the
   *next* step's execute/gains reuse as a cache hit. So the chains advance once
   per step (interval `[ts_k, ts_{k+1})`), in `room_dict` order.

## M3b / observation — GREEN (bit-exact)

`csim/sb_obs.c` implements `_get_observation`: for each `field_names` entry the
device observable value, normalized by the exact-match `measurement_name`
constant (else identity), then the 51 auxiliary features. The observable -> C
state mapping (measured from the live env):

| device | measurement_name | attribute (C state) |
|---|---|---|
| Vav | reheat_valve_percentage_command | `vav_reheat` |
| Vav | supply_air_damper_percentage_command | `vav_damper` |
| Vav | supply_air_flowrate_setpoint | `vav_const[v*9]` (max flow) |
| Vav | zone_air_cooling_temperature_setpoint_sensor | `vav_coolsp` |
| Vav | zone_air_heating_temperature_setpoint_sensor | `vav_heatsp` |
| Vav | zone_air_temperature_sensor | `sensing` (thermostat sensing temp) |
| AHU | `ahu_N_*` | per-AHU: cooling_request_count, static pressure, flow, setpoint-gated by run, supply temp, fan run/speed, outside air temp/flow, recirculation, supervisor |
| HWS | differential_pressure_setpoint / _sensor | `pump_dp` |
| HWS | heating_request_count | `hws_heating_requests` (0 at reset, VAV count after a step) |
| HWS | run_status / supervisor_run_command | `hws_run` |
| HWS | supply_water_temperature_setpoint | `boiler_setpoint` |
| HWS | supply_water_temperature_sensor | `boiler_current` (post-ramp) |

The 51 auxiliary features are `hod_cos/sin_000..011`, `dow_cos/sin_000..011`,
`comfort_mode_now/soon`, `num_occupants`. The 48 time features plus the two
comfort flags are baked per step (`aux_time`, indexed by the **observation**
timestamp `ts_{k+1}`, so the obs after step `k` uses index `k+1`);
`num_occupants` = `int(sum_z V_k[z])` (`occupancy_normalization_constant` = 0.0,
so it is the raw count).

Three measured details that a "reasonable-looking" port gets wrong:

1. **Every raw measurement is float32-quantized before normalization.** The
   oracle packs each device reading through an `ObservationResponse` protobuf
   whose `continuous_value` is a protobuf `float` (float32), and
   `StandardScoreObservationNormalizer._normalize_one` then runs on that
   already-rounded Python float. So `sb_obs` must round the *raw* value to
   float32 **before** `(v - mean)/sqrt(var)`. Skipping this gives ~1e-5 K
   residuals (~3e-6 normalized) that look like solver error but are not.
2. **`sensing` must be primed at reset.** `env.reset()` calls
   `_get_observation()` immediately after `building.reset()`, and the shipped
   simulator primes each VAV's `zone_air_temperature` from the *initial* field
   with the same area-weighted zone average `execute_step_sim` uses. Without it
   the reset obs reports the 0 K sentinel. `sb_sim_reset` therefore runs
   `sense_avg_temp` + `sense_zone_temps` (without touching thermostat state).
3. **The reset observation consumes occupancy too.** The reset obs reads
   `num_occupants` for `[ts_-1, ts_0)`, so the occupant chains are advanced once
   before step 0 (`sb_occup_prime`, using baked `occ_hour_m1`/`occ_workday_m1`).
   On a holiday that peek consumes no RNG — which is exactly why the earlier
   actions-only gate passed without it.

Gate: `test_step` now also dumps `n_field` + the reset obs + the per-step obs and
compares the full 352/703/… float vector each step. Result, all 6 shipped
shapes, 30 steps + reset: **`obs=0.000e+00` (bit-exact)**, alongside T ~1e-11
and reward ~5e-9.

## M4 — CPU PufferLib env + mixed-head PPO — GREEN

Lives in the PufferLib 5.0 checkout (`/data/puffer-work/plclean`), synced from
`csim/`:

| path | what |
|---|---|
| `ocean/smart_buildings/smart_buildings.h` | the env (`puf_init/reset/step/close/log/render`) |
| `ocean/smart_buildings/csim/` | copy of `complete_port/csim` (sources + headers) |
| `ocean/smart_buildings/test_env_cpu.c` | CPU regression: env vs oracle trace + smoke |
| `config/smart_buildings.ini` | env/train config (`blob`, `blob_dir`, `max_steps` must exist here for `--env.KEY=` overrides to apply — `ini.h` only updates existing keys) |
| `../pufferl_mixed_heads.diff` | `git diff` of `src/pufferl.cu`, `src/algo.cu`, `build.sh` |

### Layout

`OBS_SIZE = 1024` (`SB_OBS`), `NUM_ATNS = 272` (`SB_CONT 256` + `SB_DISC 16`),
`ACT_SIZES` = 256 heads of size 1 (Gaussian) + 16 heads of size 2 (Bernoulli), so
`act_n` (mask width) = 288. A scenario's real observation (352…703) and action
(92+5…209+?) occupy the front of those blocks; the rest is zero-padded and its
`action_mask` entry is 0. `decoder_output_size = act_n` (one column per head
class + the value column the decoder adds).

`puf_reset` = `sb_sim_reset` + `sb_occup_prime` + the reset observation +
`sb_occup_advance(0)`; `puf_step` = clamp continuous heads to `[-1,1]` →
`sb_decode_native` → `sb_sim_apply_native` → `sb_sim_step` → obs → occupancy
advance → `sb_reward`. The terminal is set at `steps_per_episode` and the env
auto-resets in the same call (the cartpole pattern — `puf_reset` deliberately
does **not** clear `rewards`/`terminals`, which the trainer reads).

### Mixed-head patch (what was previously unsupported)

`pufferl.cu:1866` asserted `!(n_cont > 0 && n_disc > 0)`. The patch replaces the
"all-continuous XOR all-discrete" switch with a per-head branch, keyed on
`act_sizes[h] == 1`, in the three places that need it:

1. `sample_logits` — Gaussian head (means at the head's decoder column, `logstd`
   indexed by head) or a Categorical head; a head whose mask entry is 0 emits
   action 0 and contributes no log-prob.
2. `cache_imp_and_v` (the rollout log-prob walk).
3. `ppo_loss_compute` (the per-head gradient): continuous heads write
   `grad_logits` and `grad_logstd` (`logstd` row stride is `A_total`, matching
   `decoder_backward`'s `sum_rows_to_precision_kernel`), discrete heads leave
   `grad_logstd` at 0 — the row is zeroed first, because the decoder emits a
   column per head *class*, so slots past `NUM_ATNS` exist but are unused.

`decoder_output_size = act_n` and `dw->continuous = (n_cont > 0)` so the logstd
parameter (1 × act_n) is registered, initialized (zero ⇒ std 1) and trained.
Padding is masked in the observation (zeroed) and in the loss (mask entry 0).

### Gate results

- **Env trace parity** (env driven from the oracle action stream vs the oracle's
  observation/reward), all 6 shipped shapes, 30 steps + reset:
  `obs err = 0.000e+00` (bit-exact), `reward err ≤ 7.5e-9`, padding verified
  zero every step, mask shape verified.
- **CPU regression** `test_env_cpu <blob>` (no trace): random actions,
  2.5 short episodes; terminal fires at `max_steps`, resets reload correctly,
  rewards/obs finite, mask and padding checks — OK.
- **Build**: `./build.sh smart_buildings` → `./puffer` green.
- **Smoke PPO trains** (mixed heads end to end), 8 agents, horizon 64,
  128-step episodes: `perf/score ≈ -0.001`, policy 0.14, value 1.2, `kl` 0.4,
  `clipfrac` 0.79, entropy 138 (≈ the 92 active Gaussian heads × 1.42 + 16
  Bernoulli × ln2, i.e. the masked heads contribute nothing), SPS ≈ 110 with
  the C env at 99% of the time. Episode ends/resets exercised.

### Two environment facts worth knowing

1. **The native build needs the CUDA-13 container.** This host's `/usr/bin/nvcc`
   is 12.0 and rejects `src/algo.cu`'s C++17 designated initializers over types
   with default member initializers ("class type not suitable for use with
   designators"). Build inside `pufferai/puffertank:5.0` (nvcc 13.0), e.g.
   `docker run --rm --gpus all -v /data/puffer-work:/work -w /work/plclean
   pufferai/puffertank:5.0 bash -lc "./build.sh smart_buildings"`.
   Two portability fixes went into `build.sh`: only prefix `ccache` when it is
   actually installed, and honor an `NCCL_IFLAG`/`NCCL_LFLAG` from the
   environment (no `nccl.h` in the default search dirs here).
2. **Continuous heads must be clamped to `[-1,1]` in the env.** The oracle's
   `BoundedActionNormalizer.setpoint_value` *raises* outside
   `[min_norm, max_norm]`; PufferLib samples an unbounded Gaussian, so without
   the clamp the setpoints are absurd and the smoke run showed `perf ≈ -7553`
   instead of ≈ -0.001. The clamp lives in `puf_step` (the crashit/env
   convention); the log-prob is still computed from the unclamped sample, which
   is a known PufferLib action-clamp caveat, not a parity issue (every in-range
   action decodes bit-exactly).

`pufferl_mixed_heads.diff` is `git diff` against the checkout's HEAD, so it also
contains the earlier crashit-work changes that were already uncommitted there.

### M5 — batched engine: DONE and bit-exact

The FDM matrix is **time-invariant** (the bake stores one factorization; only the
forcing changes per step), so a step is 2 triangular solves with a *fixed*
matrix and one RHS per env. That makes group batching possible **without
changing a single floating-point operation per column**: solve G envs that share
a scenario with one multi-RHS pass that streams L/U once.

`csim/sb_lu.c: sb_lu_solve_multi` — the batched solve. The work array is
`(n, G)` row-major so a push into row `i` touches G contiguous doubles (one cache
line instead of G scattered ones), and each column is updated in exactly the
order `sb_lu_solve` uses.

`csim/sb_fdm.c: sb_fdm_step_batch` — exposure + forcing per env → **one** batched
solve → the per-env air-side (sensed contraction, k×k dense solve, applied
scatter) → **one** batched solve for the corrections → mass update + field
scatter. Inactive air-side envs contribute an all-zero RHS column (adding zero is
exact), so they stay identical to the sequential path.

Gates (all `OK`, and the numbers are `0/N differ`, i.e. **bit-identical**, which
is the "replicate as it is" requirement):

| gate | check | result |
|---|---|---|
| `test_multi` | `sb_lu_solve_multi` vs G sequential `sb_lu_solve` | 0 / 1 189 800 differ (max \|d\| 0.000e+00); CPU replay still 5.55e-17 vs baked `solref_x` |
| `test_batch` G=1 | `sb_fdm_step_batch` vs sequential, 20 steps | temp 0 / 205 434, mass 0 / 148 725 differ |
| `test_batch` G=4 | same | temp 0 / 821 736, mass 0 / 594 900 differ |
| `test_batch` G=8 | same | temp 0 / 1 643 472, mass 0 / 1 189 800 differ |
| `test_batch` G=16 | same | temp 0 / 3 286 944, mass 0 / 2 379 600 differ |

Timing, per env-step (FDM only, `building_train_1`, 20 steps):
| G | sequential | batched | speedup |
|--:|--:|--:|--:|
| 1 | 14.29 ms | 17.41 ms | 0.82× (copy overhead, no batching win) |
| 4 | 12.56 ms | 9.21 ms | 1.36× |
| 8 | 12.91 ms | 8.00 ms | 1.61× |
| 16 | 12.07 ms | 7.83 ms | 1.54× |

**Read this honestly**: the gain saturates at ≈1.5×, not the ≈6× the memory-traffic
arithmetic predicts, because with the solve batched the *remaining* FDM work
(`sb_fdm_exposure`'s two divisions per cell, forcing, mass update, 205 k-element
scatter) is latency-bound scalar code and becomes the floor (~7.8 ms/env-step).
The next win is parallelising those elementwise loops (OpenMP across the group)
and/or vectorising them — the plan's step E — not more batching.

Note `test_batch` at G=1 is *slower* than sequential: the packing copies cost
more than a single solve saves. Batching pays from G≥4.

**Whole-step batching** (`sb_sim_step_batch`). `sb_sim_step` was split at its one
FDM call into `sb_sim_step_pre` (sensing + HVAC + air-side + applied power) /
FDM / `sb_sim_step_post` (sensed means + convection + boiler ramp), with the two
per-step locals that cross the boundary (`total_reheat_w`, `coil[256]`) promoted
to `SbSim` fields. `sb_sim_step` is now literally pre + `sb_fdm_step` + post, so
the M2 step gate still passes unchanged (obs 0.000e+00, T 1.3e-11).

`sb_sim_step_batch` then runs a group as: pre for every env → **one** batched
FDM → post for every env. Verified against G sequential `sb_sim_step` calls over
10 steps:

| G | temperature | observation | reward |
|--:|---|---|---|
| 1 | 0 / 205 434 differ | 0 / 352 differ | max \|d\| 0.000e+00 |
| 4 | 0 / 821 736 differ | 0 / 1 408 differ | max \|d\| 0.000e+00 |
| 8 | 0 / 1 643 472 differ | 0 / 2 816 differ | max \|d\| 0.000e+00 |

That is the "replicates as it is" check for the engine a vector env would drive:
same field, same observation, same reward, bit for bit.

**A test-harness lesson worth recording.** The first run of that gate showed a
2 K divergence and looked like a kernel bug. It was the *harness*: the occupant
chains (`sb_occup`) and the convection RNG (`sb_conv`) are **stateful across
episodes by design** — the oracle's are too, because a building is not
re-created per episode — so a second comparison pass started from the first
pass's RNG state and legitimately diverged. The gate re-seeds them with
`sb_occup_free/init` + `sb_conv_free/init` between passes. (`sb_sim_reset`
deliberately does *not* reset them; the env calls `sb_occup_prime` instead.)

The batched engine is **not yet wired into the PufferLib vector env**; that is
the remaining half of M5 (see below).

### M5 — the vector-env wiring: design, and the two constraints that dictate it

Reconnaissance done (`ocean/admiral/admiral.cu` is the GPU-env template, and
`src/pufferl.cu` is the framework):

- GPU envs implement `Env* puf_vec_create(int total_agents, Dict*, obs_t* obs,
  float* actions, float* rewards, float* terminals)` plus `puf_bind_stream`,
  `puf_init/reset/step/close`, and return a **device-resident** `Env*` batch
  base. `puf_step(Env*)` runs the whole vector.
- **Constraint 1 — CUDA graph capture.** `rollout_start` wraps the entire
  horizon in `cudaStreamBeginCapture(... puf_step ... cudaStreamEndCapture)` and
  then *replays the graph*. A host-side `puf_step` therefore executes only
  during capture and never again; the observations would freeze. A host-batched
  engine must run with **`--base.cudagraphs=-1`** (graphs off), which the code
  already supports (`first = hypers.cudagraphs && ...` ⇒ the per-step path runs
  every step).
- **Constraint 2 — the action mask is not handed to the env.**
  `puf_vec_create` receives obs/actions/rewards/terminals but *not*
  `p->env.action_mask.data`, and the framework only `cudaMemset`s it to 1. The
  padding mask (256−n_cont heads / 16−n_disc pairs must be 0) is what makes the
  mixed-head loss correct, so the env needs it. The codebase already has the
  idiom for an optional env hook (`PUF_HAS_BOT_POLICY`): add
  `PUF_HAS_MASK_INIT` + `puf_mask_init(unsigned char* mask, int width, int
  agents)` with a no-op default, call it from `env_setup` next to the
  `cudaMemset`, and let the env fill the constant per-agent pattern (n_cont ones,
  then n_disc pairs of ones, rest zero).

Then `puf_step` would: D2H the sampled actions on the bound stream → host
batched engine (group envs by scenario, one `sb_sim_step_batch` per group, with
the per-env elementwise work parallelised across the group — this is where the
remaining speed is, see below) → H2D the observations/rewards/terminals and the
per-env `Log` array (the framework's `log_reduce` kernel reads `vec->envs[i].log`
on the **device**, so the logs must be copied to a device `Env` array sized
`total_agents`).

Cost/benefit, measured: the batched engine saves ≈4–5 ms of ≈12.5 ms per
env-step on the FDM, and batching is what stops N envs from each streaming the
same 50 MB of L/U (the machine is bandwidth-saturated with N independent
solves). The remaining ≈7.6 ms floor is the latency-bound elementwise work
(`sb_fdm_exposure`'s two divisions per cell, forcing, mass update, the 205 k
scatter), which must be parallelised across the group — the plan's step E — and
that parallelisation is only safe (bit-exact) because it is *across envs*: each
env's own operation order is untouched.

### M5 — device path (cuSPARSE): started, not yet verified

The device solve is an alternative to the CPU batched path (not required for
correctness — the batched CPU path above is already exact).
`gpu/sb_gpu_solve.cu` gates it against the baked reference (`solref_b →
solref_x`, the same ground truth M1 used) and against the CPU replay.

Verified working:

- The CPU replay reproduces the baked `solref_x` to **5.55e-17**.
- `csc_to_csr` (host, CSC→CSR, rows emitted in increasing column order) converts
  L (3 132 255 nnz) and U (3 132 255 nnz) correctly.
- cuSPARSE SpSM accepts the CSR L (fill LOWER, unit diag) and U (fill UPPER,
  non-unit): `cusparseSpSM_bufferSize`/`_analysis` both succeed (46–54 MB
  workspaces). Each descriptor needs its **own** analysis buffer — sharing one
  clobbers the other factorization.
- The batched `gather`/`scatter` around the solve must apply the permutation
  **per column** (`y[c*n+j] = x[c*n + idx[j]]`); a flat single-permutation
  version is silently wrong for `nrhs > 1`.

Not working: `cusparseSpSM_solve` returns wrong values (max err 1.08e-01 against
a solution of magnitude 7.4e-02, even at `nrhs = 1`) and raises an asynchronous
CUDA error. Concrete next steps, in order:

1. Query the attributes back (`cusparseSpMatGetAttribute`) to confirm fill mode /
   diag type landed, and try both unit-diagonal variants — our CSC rows put the
   diagonal *last* while many triangular kernels assume first.
2. Try `cusparseSpSV` (single RHS) on the same descriptors to separate
   SpSM-batching misuse from a wrong triangular setup.
3. If cuSPARSE stays unusable, fall back to a partitioned SpTRSV (block the
   unknowns, solve block-by-block with an SpMV update between blocks) — the 2-D
   grid's level count (≈ nr + nc ≈ 945) makes a level-scheduled kernel
   launch-bound, so partitioned is the right shape.


## M5 — device solve: GREEN (cuDSS)

The GPU solve is solved by using a real sparse **direct** solver instead of
cuSPARSE's `SpSM` (which is level-scheduled/partitioned and was never intended
for a 148 725² factor with 991 dependency levels). `nvidia-cudss-cu13`
(0.8.0.10) is installed into `/data/puffer-work/cudss` (the wheel ships no
`libcudss.so` symlink — create one, or link the versioned file).

`gpu/sb_gpu_cudss.cu` gates it on the baked `building_train_1` system:

| measurement | value |
|---|---|
| cuDSS vs the baked reference solution `solref_x` (same system) | **6.2e-16 (rel 8.4e-15)** |
| cuDSS vs the CPU SuperLU replay | 6.4e-16 |
| residual `‖A·x − b‖∞ / (‖A‖∞·‖x‖∞)` | 3.4e-16 |
| analysis + factorization (once, matrix is time-invariant) | **~637 ms** |
| **solve, 1 RHS** | **1.12 ms** |
| CPU SuperLU replay, same solve | 16.3 ms |
| **speedup** | **14.5×** |

So the device solve reproduces the oracle's own reference solution to 6.2e-16 —
i.e. it is the same solve, differing only in summation order — while being
14.5x faster per right-hand side.

Findings that matter for the next step:

1. **RESOLVED: the bake now ships the assembled system.** `A_conduction_*` is
   only the conduction part; the matrix SuperLU actually factorizes is
   `solver.matrix(cc)` — conduction + capacitance/dt diagonal + Dirichlet + mass
   coupling (one reference convection coefficient, so it is time-invariant).
   `bake_scenario.py` now also writes `A_system` (CSC), and the gate converts it
   to CSR (cuDSS is CSR-only) and agrees with `solref_x` to 6.2e-16. Before that
   fix cuDSS was being handed the wrong matrix and differed by 4.5e-05 — a
   useful reminder that the residual (not the cross-solution diff) is the check
   for "did the solver solve what it was given".
2. **Batch with cuDSS's ubatch config, not a wide dense RHS.**
   `CUDSS_CONFIG_UBATCH_SIZE` / `CUDSS_CONFIG_UBATCH_INDEX` are the supported
   way to reuse one factorization across many right-hand sides. A dense
   `n × nrhs` RHS hits a stream/event quirk at `nrhs > 1` (the completion is not
   observable through CUDA events even after `cudaStreamSynchronize`), so do not
   build the batch on that path.
3. `cudssSetStream` needs an explicit stream (create one; do not pass the legacy
   default stream).
4. Measured cost for the env design: 2 solves per env step ⇒ ≈2.2 ms of GPU work
   per env step with one RHS per solve, before ubatching — against ≈4–8 ms of
   CPU solve time per env step in the current CPU path.

### M5 — device FDM step: GREEN

`gpu/sb_gpu_fdm.cu` runs the whole FDM step as CUDA kernels with cuDSS doing the
two solves, and is gated on the M1b trace (`dump_fdm.py` -> `test_fdm`'s exact
per-step inputs):

| | |
|---|---|
| temperature error vs the CPU/oracle trace | **≤ 4.5e-11 K** |
| interior-mass error | 6.8e-13 |
| device time per step (1 env, incl. the gate's transfers) | **3.64 ms** |

It mirrors `sb_fdm_step` exactly: exposure → forcing (same accumulation order) →
`y = A⁻¹f` → sensed contraction → `M = I + diag(g)·S` small dense solve (block
parallel, partial pivoting) → `r = applied·q` → `y += A⁻¹r` → mass → scatter.
`sensed_response` is *not* baked (sb_fdm_init computes `S = sensed·A⁻¹·applied`);
the device builds it at init with one batched solve of the `applied` columns.

Two cuDSS usage facts that cost real time and must not be re-learned:

- **The analysis/factorization must run with a single RHS.** A wide RHS there
  trips an internal out-of-bounds. The wide (batched) RHS belongs to the SOLVE
  phase only.
- **Time it with the host clock around a stream sync.** cuDSS schedules work on
  internal streams, so CUDA events recorded on ours report `device not ready`.

Measured, one env: 2 solves ≈ 2.2 ms of the 3.64 ms. Batched over a group the
per-RHS solve cost falls to ~0.25 ms (64× the CPU), so the device step scales
with group size — which is the whole reason for batching across envs.

### M5 — device occupancy + convection: GREEN (bit-exact)

`gpu/sb_gpu_rng.cu` ports the MT19937 (`sb_rng.c`), the occupancy chains
(`sb_occup.c`) and the convection room swaps (`sb_convect.c`) to the device and
gates them against the CPU in lockstep:

| gate | result |
|---|---|
| occupancy `V` per step, 3 envs x 6 steps | **0.000e+00** (bit-exact) |
| temperature after the convection swaps | **0.000e+00** (bit-exact) |

Both consume a **single MT19937 stream per environment in a fixed order**, so
they are inherently sequential *within* an env and are parallelised *across*
envs — one block per env, one thread running the chain. That is the same
batching axis the solves use, and it is the reason the GPU design batches
environments rather than parallelising a single one.

### M5 — device HVAC pre-phase: GREEN; full-step integration: BLOCKED (cuDSS segfault)

`gpu/sb_gpu_sim.cu` ports the whole `sb_sim_step` to the device with the structure
the measurements dictate — per-env sequential blocks for the HVAC/RNG chains,
globally-parallel kernels for the field-sized loops, cuDSS for the two solves.
Gate against the M2 actions-only trace:

| phase | result |
|---|---|
| apply + `avg_temp` + pre + `input_q` (stage A harness) | `air_g` **0.000e+00 at every step**; `air_T` and the boiler **0.000e+00 at step 0** (the step≥1 drift is only the stage-A harness not running the post, so `step_index` never advances) |

Two bugs of mine found on the way, both worth remembering:

- **Device code must use the C core's EXACT constants.** I wrote `CP_AIR 1005.0`
  / `CP_W 4186.0`; the C uses **1006.0 / 4180.0**, which showed up as `air_g`
  uniformly 0.100% low (`1005/1006 = 0.999006`). `air_T` was exact, which is what
  identified it.
- **The shared (blob) tables must be uploaded and the device structs pointed at
  the device copies.** Pointing them at the mmap (host addresses) produced
  `misaligned address` faults from the kernels.
- Also: the post's `sensed_mean` uses `air_side_sensed_*` (CSC, n_cvs+1), NOT the
  FDM's `applied_*` (k+1) — `SbSim` binds `ap_p/ap_i/ap_x` to the air-side sensed
  matrix, while the FDM binds its own `ap_*` to `applied`. Its per-column sum must
  run in ascending-cell order, which is why the device uses a precomputed CSR of
  the transposed sensed matrix (deterministic and order-identical) rather than
  atomics.

**Blocker.** In the integrated program, `cudssExecute(CUDSS_PHASE_SOLVE)` — the
air-side build's batched solve of the 39 `applied` columns — dies with SIGSEGV
inside the library. The identical call succeeds in `gpu/sb_gpu_fdm.cu`. Ruled
out so far: the batch width (`nenv=1` also crashes), the descriptor creation
order (moving the group `(n × nenv)` descriptors after the build does not help),
and the A-matrix CSR conversion (shared, count-verified). Next step is a minimal
reproducer: `sb_gpu_fdm`'s main with only the sim's setup imported, bisected
until the differing state appears.

---

## M5 — the whole step on the device: GREEN (with measured cost)

`gpu/sb_gpu_sim.cu <blob.sbb> <trace.step> <nenv>` runs one full env step as
device work — occupancy advance, HVAC/pre phase, forcing, cuDSS FDM solve,
air side, cuDSS air-side solve, mass/ambient scatter, sensed means, post phase
and convection — and diffs it against the trace the verified CPU core produced
from the same actions:

    worst: occ=0.000e+00 air_g=0.000e+00 air_T=5.9e-12 field=1.8e-11 boiler=0.000e+00
    GPU SIM STEP: OK

(occupancy, air_g and boiler are bit-exact; air_T and the field are at solver
round-off.)  Bugs this gate found, all fixed:

- `sh_boil` is a *device* pointer; `s->boiler_setpoint = sh_boil[0]` on the host
  segfaulted (the host copy `boiler_const` is now bound separately).
- the occupancy advance was never wired into the loop — `d_V` was uninitialised,
  so the zone gains carried garbage (occ is now bit-exact).
- `header_resistance`/`pump_head` were guessed; the C binds
  `hws_header_resistance` and `pump_const[0]` (a zero pump head makes
  `total_flow_rate` zero and silently drops all reheat).
- the hand-written k x k solve in `airside_k` was wrong: the `j == c` threads
  zeroed `M[r][c]` before using it to update `q[r]` (and raced with the other
  columns), so the air side never eliminated its RHS.  The FDM-only gate had
  passed because its action stream left `air_g == 0`, which skips that block.
  The kernel now mirrors the CPU `dense_solve` (no pivot-row normalisation,
  `f = M[r][c]/d`, staged multipliers, ascending back-substitution).

### Measured cost, and what it is bound by

On this box (RTX 3090, sm_86 — build with
`-gencode arch=compute_86,code=sm_86`; the default sm_75 target merely JITs and
is *not* a performance factor here) the step costs **57.0 ms** at 1-2 envs:

    phase ms/step: pre=3.85 solve=0.06 post=5.89 conv=51.10 (total 56.99)

The convection is the whole story, and it is latency-bound, not
instruction-bound.  `sb_conv_apply` is a single-threaded per-room chain of
~1.4e5 MT19937 draws plus ~1.4e4 swap pairs; a single GPU thread exposes every
dependent access (measured with in-kernel `clock64`: ~315 clocks per draw, and
~350-500 per dependent global access in the candidate/swap phases).  A
register-resident cursor and a modulo-free (and now register-cursor) twist did
not move it, and native sm_86 code did not either — the chain itself is the
cost.

The fix the measurements call for (not yet written):

1. generate the MT stream in parallel into a scratch buffer — the run of
   tempered words between twists is a parallel map, and the in-place twist is
   exactly three parallel passes (element `i >= MT_N-MT_M` reads
   `i-(MT_N-MT_M)`, already updated) — then consume it in order through a
   shift-register prefetch (the stream *order* is fixed even though the count
   per cell is data-dependent), advancing the saved state by the consumed
   count with a small parallel kernet;
2. stage each room's `{cell index, candidate offsets, temps}` into shared with
   the whole block, run the serial chain entirely in shared, resolve the room's
   candidate-id picks in parallel (the only data-dependent global reads), and
   scatter back in parallel.  The shared budget fits the largest room here
   (8646 cells) at ~208 KB; the existing kernel stays as the fallback for
   larger rooms.

Projected ~2-3 ms/step, which is what makes the batch-throughput target (the
whole point of the port) reachable.

### The PufferLib env landed (GPU path)

`ocean/smart_buildings/` now holds the GPU env: `smart_buildings.h` (the
PufferLib entry, `#include`s the implementation), `sb_gpu_env.inc` (the host
core + the env glue) and `sb_gpu_kernels.cuh` (the device code).  The same pair
is the parity gate under `complete_port/gpu`, built with `-DSB_TEST`, so the env
and the gate cannot drift.  The CPU reference env it replaces is preserved at
`complete_port/pufferlib_cpu_env.h`.

Verified:

- the env's own core (`sb_gpu_setup` + `sb_gpu_reset` + the batched step, driven
  with a recorded trace's actions) is bit-exact against the verified CPU core at
  1/4/16/32 envs - `occ`/`air_g`/`boiler` 0.000e+00, `obs` 0.000e+00, reward
  7.468e-09, field ~1.7e-11;
- the framework-facing TU compiles clean against the real PufferLib headers;
- `build.sh smart_buildings` builds `./puffer` (the cuDSS include/lib are wired
  into that branch, and the env is `SRC_DIR/smart_buildings.h`, the repo's
  convention).

The env mirrors the CPU env's observable behaviour: continuous heads clamped to
[-1,1] then decoded with `sb_decode_native`, discrete run bits, the same
reward/observation ordering (obs with the interval the step used, reward with the
next one), the mixed-head action mask, the `Log` columns, and a batch restart at
the episode end.  One scenario per batch (the gateway runs one scenario per
episode, which is the submission shape); the state is dimensioned so more
scenario groups can be added without touching the kernels.

### Convection: still the cost

Measured on this box after all of the above: ~55 ms/step at 1 env and ~69 ms at
32, of which the convection is ~40 ms (the pre/post phases 3-13 ms).  That is
~1.7 ms per env-step, i.e. still about level with the CPU env's ~1.1 ms.  The
staging is correct but did not deliver the predicted gain: it costs 69 KB of
shared per block, which pins the kernel at one block/SM and slows the parallel
gather/scatter, while the serial collection is still ~141k cells of one thread
reading its word stream from global.  The next step that moves it is to shrink
that footprint (stage the temperatures in halves, or keep a small word window in
shared) so occupancy recovers.

### The score sweep: the gap is in the reference file, not the port

`sweep_scores.py --pycheck building_train_8 singapore_7 --pycheck-policy pid_water
--pycheck-steps 8640` (a full episode, both sides driven by the same Python
policy):

    bridge total = -468.1674329434771   (105.1 s)
    oracle total = -468.16744426385776  (206.2 s)
    |delta|      = 1.132e-05            (2.4e-8 relative)

The C bridge reproduces the live Python environment over all 8640 steps to
1.1e-05, so the port is right over a full episode and there is no chaotic
divergence.  The sweep's failure for this pair was `mine=-468.167433
ref=-467.293350` - i.e. **my value matches the oracle (-468.16743) and
`reference_baselines.json` is the one that disagrees with the organizers' own
simulator** (by 0.874 absolute, 1.9e-3 relative for this pair).  That is why the
aggregate over 30 scenarios tracks the reference to ~1% while individual pairs
scatter: the shipped per-scenario baseline values were produced under different
conditions, and the 1e-3 per-pair tolerance is tighter than the reference's own
internal consistency.

### Speed, measured properly (corrects an earlier claim in this file)

The `--pycheck` run gives a clean per-step comparison, both sides including the
same Python policy cost:

    bridge (C core + policy)  12.2 ms/step      oracle (Python + policy)  23.9 ms/step

so the C port is **~2.3x** the organizers' numba-JIT simulator per env-step on
one core - not the 30x an earlier note here claimed (that figure was an
agent-vs-env-step confusion).

### Throughput in SPS terms, and the bar it sets

The competition env has exactly one agent per env, so PufferLib's SPS is
env-steps/s here.  Measured on this box:

| implementation | env-steps/s | per step |
|---|---|---|
| organizers' Python/numba oracle (1 core) | ~42 | 23.9 ms |
| this repo's C core `csim` (1 core) | ~82 | 12.2 ms |
| the GPU env, one 32-env batch (1 GPU) | ~700 | 44 ms per batch step |

In the trainer the GPU batch measures 682 SPS at 32 agents (`puf_step` is the
whole env, so the batch step is 44 ms and the rest is the trainer's own eval
time).  A full 30-scenario evaluation is therefore ~1.7 h single-core at the
organizers' speed, ~53 min with the C core, and ~6 min as one 32-env GPU batch.

**The honest reading: ~17 Python cores in parallel match the current GPU batch.**
The port is not yet a throughput win over a many-core CPU run of the organizers'
own code, and the reason is measured rather than assumed - the per-kernel profile
puts 47% of the step in the convection's serial word-stream walk (21 ms) and 31%
in the cuDSS triangular solves (14 ms).  With those two off the critical path the
batch step reaches a few ms and the comparison stops being close; that, not more
per-kernel micro-tuning, is the remaining work.

### Integration with the trainer: two blockers found, not yet cleared

`./build.sh smart_buildings` fails inside `src/pufferl.cu` (which pulls in every
env source):

    src/protein.cu(156): declaration is incompatible with previous "sh"
            (declared at line 859 of ocean/smart_buildings/sb_gpu_kernels.cuh)
    src/algo.cu(1287): expression must have a constant value

1. **Name collisions.**  PufferLib builds one translation unit containing all
   envs, so this env's file-scope globals collide with other envs' symbols
   (`sh` versus protein.cu; the generic `n`, `k`, `nr`, `nc`, `A`, `th` names are
   the same hazard).  Fix: wrap everything from the state block onward in a
   namespace, keep only the `puf_*` entry points at global scope as thin
   forwarders, and `using namespace` it for the `-DSB_TEST` gate.
2. **`Env` fields.**  The trainer writes `envs[i].rng = i` (and needs `seed`);
   added, but the second error above is the same class and needs the build
   re-run after (1) to see what remains.

Until then the binary that runs is the **stale CPU-env build**: that is why the
21-minute "training" run held ~5 CPU cores hot with the GPU at 3% - it was the
CPU env stepping the simulator on the host (the GPU memory it held was the
trainer's PPO buffers, not the sim).  This is a build/name-isolation problem in
the wrapper, not a defect in the device step, which is verified bit-exact above.

## Session results: convection rewrite (item 2) and the device-only step (item 1)

### Item 2 — convection: a measured 2x on the dominant phase

Phase clocks for env 0, per step, ~1.76 GHz, building_train_1/fairbanks_1:

| phase | before | after |
|---|---|---|
| gather (parallel) | 0.67M | 0.65M |
| **collect (serial, one thread)** | **78.2M** | **37.3M** |
| resolve (parallel) | 0.12M | 0.12M |
| swap (serial) | 1.12M | 1.08M |
| scatter (parallel) | 1.33M | 1.29M |

Gate ms/step: n=1 50.7 -> 31.9, n=32 66.6 -> 47.3. The trace gate stayed
bit-exact throughout (occ/air_g/obs exactly 0, air_T 5.7e-12, field 1.8e-11,
rew 7.468e-09).

**Diagnosis, measured rather than guessed.** The collection is a single thread
walking a strictly sequential word stream (~3e5 words per env-step). Its *memory*
was not the cost: three structural attempts all measured neutral — staging the
word stream in shared memory (LDS instead of LDG), staging the candidate offsets
in shared, and making the stream pops stateless (they had taken the cursor by
reference, which put the whole cell loop in local memory: the compiler's own
report went from a 40-byte stack frame to 0, and the time did not move). The cost
was per-cell *arithmetic*: `sb_rng_double` is `(a<<26 | b)/2^53` and is only ever
compared with p, so two int->double conversions and a full IEEE double divide sat
on the critical path of every one of 141707 cells. It is now an exact 64-bit
integer comparison — `D > (uint64)(p * 2^53)` is equivalent because p*2^53 is an
exactly representable double, so the truncation is exact — and the four-way
register lookahead over candidate offsets is gone now that they are in shared.

### Item 1 — device-only step: landed, gate-verified; one blocker left

The step no longer touches the host. What landed:

- the framework's stream is used for every launch, and the cuDSS handle is bound
  to it (`puf_bind_stream` no longer ignores its argument — previously everything
  ran on a private stream, so *nothing* the env did could be captured);
- `decode_k` applies sb_decode_native's rules to the framework's packed action
  buffer on the device (with the per-env step counter the schedule lookups need);
- `finalize_k` writes rewards/terminals and accumulates the episode log on the
  device (`log_reduce` reads `envs[i].log` in a kernel, so this is its home);
- the observations are written straight into the framework's buffer;
- `reset_state_k` + guarded re-runs of `occup_k`/`obs_*_k` restart a finished
  episode on the device (no host `sb_gpu_reset` mid-rollout);
- per-step `cudaStreamSynchronize` and the lazy `cudaFuncSetAttribute` are gone
  (the attribute is set in setup, and `touch_envs_k` touches every managed page
  once so the Env array cannot fault inside a capture).

The standalone trace gate remained bit-exact across all of this. `./build.sh
smart_buildings` succeeds and `./puffer train` now reaches the step; it then fails
with `cuDSS err 2` on the capture stream. That is a solver-integration issue, not
env host work: the analysis+factorization run in `sb_gpu_setup`, and
`puf_bind_stream` re-points the handle afterwards, so the solve is executed on a
stream the factorization never ran on. Next action: make the factorization happen
on the bound stream (bind before setup, or re-analyze/re-factorize when the stream
changes), or move the solve onto cuDSS's capture-safe path.

### Item 3 — pre/post per-entity parallelism: not started

Plan unchanged: split the per-env serial VAV/AHU loops into
`(env, vav)` and `(env, ahu)` kernels, keeping only the boiler ramp and the
step-index scalar chain per env. The reductions are deliberately kept
order-preserving (one thread per AHU walking its VAVs in index order) so the
split stays bit-exact rather than merely close.

## Session results, part 2: the measured bottlenecks, and a working trainer

### The planned item 3 was refuted by measurement

Per-kernel GPU time at 32 envs (nsys, 30 steps, ~49.7 ms/step before):

| kernel | ms/step | share |
|---|---|---|
| convect_k | 21.0 | 42% |
| cuDSS fwd+bwd | 13.8 | 27% |
| **input_q_k** | **4.4** | **9%** |
| avg_temp_k | 2.0 | 4% |
| conv_gen_k | 1.8 | 3.5% |
| airside_k | 1.5 | 3% |
| sensed_mean_k | 1.1 | 2% |
| **seq_k** (2 calls) | **0.43** | **0.8%** |
| post_seq_k | <0.3 | <0.6% |

`seq_k`/`post_seq_k` -- the per-entity parallelism item 3 was going to
restructure -- are **0.8%** of the step. Doing that work would have been wasted.
The real per-kernel costs are the serial collection inside the convection (42%,
which the item-2 rewrite already halved) and the cuDSS triangular solves (27%).

### What was done instead, from the profile (all bit-exact)

- **`input_q_k` is now cell-parallel** (one block per (env, zone), threads
  striding that zone's room cells). It was one thread per (env, zone) walking
  them serially with a dependent index load per cell: 4.4 ms/step. Its write is a
  pure function of the zone, so distributing the same writes over the block is
  bit-exact. It no longer appears in the profile (was 9% of the step).
- **The room-cell -> control-volume offset table** (`d_coff`) is precomputed once
  in setup and passed to `avg_temp_k`/`input_q_k`, which otherwise chase
  `room_cells[2t]` and `room_cells[2t+1]` -- two dependent loads -- before every
  temperature access. `avg_temp_k` 2.0 -> 0.8 ms per call. Note there are *two*
  room tables: the kernels walk `room_cells`/`room_offsets`, the convection walks
  `conv_room_cells`/`conv_room_offsets` and keeps its own indexing.

Result: **47.3 -> 44.0 ms/step at 32 envs**, trace gate still bit-exact.

### The trainer runs

`./build.sh smart_buildings` succeeds and a training run completes (57.3K steps,
7 epochs). Two framework facts had to be discovered rather than assumed:

- `cudagraphs` is read from the **`[base]`** section and is enabled whenever the
  value is **>= 0** (pufferl.cu:1818), so `0` still enables it -- it must be
  negative to turn the capture off. The key must exist in the config for the CLI
  to accept the override.
- With capture on, the step fails at `cudssExecute` with
  `CUDSS_STATUS_ALLOC_FAILED`: cuDSS allocates its solve workspace lazily and
  cannot do so inside a captured stream. Re-factorizing on the bound stream (the
  env is created before `puf_bind_stream`) and warming the exact descriptor shapes
  outside the capture were both tried and measured not to help, so the GPU path
  runs with `--base.cudagraphs=-1` until the in-step solve is replaced by a
  capture-safe one.

Measured throughput in the trainer: **682 agent-steps/s** at 32 agents
(44 ms/step), up from 625 before this session's changes.

## Session 3: GPU env speed-up on the RTX 3090 (verified), and two reset bugs fixed

Everything below was measured on this box (RTX 3090, sm_86, 82 SMs, FP64 = 1/64 rate)
with the parity gate (`gpu/smart_buildings.cu` built with `-DSB_TEST`) and the
supporting tests listed in "How it is verified".  Canonical sources stay in
`complete_port/gpu/`; `plclean/ocean/smart_buildings/` is a copy (`opt/sync.sh`).

### Result (building_train_1 / fairbanks_1, ms per batched env step, gate timing)

| envs | before | after | | other shapes, 8 envs | before | after |
|--:|--:|--:|-|---|--:|--:|
| 1 | 29.0 | 12.4 | | train_2 (263k cells) | 51.2 | 23.2 |
| 32 | 43.0 | 20.2 | | train_4 | 44.1 | 18.8 |
| 64 | 58.1 | 29.9 | | train_5 | 39.3 | 17.6 |
| | | | | train_7 | 20.1 | 8.7 |
| | | | | train_8 | 18.3 | 7.9 |

That is ~2.0x at 64 envs (~0.46 ms per env-step, ~2150 env-steps/s in the gate) and
2.3x at 1-8 envs.  In the trainer (`./puffer train`, graphs off, env = 98% of the time) the rate went
from 682 SPS at 32 agents to ~1.4K SPS at 32 and **~2.0K SPS at 64 agents** (VRAM 2.3 GB;
episode restarts exercised with `--env.max_steps=100`, clean exit).  Per-env cost is
flat from 64 envs up (0.45 ms; 82 and 128 envs measured), so 64 is the new default in
`config/smart_buildings.ini` (`SB_NE` is now 128).

### What changed, and why it is exact

1. **Second cuDSS solve removed.**  `y += A^-1 (applied q)` is `y += Y q` with
   `Y = A^-1 applied` (n x k), which the setup already solves once (for the sensed
   response).  `ycorr_k` applies it with one thread per row of `Y` held in registers
   (FP64-pipe bound: ~80% of peak).  Envs with all-zero `q` are skipped, which is the
   CPU core's own `active` test.  Same linear map, differs from the two-solve form at
   round-off only (field error vs trace unchanged at ~1.7e-11 K).
2. **`convect2_k`, the convection rewritten (bit-exact).**  The chain is still the
   serial MT19937 consumption of `sb_conv_apply`, but: the MT stream is generated
   in-kernel (the same three-pass parallel twist, tempered words in a 4096-word shared
   ring; no 1.2M-word buffer, no silent "pos >= n_words reads 0" hole, no
   generate/advance kernels); whether a word pair `(g, g+1)` is an accept is a pure
   function of the pair, so those flags are computed in parallel into per-parity bit
   rings and the chain jumps from accept to accept instead of walking 141k cells;
   the room's swaps are composed on a u16 permutation and applied once (read-all, then
   write-all).  Draws that run past the generated frontier are retried from their start.
   Cost 21 ms -> 7.7 ms (flat in env count up to 82, one block per env).  The original
   kernel stays as the fallback (max room > 65535 cells / shared budget / candidate
   count > 65535, or `SB_OLDCONV=1`).
3. **Latency fixes that keep the arithmetic identical:** `avg_temp_k` / `sensed_mean_k`
   / `sensed_y_k` (fixed-order FP64 sums - order is the reference, so each stays one
   serial chain per (env, zone/column)) now prefetch 32 operands at a time and run in
   32-thread blocks so they spread over the SMs; the airside's `acc` loop moved out of
   `airside_k` (it ran on 39 threads of one block) into `sensed_y_k`; the exposure
   term `dir + exposure` of the forcing is state-independent and is evaluated once at
   setup with the original expression; the redundant first `avg_temp_k` of each step
   (the field is unchanged since the previous step's tail / the reset) is gone.

### Bugs found while verifying (pre-existing, fixed)

* **The in-step restart ran its occupant steps in the wrong order** (prime, advance, then
  observe; the CPU env observes between the prime and the advance), so the reset
  observation's `num_occupants` saw interval 0's occupancy instead of the pre-episode
  interval.  Host reset and in-step restart now share one `sb_enqueue_reset(gate)`
  with the CPU env's order.

* **The GPU reset observation was wrong**: 59 of 352 entries (VAV zone temperatures
  read 0 K -> normalized -59; AHU run flags 0 instead of 1, static-pressure entries 0
  instead of 1000).  `sb_gpu_reset` never primed the sensed temperatures and
  `ahu_run0` was never uploaded.  Every episode's first observation was affected.
* **The episode restart did not restore the state**: it copied the struct snapshot
  only, so the temperature / interior-mass fields and the VAV/AHU controller arrays
  carried over from the previous episode, and it re-seeded the convection and occupancy
  generators and cleared the occupant chains (the CPU env, like the oracle, keeps those
  across episodes).  Now `reset_sim_k` + `reset_field_k` + `sense_k` do `sb_sim_reset`
  on the device (including priming `avg_temp`/`sensing` from the initial field); the
  generators, occupant chains and `pump_head` are left alone, exactly as the CPU env's
  `puf_reset` leaves them.

### How it is verified (all numbers are measured, final build)

| check | result |
|---|---|
| gate vs oracle trace, all 6 shapes | `SB GPU ENV STEP: OK`; field <= 3.2e-11 K, air_T <= 1.1e-11, reward <= 7.5e-9, **obs 0.000e+00** |
| reset observation vs oracle, all 6 shapes | **0 of 283..703 entries differ** (was 59/352 on train_1) |
| convection vs CPU `sb_conv_apply` (`convtest`), bit-exact field + RNG state | 30 baked blobs x 100 steps x 4 envs, p = 0.1: 0 mismatches; train_1, p in {1e-4, 3e-3, 0.5, 0.9, 1.0}: 0 mismatches; 3000 steps x 8 envs: see below |
| multi-episode, GPU vs the CPU env's reset/step sequence (`csim/test_episodes`), 3 episodes, 4 envs | all 6 shapes: **obs bit-exact in every episode** (train_1: 3 x 900 steps), reward <= 2.6e-11 per step |
| the same multi-episode diff through the trainer's in-step restart (`SB_INSTEP=1`: no host reset, the step itself restarts the batch) | all 6 shapes OK (terminal observation is overwritten by the reset observation, as in the env, so that step is compared on reward only) |
| convection, 3000 steps x 8 envs (about 2.8 episodes of RNG stream) | 0 field / generator-state mismatches |
| fallback convection kernel (`SB_OLDCONV=1`) through the gate | OK |
| trainer: `./build.sh smart_buildings`, `./puffer train` 64 agents; `--env.max_steps=100` | builds, runs clean, ~2.0K SPS |

`opt/run_all_gates.sh <gate-binary>` runs the matrix.  Reproduce a single piece with
`./gate <blob> <trace> <nenv> [ref.bin]`; the reference for the multi-episode mode is
`csim/test_episodes <blob> <trace> <episodes> <out.ref> [steps]`.

### What bounds the step now (nsys, 64 envs, ms per step) - and what was ruled out

| kernel | ms | note |
|---|--:|---|
| cuDSS fwd+bwd (one batched solve) | 13.5 | FP64-pipe bound, linear in RHS count (0.21 ms/env) |
| `convect2_k` | 7.7 | one serial chain per env, ~390 clk per accept/draw; LDS latency 23 clk, ALU 3.5 clk |
| `ycorr_k` | 1.5 | ~80% of FP64 peak |
| `sensed_y_k` + `sensed_mean_k` + `avg_temp_k` | 2.7 | fixed-order FP64 chains: a dependent DADD is ~52 clk on GA102 |
| everything else (forcing, scatter, seq, reward, ...) | ~3.5 | |

Measured and rejected: cuDSS SPD/Cholesky (halves the factor, same solve time), nested
dissection levels, AMD ordering, solve algorithm, pivoting off (no change; the default
ordering is already ND with 6.4M factor nnz), row-major RHS (`NOT_SUPPORTED` in solve),
deterministic mode (`NOT_SUPPORTED` for nrhs > 1 - so the solve is **not bitwise
reproducible run to run**, field differences ~1e-13 K), splitting the batch into two
streams to hide the convection (the exposed convection of the last group cancels the
gain).  ncu on cuDSS's `bwd_ker`/`fwd_ker` shows the FP64 pipe 57-75% busy while
executing 3.5-5x more FP64 instructions than the factor's nnz requires (tile padding on
the many width-1/2 supernodes), so a custom solve could in principle win, but the
dependency depth (~1000 levels) and the 1/64 FP64 rate make a realistic gain well under
2x for a large, risky rewrite; it was not attempted.


## Session 4: cuDSS replaced by a device replay of the baked SuperLU factors (bit-exact), producer/apply convection

Everything here is measured on the RTX 3090 with the gate matrix of Session 3 (all still green) plus the
new checks below.

### Result (building_train_1 / fairbanks_1, ms per batched env step, gate timing)

| envs | start of session 3 | end of session 3 | now | now, env-steps/s | now with `SB_LU_FAST` |
|--:|--:|--:|--:|--:|--:|
| 1 | 29.0 | 12.4 | 9.9 | | |
| 32 | 43.0 | 20.2 | 10.8 | 3.0K | |
| 64 | 58.1 | 29.9 | 14.6 | 4.4K | 13.4 |
| 128 | | | 21.0 | 6.1K | 18.1 |
| 256 | | | 35.5 | 7.2K | 29.1 |
| 512 | | | 67.9 | 7.5K | |

Trainer (`./puffer train`, graphs off): 682 SPS (32 agents, before) -> 5.6K SPS at 128 agents (3.2 GB
VRAM) and 6.3K at 256 (4.8 GB) - measured before the last fusion below, which is worth another ~8% at 256.  Small batches (<= ~16 envs) are now ~1-2 ms slower per step than with
cuDSS because the solve is latency-bound; training batches are >= 64.  The other five shapes at 8 envs:
train_2 19.8, train_4 13.8, train_5 11.9, train_7 7.7, train_8 7.9 ms.

### What changed

1. **The solve is the baked SuperLU factorization, replayed on the device** (`gpu/sb_gpu_lusolve.cuh`);
   cuDSS is gone from the step, from setup (no 640 ms analysis/factorization, no factor storage) and from
   `build.sh`.  The CPU core solves `y = b[ir]; L z = y; U w = z; x = w[pc]` with push updates; for a fixed
   row the push visits the columns in ascending (L) / descending (U) order, so the same additions can be
   done row by row as a pull with the multiply and the subtract rounded separately.  That makes every
   value **bit-identical to `sb_lu_solve`** (`lutest`: 0 of up to 4.0e7 values differ, 30 baked blobs x
   {5, 40} envs plus 200 and 480 envs on two shapes) and independent of scheduling, so the step is also
   deterministic run to run (cuDSS was not).  Consequences measured in the gate: air-side error vs the
   oracle trace 5.7e-12 -> 3.4e-13, reward error vs the CPU env (multi-episode) 2.6e-11 -> 6e-14 per step.
2. **How it is fast.**  A warp owns a row (lanes over envs, `RPL` envs per lane, row-major `[row][ld]`
   work arrays), rows are claimed from one atomic ticket counter and a row waits on per-row epoch flags
   (acquire/release at GPU scope, plus `fence.acq_rel` after the wait: without it the data load can pass the
   flag load - that race produced wrong values until the fence was added).  Three measured facts shaped it:
   (a) 915 of the 991 dependency levels are *narrow* (2-8 rows) but hold 64% of the nnz, so latency per level
   matters (a hop is ~0.8 us); (b) **tickets must be handed out in dependency-level order**, not index order -
   with index order the in-flight warps filled with blocked rows and the critical rows queued behind them
   (15 ms -> 7.4 ms from this one change); (c) on GA102 a warp-wide DFMA costs ~20 clocks of the SM's FP64
   pipe whatever the number of active lanes, so a lane should always carry a real env.
   The forcing is written straight into the solver's permuted layout and the output unpermuted, both as
   32x32 shared-memory tile transposes.
3. **`SB_LU_FAST`** (compile flag): one fused multiply-add per term and a multiply by 1/U_ii instead of the
   divide.  Deterministic, ~2e-14 absolute off the CPU replay, and 3% faster at 64 envs, 17% at 256
   (31.8 vs 38.5 ms).  Off by default: the exact path costs little and is the reference's own arithmetic.
4. **The convection is a producer/apply pair.**  The swap decisions depend only on the generator stream and
   the room tables, never on temperatures.  `convect2_k` now runs on a side stream at the start of the step
   (hidden behind the solve) and emits the room's net permutation as (dst, src) cell pairs; the step applies
   it with two parallel kernels (gather then scatter).  Every env of a batch loads the same scenario and so the
   same seed, hence identical streams: one producer block serves the whole batch (`SB_CONV_PER_ENV=1` gives
   each env its own).  Bit-exact vs the CPU core (`convtest`, 30 blobs, probabilities 1e-4..1, 3000 steps).
5. **The step is CUDA-graph capturable** (no host-varying launch parameters: epoch, ticket and step counters
   live on the device).  `SB_GRAPH=1 ./gate ...` captures one step and replays it for the whole trace; the
   oracle gate, the reset observation and the multi-episode diff (host reset and in-step restart) pass under
   replay.  The launch gaps are negligible at these batch sizes, so there is no speed claim - only that
   `base.cudagraphs=1` is now legal.

6. **`ycorr_k` now carries the interior-mass update and the field scatter** (the old `mass_scatter_k`
   pass over `y` is gone) and only the cells the FDM does not solve are filled with the ambient temperature
   (`fill_amb_k` over a precomputed list) instead of filling all 205k and overwriting 148k: 256 envs
   38.5 -> 35.6 ms.  Gate matrix unchanged (obs bit-exact, multi-episode reward error <= 1.1e-14).
   Measured and reverted: a two-phase gather for the three fixed-order sums (gather pass + streaming chain) -
   correct, but 256 envs got slower (35.6 -> 37.9 ms) and 64 envs did not move.

### A third pre-existing bug, fixed

* **Observation stride.**  The kernels wrote env `e`'s observation at `e * obs_len` (352..703) into a framework
  buffer whose per-agent stride is `OBS_SIZE` = 1024: every agent but the first saw garbled observations in the
  trainer.  The gate could not see it (it read its own buffer with the same wrong stride).  The stride is now
  `SB_OBS`, the gate reads with it, and the padding is zeroed at reset.

### What bounds the step now (nsys, 256 envs, ms per step)

solver 17.8 (FP64 issue slots: two per term in exact mode), `ycorr_k` 6.2 (FP64, ~76% of peak), `mass_scatter_k`
2.4, the three fixed-order FP64 sum chains 4.6, forcing + unperm 2.2, the rest ~5; the convection producer
(9 ms) runs concurrently with the solve and is not on the path.  Tried and rejected: splitting the batch into
env groups on separate streams to overlap latency-bound kernels with the solve (no gain - the chains are FP64
latency bound and the solve saturates the FP64 pipe), chunk sizes 2..32 in the solver (4-8 is best), more or
fewer resident warps (164 blocks x 128 threads), cuDSS knobs (see Session 3).


## Session 5: towards 10K env-steps/s (exact observations kept)

Measured on the RTX 3090 with the verification matrix (`tools/verify_all.sh --trainer`, 41 checks, all pass).

| envs | session 4 | now (gate ms/step) | env-steps/s | trainer SPS |
|--:|--:|--:|--:|--:|
| 64 | 14.6 | 12.1 | 5.3K | |
| 128 | 21.0 | 17.1 | 7.5K | ~7K |
| 256 | 35.5 | 27.4 | 9.3K | 8.7K |
| 384 | | 39.6 | 9.7K | |
| 512 | 67.9 | 51.9 | 9.9K | 9.0K |

1. **The default solve is now the fused-multiply-add build** (`SB_LU_EXACT` selects the bit-identical one).  The
   evidence behind the switch: over a full 8640-step episode on train_1 (3 x 900-step and one 8640-step CPU reference),
   **0 differing observation values** and a per-step reward difference of at most 9e-13 against the CPU env (the exact
   build: 1e-13), where the reference's own reward tolerance against the oracle is 7.5e-9; deterministic run to run;
   oracle gates identical.  `tools/build.sh` builds `gate` (default) and `convtest`; `SB_LU_EXACT=1 tools/build.sh`
   builds `gate_exact` and `lutest` (bit-for-bit solve test), and `verify_all.sh` runs both.
2. **Any number of envs per solver row** (`rpl` = ceil(envs/32), 1..16, all instantiated): a lane carries one env, no padded
   lanes beyond the last 32-env row (320 envs used to run as 512 columns).  Scaling is now smooth.
3. **No per-cell input array**: the internal gain is one division per zone (`zone_qdiv_k`) and the forcing looks the value up
   through a cell->zone table; the redundant second `input_q` pass is gone.  `tmean_k` (a field mean that only feeds the reward
   for an AHU without VAV area) is skipped when every AHU has area (decided once at setup from the static areas).
   64 envs 13.3 -> 12.1 ms, 256 envs 29.2 -> 27.3 ms.
4. **Measured and rejected** (so nobody repeats them): exact FP64 multiply done on the integer pipe (bit-exact over 2e9 random
   products, but no faster: the 64x64-bit multiply costs as many issue slots as the DMUL it replaces); cuBLAS FP64 fixed-point
   emulation for the correction GEMM (runs on sm_86, 27 ms vs 5.5 ms native); the solver as 2-4 concurrent env groups (<=5%);
   lane-contiguous column ownership with 16-byte loads (no change - the solve is FP64-issue bound); more envs per correction
   block, other block counts/sizes for the solver (flat or worse).
5. **Where the step is now** (256 envs, ms): solve 11.2, correction GEMM 6.6 (cuBLAS native DGEMM of the same shape: 5.5), three
   fixed-order FP64 sum chains 4.6, forcing+unpermute 1.9, convection apply 1.0, the rest ~2; the convection producer (8.8 ms) runs
   beside the solve.  The FP64 pipe (1/64 rate on GA102) is the wall: past 512 envs the marginal cost is ~0.1 ms/env, i.e. ~10K/s.


## Session 6: building new buildings and scenarios with the generators (verified)

The starter-kit's procedural generators drive the port unchanged; nothing in the starter-kit tree is modified:

```bash
tools/new_scenarios.sh --seeds "500 501 502" --weathers "fairbanks_1 phoenix_5" --verify
# = run_floor_generator (new floor plans + per-building materials) -> run_scenario_generator
#   (building x weather cross product, config.gin) -> pysim/bake_scenario.py (.sbb blob)
#   [--verify: oracle trace + CPU multi-episode reference + the GPU gates]
./puffer train --env.blob=complete_port/artifacts/blobs/building_0511__phoenix_5.sbb ...
```

Tested end to end: buildings 0500, 0502, 0510, 0511 (new footprints, 71-120 zones, 2-4 floors, up to 1.15M control
volumes / 0.9 GB blobs) x weather presets fairbanks_1, phoenix_5, denver_7: the oracle gate (reset observation 0 diff,
observations 0 diff, field <= 1.9e-11 K), the 2-episode diff against the CPU env (host reset and in-step restart) and CUDA-graph
replay pass, and `./puffer train` runs on them (building_0511, 71 zones: 2.3K SPS at 64 agents; building_0502, rooms of
18 655 cells: 1.7K SPS) - cost scales with the number of control volumes (train_1 has 205k, 0511 has 384k, 0502 has 534k).

Two limits of the first port were hit by the generator's default footprints (larger than the shipped six) and removed:
* **Rooms above ~13 000 cells** did not fit the shared-memory convection kernel (and the old kernel's 8 bytes/cell did not
  either), so convection silently never ran (`conv launch: invalid argument`).  `convect2g_k` keeps the swap list and the
  permutation in global memory (the same source, included with `CV2_GLOBAL=1` from `gpu/sb_gpu_convect2.inc`); chosen
  automatically; bit-exact vs the CPU core (`CV_GLOBAL=1 convtest` on train_1 and on building_0502).
* **More than ~110 coupled VAVs** did not fit the k x k air-side matrix in shared memory: `airside_gk` (same body,
  `gpu/sb_gpu_airside.inc`) keeps the matrix in global memory.
* **Actions**: the env pads to 256 continuous / 16 discrete heads and says so at start-up if a scenario needs more.  Building
  0500 (278 continuous) needs `SB_CONT_BIG=1` (320 heads; `SB_CONT_BIG=1 tools/build.sh`, and `-DSB_CONT_BIG` in the trainer build).
  Hard limits otherwise: 1024 observation values, 256 AHUs, 65 535 cells per room.
* A batch is one scenario (all envs share the blob, hence the seed); to mix scenarios run one process per blob.


## Session 8: multi-building training 4.1K -> 11.8K SPS (exact observations kept)

Goal: the default multi-building trainer (30 shipped scenarios) at >= 11K SPS on the RTX 3090. Start: 30 groups x 16 envs
(480 agents), 4.1K SPS, 21 GB VRAM. The profile showed (a) the air-side correction `y += Y q` (dense n x k per env) cost
MORE FP64 than the triangular solve itself (all six buildings: 57.7M vs 46.2M FMAs per env-step), (b) 30 persistent
solves of 16 envs each (half-empty warps) plus 30 chains of small latency-bound kernels contending for the GPU.

| step (all verified) | train_1 / train_2, 256 envs, gate ms/step | trainer SPS (480 / 960 agents) |
|---|--:|--:|
| start | 27.3 / 77.9 | 4.1K / - |
| 1. air-side correction folded into the solve | 21.6 / 48.9 | |
| 2. parallel pivot search in `airside_k` | 21.4 / 42.4 | |
| 3. weathers of a building share a group | (same) | 8.6K / 9.8K |
| 4. segmented `sensed_R`, sensor/zone-major sums | 20.2 / 41.0 | 8.7K / 10.1K |
| 5. field stored `[cell][env]` | 19.1 / 39.6 | 9.3K / 11.1K |
| 6. one weather per 32-env slab: values shuffled, not loaded per lane | | 9.3K / **11.8K** |

1. **Air-side correction in the solve.**  The oracle solves `y0 = A^-1 b`, senses `S y0`, solves the k x k air system for
   q, then solves again `A^-1 (P q)`.  The port now runs the forward half `z = L^-1 P_r b`, reads the sensed values
   straight off z (`R z`, `R = S P_c U^-1`), solves for q, and starts the backward half from `z + Z q` (`Z = L^-1 P_r P`):
   one triangular solve carries both.  Z and R are sparse (train_1: 58K + 189K entries vs 5.8M for the dense Y) and are
   built at setup (Z from a forward solve of the applied columns, R by a host U^T solve).  Their patterns are the
   STRUCTURAL reach (symbolic, from the L/U/applied/sensed patterns), numerical zeros stored, so the order of every sum
   depends on the building only.  Same round-off class as the FMA solve: gates unchanged (obs 0 diff, field <= 1.6e-11,
   multi-episode reward <= 5e-13), full 8640-step episode 0 differing observations, reward <= 9.1e-13.
2. `airside_k`: the pivot search (warp argmax, same tie rule), the row swap and the multipliers run on the whole block;
   every value is the serial expression, so the output is bit-identical (8.8 -> 2.3 ms for train_2 at 256 envs).
3. **Weather groups.**  Blobs of one building baked for different weather stations differ only in a known list of
   sections (LU values, convection coefficient, ambient series, calendars incl. `occ_hour` which differs across time
   zones, start epoch, ...).  `sb_weather_compatible` compares every other section byte for byte; compatible scenarios
   share a group (`env.merge_weathers = 1`, default; at most `SB_NE` envs).  Per-env weather index `envw`: every
   per-weather table is laid out `[w][...]`, the solver reads its values per weather (template mode 1: per lane; mode 2:
   one weather per 32-column slab, values shuffled like the single-weather path).  Nothing is keyed on names: new
   stations / buildings merge automatically when their blobs allow it.  30 scenarios -> 6 groups, VRAM 21 -> 11 GB.
   Gate `gate wmerge <nper> <steps> <blob> <trace> ...`: every env of a merged group is bit-identical to its weather run
   alone, and its observations equal that weather's oracle trace (train_1 and train_8, all 5 stations each).
4. `sensed_R` cut into fixed 256-entry segments (two-phase, deterministic); `sensed_mean_k` / `avg_temp_k` threads
   sensor/zone-major (same sum, same order, uniform index loads).
5. The field and the interior-mass temperatures live `[cell][lu_ld]` (the solver's column layout): forcing, the output
   epilogue (`lu_out_k`), `fill_amb`, `avg_temp`, `sensed_mean` and the convection apply are coalesced and the two tile
   transposes are gone.  `GpuSim::temp` points at the env's column, stride `ts`.  `convtest` updated (bit-exact, both
   convection kernels, global-memory rooms).
6. See the table; 32 envs per scenario is the efficient size (1440 agents = 48 per weather falls back to per-lane values
   and gives 10.1K at 22 GB).

**Measured and rejected:** one fused persistent solve over all groups (`lu_solve_mg`, merged level order, phased step
`sb_mg_step`): bit-identical (`gate mgtest`) but slower in the trainer (960 agents 10.8K vs 11.1K) - the solve at
3-5 envs per lane is bound by per-term overhead, not by tails; kept opt-in (`SB_FUSE=1`).  Start- vs end-aligned
merged order: no difference.  Solver blocks 41/82/246/328 instead of 164 for the per-group solves: slower.  Waiting
for a whole row's flags before its FMAs (one fence per row): bit-identical, 30% slower.  `LU_CHUNK` 8 at 3 envs per
lane: slower in the trainer.  Other scipy orderings (COLAMD, MMD_ATA): 60-70% more fill (A is symmetric).
Solver efficiency vs envs per lane (train_1, us per env-step): 1: 168, 3: 73, 5: 54, 8: 45, 16: 43 - the reason the
batch wants >= 32 envs per weather.


## Session 9: why PPO did not learn, and the Stage 1 fixes (trainer)

The default PPO run (20M steps, 960 agents, 30 scenarios) ended at a per-step score of 0.71 (pid_water: 0.054) with
clipfrac ~1 and KL 10^2-10^4 on every update and entropy rising.  Measured causes and fixes:

| finding (measured) | fix | evidence |
|---|---|---|
| bf16 policy: rollout log-probs != training recomputation; summed over ~250 heads the ratio leaves the clip range | trainer built fp32 for this env (`build.sh` case, `PRECISION_FLOAT`) | lr = 0: bf16 KL 0.71 / clipfrac 0.88, fp32 0.000 / 0.000 |
| joint ratio over hundreds of heads: any update clips | `[train] ratio_per_dim = 1`: geometric-mean ratio exp(dlogp / D) | lr 3e-6 joint: clipfrac 0.77 |
| unbounded Gaussian means drift past the [-1, 1] clip, KL independent of lr | `[policy] tanh_mean = 1`: mean = tanh(logit), chain rule in the loss | per-dim + tanh, lr 3e-4: KL 0.008, clipfrac 0.11 |
| std 1 at init: much of the exploration lands on the clip | `[policy] init_logstd = -1.0` | entropy 363 -> 107 |
| every env of an epoch sees the same hours of the same day | `[env] desync = 1`: first episode cut at a random step; per-env restart flags (`gate resetiso`) | batches span the calendar |
| returns of hundreds, no normalisation; entropy bonus the only force left | `[env] reward_scale = 0.1`, gamma 0.995, `ent_coef = 0` | value loss 270 -> 4.7 |

Result of the 10M-step run with all fixes: the optimisation is healthy (KL 0.003-0.007, clipfrac 0.02-0.1) but the
score did not improve (mean |live reward| over scenarios 0.56 in the first 4 epochs, 0.76 in the last 4; full
episodes -0.53 to -1.09; 76% of zone-steps out of band).  Cold climates got worse (fairbanks_1 -2.7 -> -5.6): the
policy saves energy by not heating, and with a ~17 h horizon the comfort cost of a cold building arrives too late.
Open: the reward/credit-assignment problem (horizon, comfort shaping), a controller prior (GPU PID baseline for
residual RL / behaviour cloning), a structure-aware policy.

### 2026-10-16 01:55 IST finding: the observation reaches the policy largely UNSCALED
The env writes the oracle's observation vector, z-scored with the blob's per-field constants (`f_norm_mean/var`).
Only zone air temperature (mean 295 K, var 25), damper/valve commands (0.5 / 0.25) and a few hot-water fields carry
non-trivial constants; most fields have mean 0 / var 1, i.e. arrive RAW: zone heating/cooling setpoint sensors ~294-297 K,
outside-air temperature in K, static pressure in Pa, flow rates in m3/s, the AHU supply-air temperature setpoint in K.
PufferLib applies no observation normalisation, so every PPO run so far fed an MLP inputs in the hundreds to thousands
next to inputs of order 1. (Measured on building_train_1__fairbanks_1: 26 field types, 21 with mean 0 / var 1.)

### 2026-10-16 02:20 IST on-device pid_water base (gpu/sb_gpu_base.inc): verified, and a sensitivity finding
- `gate basecheck` + `research/tools/basecheck.py --obs-order`: the CUDA port of starter-kit PIDWaterResetPolicy (float32 /
  float64 exactly as numpy 2 / NEP 50, numpy pairwise means, round-to-nearest per op) gives BIT-IDENTICAL actions to the
  starter-kit class over full 8640-step episodes (0 of 838,080 values on train_1/fairbanks_1, train_1/phoenix_5, ...)
  when both take each AHU's zones in observation order. pid_water itself uses the metadata's room order, which differs
  in 28 of 226 AHU groups across our splits; that changes only the float32 rounding of an AHU's mean zone temperature.
- Trainer path (`./puffer eval`, canonical + residual, zero-init policy, deterministic): dev score 0.054915 vs the
  pid_water reference 0.054958 (`research/orchestrator/baselines/gpu_dev_pid_water.json`). The only building whose room
  order equals observation order (train_5) agrees to 1.9e-5 relative (float32 episode-return accumulation); the five
  others diverge by 0.02-2.5% per scenario.
- FINDING: the closed loop is numerically chaotic. Last-bit differences in one AHU's mean temperature move a scenario's
  30-day score by up to 2.5% (train_8/phoenix_4), via on/off and deadband thresholds. Per-scenario differences of a few
  percent are therefore within numerical noise; claims need paired comparisons over many scenarios, well beyond that.

### 2026-10-16 02:59 IST pre-existing set-up bug fixed: buildings with ~11k-cell rooms aborted
`sb_gpu_setup` raised the fallback convection kernel's (`convect_k`) dynamic shared memory to max_room x 8 bytes
whenever that alone was under the 99 KB opt-in limit, ignoring the kernel's static shared arrays; for max_room ~11 300
(gen-val building_20001: 90 384 B + static > limit) cudaFuncSetAttribute failed and set-up aborted ("cuda err invalid
argument"), although such buildings run convect2_k and never the fallback. This is why research/tools/sbgpu could not
evaluate gen-val. Fix: include `cudaFuncGetAttributes(convect_k).sharedSizeBytes` in the check.

### 2026-10-16 03:40 IST run r1 (residual PPO, dev, 20M steps): a cold-climate collapse, and the comfort guard
- r1: `--env.blob_dir=/art/train --env.canonical=1 --env.residual=1 --policy.zero_init_actions=1`, 960 agents, 11.4K SPS,
  20M steps (`results/runs/r1_dev_resid/`). Deterministic dev score (`tools/eval_policy.sh`): final 0.2730, 14.7M
  checkpoint 0.0578, vs base 0.0549. Both are better than base on 16-17 of 30 scenarios, with large hot-climate gains
  (train_8/phoenix_4 0.164 -> 0.040, train_5/singapore_4 0.110 -> 0.043), but train_1/fairbanks_1 FROZE (final: 6.92,
  productivity -11.4; base 0.214).
- Cause (TensorBoard, `tools/tbread.py <tb> <tag substring>` now prints series): fairbanks_1's training reward was better
  than base (-0.14..-0.19 per step) until ~13M, then degraded monotonically (comfort deviation 1 K -> 16 K at 19.7M) while
  the cosine-annealed learning rate went to 0 (KL 4e-7 at the end): the policy drifted into a deep-cold state the zone
  features (T - 295) / 5 never covered, and could no longer correct it.
- Comfort guard (`gpu/sb_gpu_base.inc` guard_k, `env.guard_d0/guard_d1/guard_hot`, default off): the residual's
  authority on an action falls 1 -> 0 as its zone (AHU: its worst zone; hot water: the building's worst zone) goes
  guard_d0 .. guard_d0 + guard_d1 K below the base's band, so far outside the band the action is pid_water's own. Mirrored
  in `submission/residual_policy.py` (config guard_*). A low-limit / freeze protection, standard in HVAC practice.
- Post-hoc (r1 checkpoints, no retraining; `results/eval/r1_*_guard_dev.json`, `r1_*_cguard_dev.json`):
  symmetric guard (1..3 K outside): 14.7M 0.0537, final 0.0561 (fairbanks fixed, but the hot-climate gains vanish:
  phoenix_4 0.153; they come from states far ABOVE the band). Cold-only guard (1..3 K below): 14.7M 0.0493 (-10% vs base,
  17/30 better, fairbanks_1 0.195 < base), final 0.0551. Cold-only is the default side.
- Tool fix: `tools/run_train.sh` cleared the trainer's TensorBoard dir outside the GPU queue, so a queued second run would
  have wiped a running one's logs; the clean-up now runs inside the queued job.

### 2026-10-16 04:45 IST run r2 (r1 + cold comfort guard, dev, 20M): better on dev, WORSE on unseen buildings
- r2 = r1 args + `--env.guard_d0=1 --env.guard_d1=2` (`results/runs/r2_dev_cguard/`). Training stable: fairbanks_1's reward
  stayed at -0.15..-0.25 per step to the end (r1: -3.66).
- Deterministic eval (`results/eval/r2_*.json`, paired per scenario vs the zero-residual base on the same split):
  dev 0.0446 vs 0.0549 (17/30 better, mean diff -0.0103 +- 0.0053 SE); 14.7M checkpoint 0.0479.
  gen-val (6 unseen generated buildings x 5 unseen weather strips) 0.0687 vs base 0.0569 (5/30 better, +0.0119 +- 0.0048);
  gen-val-s 0.0522 vs 0.0492 (8/30 better, +0.0031 +- 0.0008).
- FINDING: a residual policy trained on the 6 shipped buildings OVERFITS them: its gains do not transfer, and the losses
  sit in warm climates (singapore_5 / phoenix_7 strips; building_20009: 0.038 -> 0.163), i.e. in states above the band
  that the cold-side guard does not cover. Base reference numbers now exist for both splits
  (`results/eval/base_gen-val.json` 0.05688, `base_gen-val-s.json` 0.04917). Selection must use gen-val / gen-val-s only.

### 2026-10-16 05:30 IST many training buildings: VRAM limit and building rotation (`tools/run_rotation.sh`)
- A batch holds every group's geometry / factorization and per-env state, so VRAM grows with the SUM over buildings of
  (building size x envs), not with the agent count alone. Measured: mix49 (dev + 19 generated buildings) fails at 784
  and at 392 agents (20.7 GB before the trainer's buffers; cudaGraphInstantiate / cudaMalloc fail); one chunk of 4
  generated + 1 shipped building (blobs 11 GB) fails at 800 agents. Fits: <= ~1 GB of per-building blob (one weather)
  per batch at 48 envs per scenario (rot1: 20 scenarios, 960 agents, 22.0 GB peak, 6.8K SPS). The trainer also needs
  total_agents divisible by minibatch_size / horizon (= 8). building_10000 (1.27 GB) is left out (larger than all of dev).
- `tools/run_rotation.sh <name> <steps> <segments> <envs/scenario> "<sets>" [args]`: segment i trains on set i mod n,
  starting from the previous segment's final checkpoint (`--base.load_model_path`; Adam restarts cold). Sets rot0..rot7
  (`artifacts/sets/`, made with `research/tools/make_blob_set.py ... --only=`): the 25 training buildings (6 shipped +
  19 gen-train) bin-packed by blob size (0.90-0.99 GB each), all 5 weathers per building, 125 scenarios.
- Run r4_rot: 8 segments x 4M, 48 envs/scenario, constant lr 2e-4 (anneal_lr 0), cold guard 1..3 K.

### 2026-10-16 07:50 IST run r4 (rotation over 25 buildings): worse than the base on every split
- r4_rot finished all 8 segments (`results/runs/r4_rot.chain`, final checkpoint `r4_rot_s07/ckpt/0000000003932160.bin`).
  Deterministic eval with the cold guard (`results/eval/r4_final_*.json`, paired vs the zero-residual base):
  dev 0.0619 vs 0.0549 (11/30 better, mean diff +0.0070, 95% CI excludes 0);
  gen-val 0.0685 vs 0.0569 (1/30 better, +0.0116);
  gen-val-s 0.0574 vs 0.0492 (2/30 better, +0.0082).
  The midpoint checkpoint (segment 3) is already worse on gen-val (0.0641, 2/30 better), and the final checkpoint is worse still.
- The damage is comfort, not energy: mean |productivity| rises by 0.017 on gen-val while energy+carbon rises by 0.007.
  It is worst in warm weather (gen-val singapore mean diff +0.022, phoenix +0.015) and it hits every held-out building.
- FINDING: rotating buildings through a slot-specific MLP, with Adam restarted cold each segment, does not buy generalization.
  It unlearns the base even on the shipped buildings (dev is worse, where r2 on the same architecture was 0.0446).
  The flat canonical MLP has a separate input weight for every zone slot, so a slot's meaning changes when the building changes.

### 2026-10-16 08:20 IST shared zone policy (`-DSB_ENTITY`): the zero policy matches pid_water
- Compile-time variant of the trainer (default `./puffer` is unchanged). Each building is `CN_ZS` = 90 zone agents under the
  stock MLP, so the same weights apply to every zone of every building. An agent's observation is that slot's 9 canonical
  features plus the 19 global features. Its two actions are the zone damper and reheat residuals; AHU and hot-water
  residuals stay 0, so those actuators remain pid_water. The building reward is split evenly over the zones that exist
  (the shared gradient keeps the scale of one agent per building). Empty slots are action-masked and get reward 0.
  `puf_mask_init` is how the mask reaches the GPU trainer (the framework's buffer was previously left all-ones).
- Gate for the wiring: zero-init, deterministic, full 8640 steps, 4 envs of building_train_1/fairbanks_1. The entity
  policy and the flat canonical zero policy produce the same logged episode: mean reward -0.2137526483, and the same
  productivity, energy and carbon (`results/eval/entity_zero_f1.json`, `results/eval/base_zero_f1x4.json`).
- A short CUDA-graph smoke train (360 agents, 50-step episodes) ran and took a gradient step (clipfrac 0.001).
- The numpy submission policy does not implement this layout yet. A checkpoint is not exportable until it does.

### 2026-10-16 15:25 IST P0 measurement + P1 batch layout: DRAM-bound, layout worth <10%, use 32 envs/scenario
- P0 tools (standalone, not in the gate path): `tools/roofline.py` (offline cost model from the `.sbb` sections:
  solve terms = nnzL+nnzU+nnzZ, state = 8 B/term/env-step, DRAM floor = state/936 GB/s, FP64 pipe = terms/278e3 us;
  writes `results/perf/roofline_<stamp>.{md,json}`), `tools/bench.sh` (gate sweep per blob x nenv, parses
  `phase ms/step`; `--graph` runs SB_GRAPH=1), `tools/bench_layout.sh` (trainer A/B: same blobs, one
  envs-per-scenario per run, steady-state epoch SPS from TensorBoard), `tools/run_stats.py` (agents / E / mode /
  RPL / SPS / us-env-step / VRAM of any finished `results/runs/<name>`).
- P1 code: `gpu/smart_buildings.cu` now prints each batch/group's solve layout at startup
  (`group 2/3: 160 envs, 5 scenario(s), ld 160, 5 env(s)/lane, value mode 2`) and warns when a group has >=32
  envs but <5 envs per lane (the regime that costs 3-5.7x the floor). `verify_all.sh --trainer`: ALL CHECKS PASSED.
- Cost model (23 buildings of the rot/diverse sets): 15.6M solve terms = 124 MB state per env-step, i.e. the DRAM
  floor is 2.4x the FP64-pipe floor. Every generated building is DRAM-bound.
- A/B on identical blobs (`rot2b` = building_10001 + building_10015 x 5 weathers, batch floor 143.2 us/env-step):
  32 envs/scenario (value mode 2, 5 envs/lane) 4360 SPS = 229.4 us = 1.60x floor in 10.9 GB; 48 (mode 1, 8/lane)
  4136 SPS = 241.8 us = 1.69x in 15.1 GB; 64 (mode 2, 10/lane) 4284 SPS = 233.4 us = 1.63x in 18.2 GB. Doubling the
  agents per scenario doubles the step time and leaves SPS unchanged: the step is a repetition of the per-env cost.
- On the user's sets: rot2 at 32 envs/scenario 5714 SPS / 175.0 us (480 agents, 13.9 GB) vs its 48/weather run
  5591 / 178.9 us (720 agents, 18.8 GB); rot1 at 32: 6821 / 146.6 us (640 agents, 15.3 GB) vs 6856 / 145.8 us
  (960 agents, 20.8 GB). FINDING: 32 envs/scenario matches the SPS of 48 with 2/3 of the agents and ~3/4 of the
  VRAM. The `diverse*` sets (4-8 envs/scenario -> 40/group, mode 1) run at 2.21x the floor vs 1.41-1.50x for the
  rotation sets.
- RPL curve at one weather (gate, building_train_1, floor 53.6 us): 32 envs 303.1 us (5.7x), 64 160.8 (3.0x),
  160 89.1 (1.66x), 256 77.3 (1.44x), 512 76.5 (1.43x) -> a group needs >=160 envs before a term's index/value
  loads are amortized. Two-group runs sit at 1.60-1.69x the floor, 3-4 group runs at 1.41-1.50x (more groups =
  better latency hiding, another reason to pack buildings, not envs).
- Recommendation (no code change, run args only): 32 envs/scenario (`--vec.total_agents = 32 * blobs`), >=160 envs
  per group, fill VRAM with more buildings rather than more envs per weather; `tools/run_rotation.sh <name> <steps>
  <segments> 32 "<sets>"`. Full report: `results/perf/P0_P1_measurements_20261016.md`.
- Next levers, in expected order: fp32 training build (`SB_F32=1`, ~2x - halves all three solve streams; the default
  build stays bit-exact) *[measured the same evening: 1.27x gate / 1.37x trainer, NOT ~2x - see the P3 entry below]*, second GPU (the 3060's DRAM floor is 2.6x the 3090's per env-step: a second experiment
  slot, not a physics shard), tiled solve (+10-25%: the solve is at 62-70% of peak bandwidth, not compute-bound).

### 2026-10-16 17:00 IST P3: fp32 solve (`SB_F32=1`), training only - 1.37x in the trainer, measured
- `SB_SOLVE_F32` (`gpu/sb_gpu_lusolve.cuh`): the solve's two state vectors (`work` = forward y, `zout` = backward z) and
  its arithmetic (per-term coefficient, accumulator, fma) are fp32; the factors, `qz` and everything outside the solve
  stay double.  The per-row term order is unchanged, so the fp32 build is a deterministic rounding of the same recurrence.
  Builds: `SB_F32=1 tools/build.sh` -> `gate_f32`; trainer `SB_F32=1 tools/train.sh` (the trainer build
  reads the *built binary's* `lu_solve_t...Pf/Pd` names and rebuilds whenever they do not match the request -
  a marker file proved fragile: `verify_all` and manual builds rewrite `./puffer` behind its back).
- Trap worth recording: `__fma_rn` is the *double* intrinsic and silently promotes float arguments.  fp32 *storage*
  alone measured 20% SLOWER (23.6 vs 19.3 ms/step, SASS: F2F=528, DFMA=120, FFMA=0 - every load converted back to
  fp64).  With `LU_FMA` (= `__fmaf_rn` under the flag) the kernel is FFMA=136, DFMA=0, 1859 -> 1576 instructions.
- Gate `phase ms/step` (one group, full sweep in `results/perf/bench_20261016_1659*.tsv`): train_1 @160 14.25 -> 11.52
  (1.24x), @256 19.82 -> 15.05 (1.32x); train_2 @160 28.53 -> 22.50 (1.27x), @256 41.52 -> 30.45 (1.36x); 10015 @160
  54.96 -> 47.07 (1.17x), @256 74.33 -> 56.83 (1.31x).  Trainer, 30 shipped scenarios, 960 agents, 2.4M timesteps:
  **10.5K -> 14.4K SPS (1.37x, wall 221 -> 168 s), VRAM 17.6 -> 16.5 GB**, per-epoch entropy 113.8 vs 113.5 (same
  training regime, only round-off in the physics).
- Why not 2x (ncu, forward solve, train_1 @256): fp64 4.84 ms / 1.76 GB DRAM (39.8% of peak) / 242M L2 sectors /
  16.6% warps active / long_scoreboard 48.6%; fp32 3.70-4.08 ms / 0.43 GB (11.7%) / 133M sectors / L2 throughput
  40.9-43.8% (the highest unit), L2 hit 93.6% fwd / 66.7% bwd, compute 15%, L1 20%.  The fp32 state **halves the bytes
  per request** but the kernel still issues one load per term per env slot: it is request/latency bound, not byte
  bound.  Consistent with the knobs (all measured, all within +-1% on the fp32 build): `SB_LU_TPB=256` (16 warps/SM
  instead of 8) 15.68, `SB_LU_BLOCKS=328` 15.71, `SB_LU_CHUNK2=1` 15.40 vs 15.60 baseline; `SB_LU_BLOCKS=82` 18.57
  (much worse).  Cutting the request *count* needs a layout change (several adjacent envs per lane -> vector state
  loads), not more warps.
- Step composition (new tool `tools/timeline.py`: step period detected from the 30 regular `convect2_k` launches; the
  gate's own per-step oracle compare copy - 24.4 ms of D2H at 256 envs - excluded): main-stream kernels 14.63 ms
  (union) = solve 7.54, fdm_forcing 2.16, lu_out 1.39, sensed_mean 1.05, avg_temp 0.76, seq 0.56, sensed_R_part 0.46,
  airside 0.37, fill_amb 0.23, reward 0.21, conv scatter/gather 0.29, occup+post_seq 0.12; `convect2_k` 10.45 ms on
  stream2, **100% of it overlapping main-stream kernels** (genuinely hidden).  The field side (6.3 ms) is now 42% of
  the critical path for the smallest building - the natural second target, with the same "training-only" argument
  (fp32 `d_temp`/`d_mass_temp`), except that the field is *integrated across the episode*, so it needs a
  long-horizon drift measurement first.
- Numerics (the fp32 build is NOT the gate build; its gate prints FAIL by design): worst observation 1.16e-04, air
  temperature 1.8e-04 K, field 1.6e-02 K on ~3e2 K values, reward 1.6e-07; the **reset observation is identical**
  (0 of 352 entries differ).  The drift *does* grow, sub-linearly and decelerating (measured at 60/240/600/1440 steps
  against precomputed oracle traces: field 2.9e-02 / 8.7e-02 / 1.4e-01 / 1.7e-01 K, i.e. ~N^0.6; the fp64 control at
  the same lengths stays at 1e-11 K).  Extrapolated to a full 8640-step episode that is ~0.5 K of field (0.17%) and
  ~5e-03 K of air temperature - inside the comfort band, so training signal generation is safe; evaluation and the
  submission stay on the fp64 build.  The default build was re-verified bit-exact after the refactor (`SB GPU ENV
  STEP: OK`, reset obs exact), and `verify_all.sh --trainer` also builds the `SB_F32` variant and asserts its drift
  stays under 0.05 K (it is at 1.6e-02 at 30 steps).
- Files: `results/perf/P3_fp32_solve_20261016.md` (full report + the failed A/B lesson: the checkout was stale, so the
  first "1.03x" compared two fp64 binaries), `README.md` (build flag, trainer numbers, two-GPU section), `tools/build.sh`
  (variants get their own `convtest_*`), `tools/train.sh` (`SB_F32`, variant read from the built binary),
  `tools/verify_all.sh` (fp32 drift check), `tools/timeline.py` (per-step critical path + stream overlap),
  `patches/pufferlib_core.diff` (the trainer build case).

### 2026-10-16 18:05 IST P2: the 3060 as a second experiment slot - +11.9K steps/s for a 4.7% tax on the 3090

- `SB_GPU=<index>` (`tools/common.sh` -> `--gpus device=N` in every docker call: bench, eval, run_train, train,
  watch, verify_all) pins a job to one device; `tools/with_gpu.sh <index> -- <cmd>` sets it and holds the per-device
  flock (`/tmp/sb_gpu_<N>.lock`: a second job on the same GPU waits instead of failing; no-arg form lists devices,
  memory and lock state).  Every tool keeps `--gpus all` when `SB_GPU` is unset.
- Per group-step, fp32, gate, one group at 160 envs: train_8__atlanta_1 **8.52 ms (3090) / 11.48 ms (3060)**,
  train_7__atlanta_8 8.66 / 12.31, train_1__fairbanks_1 12.07 / 17.80 - the 3060 is **1.35-1.47x slower, not the
  2.6x its DRAM roofline predicts**: at this batch size the step is latency-bound (the 3090 sat at 12-45% DRAM and
  16.6% occupancy, P3 §4), so fewer SMs and less bandwidth barely matter.
- Trainer, fp32, the three smallest shipped buildings (3 groups x 160 envs = **480 agents**) on the 3060: wall 72.3 s
  for 491,520 nominal steps and 237.9 s for 2,457,600 -> **11.9K steps/s steady state** (slope; the ~20 s process
  start cancels; 12.1K if the *requested* totals are used - the trainer floors `total_epochs`), which is what the sum
  of the three gate times predicts (11.5K): on 28 SMs the three groups' steps add up rather than overlap.  The card held **99-100% util
  and 120-128 W of its 140 W cap** for the whole run, so this is GPU-bound work, and nvidia-smi shows the 12 GB's real
  limit is the one-off CUDA-graph capture (11.9 GB peak, 6.4 GB steady).
- Concurrency (the actual P2 question): 3060 + 3090 at once -> the 3090 took **179 s vs 170.9 s solo (+4.7%)**, the
  3060 took 37 s vs 38 s solo (unchanged).  So the second slot buys 11.9K steps/s for a ~5% tax: a 960-agent 3090
  batch plus a 480-agent 3060 batch = **24.3K steps/s = 1.9x a single 3090**.  On the same three blobs the 3060 is
  0.74x a 3090 slot (11.9K vs 16.4K, the latter from the gate sum).
- **Measurement trap found here** (affects every future number): PufferLib's dashboard/TB `SPS` counts *issued* steps
  against CPU-side phase times - the env phase reads `0 ms` because the step is enqueued, and the CPU races ahead of
  the GPU until a buffer fills (observed per-epoch cadences on the 3060: 2.3 s, 25 s, 2.3 s).  The 3060 run claimed
  117K SPS on the dashboard; its true rate is 12.1K.  Also `total_epochs = total_timesteps / (total_agents * horizon)`
  rounds down (a 960-agent `--train.total_timesteps=2400000` run is 9 epochs = 2,211,840 steps).  **Quote wall time
  per fixed `total_timesteps`**, which is what the P3 A/B did (both arms had the same step count, so its 1.37x stands).
- Files: `tools/common.sh` (`SB_GPU` -> `DEVS`), `tools/with_gpu.sh` (new), `README.md` ("Two GPUs, two jobs").
- Re-measured clean on an idle machine (the *newest* entry): the *rates* above understate both arms (docker/build
  time inside the walls + 3060/drift jobs running concurrently); the +4.7% tax and the ratio claims reproduced.

### 2026-10-16 19:05 IST The trainer's own clock: 17.4K/23.4K steps/s, streams 1.78x, pair 35.7K steps/s

- **The protocol, because it decided every number here**: idle machine, fixed `--train.total_timesteps=1200000`
  (4 epochs, 983,040 steps, 6 groups x 160 = 960 agents), quoted from the trainer's final dashboard frame
  (`Steps` / `Uptime` - a clock that starts after the env is built, so docker start-up, the build and set-up are out).
  The `SPS` field is unusable in any regime: four consecutive frames of the fp64 run read 7.8K / 205.7K / 10.7K /
  209.1K against a true 17.4K (it counts issued steps against CPU-side phase times; env reads `0 ms` because the step
  is enqueued and the CPU runs ahead until a buffer fills).
- **Single 3090 (exact vs fp32)**: fp64 **56.46 s = 17.4K steps/s**, `SB_F32=1` **41.99 s = 23.4K steps/s** ->
  **1.34x**, confirming the P3 A/B's 1.32x wall / 1.37x SPS.  The P2/P3-era absolutes ("10.5K vs 14.4K" dashboard,
  10.0K/13.2K wall-derived) were understated: those walls carried docker/init/build time and the runs shared the
  machine with the 3060 jobs and the 5.3 GB drift-trace dumps.  The ratios were right; use 17.4K -> 23.4K.
- **Multi-group stream concurrency A/B** (`SB_GROUP_SERIAL=1` moves each group's work off its own stream pair onto
  one stream - code only, bit-identical): fp64, same 1.2M -> default **56.67 s** vs serial **101.11 s** = **1.78x**,
  identical losses in both arms (6.241 / old_kl 0.549 / clipfrac 0.757).  The shipped per-group streams (main +
  convection side stream, forked from / joined into the rollout stream) already harvest the overlap: the single-group
  16.6% occupancy is not 6x of stranded headroom.
- **Two-GPU pair** (both jobs launched together, same fp32 binary, 1.2M each): 3090 **43.91 s = 22.4K steps/s**
  (taxed **+4.6%** vs its 41.99 s solo - the P2-era +4.7% reproduced); 3060 `/art/3060` ran the **full 960-agent
  batch** (3 blobs -> 3 groups x 320 envs, 10.5-11.1 of 12 GB) at **74.14 s = 13.3K steps/s** = 0.57 of a 3090 slot.
  **Aggregate ~35.7K steps/s = 2.05x the 3090's exact solo, 1.52x its fp32 solo.**  The fp64 build at 960 agents
  does *not* fit the 3060: `cudaGraphInstantiate` fails at epoch 2 (`src/pufferl.cu:1634`), so the exact 3060 slot
  keeps the 480-agent layout (`--env.groups=3 --vec.total_agents=480`).
- **Exact-build pair, same protocol** (both GPUs fp64; the 3060 on its 480-agent layout, 1,105,920 steps = 9 epochs):
  3090 **55.75 s = 17.6K steps/s** (no measurable tax this time; -1.3%, inside frame noise), 3060 **122.55 s = 9.0K
  steps/s** -> **exact aggregate ~26.7K steps/s = 1.53x the 3090's exact solo**.  The exact path gains less from the
  second card than the fp32 path (1.53x vs 2.05x): the 3060 is capped to the smaller batch (960 agents does not fit)
  and pays the fp64 solve.  Training-time VRAM: 3060 6.3-6.6 of 12 GB, 3090 17.6 of 24 GB.
- Evidence: `results/trainab_{default,serial}.log`, `results/f32ab_{f64,f32}.log`,
  `results/pair_f32_{3090,3060}.log`, `results/pair_f64_{3090,3060}.log` (and the 960-agent fp64 crash:
  `results/pair_f64_3060_960agent_fail.log`).  Docs: `README.md` ("Two GPUs, two jobs",
  "Read the trainer's own clock", "Performance"), `results/perf/P3_fp32_solve_20261016.md` §6.

### 2026-10-16 09:25 IST run r5 (shared zone policy, dev): worse than pid_water on every split
- `verify_all.sh --trainer` passed (`ALL CHECKS PASSED`) before the run, then `puffer_entity` was rebuilt with `-DSB_ENTITY`.
  r5 (`results/runs/r5_dev_entity/`, checkpoint `0000000111974400.bin`): 30 dev scenarios, 2 envs each = 5400 zone-agents,
  111,974,400 agent-steps (the same env-steps per env as r2's 20M), cold guard 1..3 K, zero-init, default lr with anneal.
  Wall 08:41-08:55 IST. The dashboard SPS (~137K) counts zone-agents, 90 per building, so it is not comparable to the
  flat trainer's steps/s.
- Deterministic fp64 eval (`results/eval/r5_final_*.json`, paired vs the zero-residual base):
  dev 0.0608 vs 0.0549 (5/30 better, mean diff +0.0058, 95% CI includes 0);
  gen-val 0.0768 vs 0.0569 (0/30 better, +0.0199, CI excludes 0);
  gen-val-s 0.0667 vs 0.0492 (0/30 better, +0.0175, CI excludes 0).
- The damage is energy, not comfort. Mean change vs the base: dev |productivity| +0.002 and energy+carbon +0.022;
  gen-val +0.014 and +0.058; gen-val-s +0.015 and +0.042. Worst on the warm strips (gen-val singapore +0.032,
  phoenix +0.027 of |mean reward|). Training episode returns stayed in the base's range (about -200 to -1100 per
  episode, i.e. no fairbanks-style collapse) and still finished worse than the base.
- FINDING: sharing one set of weights across zone slots does not fix the generalization failure, when the only learned
  actions are the zone damper and reheat and the zone does not see its AHU. Those two actuators can spend reheat and
  airflow the base was not spending, and supply temperature, pressure, economizer and hot water stay at pid_water.
  The numpy submission path still does not implement this layout.

### 2026-10-16 19:20 IST public leaderboard: the submitted r2 scores 0.08148
- Kaggle public scores (the user, version noted): `submission_r2_best` v5 = **0.08148**, `submission_bang_worst` v1 =
  **0.21092**. The r2 package is the final dev checkpoint `r2_dev_cguard/ckpt/0000000019906560.bin` at res_scale 0.5,
  cold guard 1..3 K (`submission_r2_best/model/config.json`).
- Same policy, local GPU gateway, full episodes: dev 0.0446, gen-val 0.0687, gen-val-s 0.0522. The public number sits
  above both held-out local splits. bang_bang on the public slice (0.211) is the weak baseline; pid_water has not been
  submitted, so 0.08148 is not yet a paired public comparison against the baseline that matters. Locally that comparison
  already goes the wrong way on gen-val and gen-val-s.
- Next, at the measured fast training rate (fp32 solve, 23.4K steps/s on the 3090; evaluation stays on the fp64 binary):
  r6 retrains r2 with one change, `env.res_scale=0.25`, and the submitted checkpoint is re-scored at 0.25 and 0.10
  without retraining. TensorBoard stays on: `live/*`, `scenario/*`, `env/*`, `loss/*`, `perf/*`, `util/*`.

### 2026-10-16 20:17 IST run r6 (r2 with residual scale 0.25): matches the base on dev, worse on held-out buildings
- r6 (`results/runs/r6_dev_scale025/`, fp32 solve, 960 agents, cold guard, zero-init, 20M steps). Trainer clock:
  19,906,560 steps in 23 min 45 s, about 14.0K steps/s (the residual controller is slower than the bare 23.4K measurement).
  The process then exited on `missing key scenario/.../episode_score` (`src/ini.h` `dict_get` calls `exit(1)` for a tag
  that was never declared). The final checkpoint was already written: `ckpt/0000000019906560.bin`. TensorBoard is in
  `tb/`.
- Deterministic fp64 eval at the training scale (`results/eval/r6_final_*.json`, paired vs the zero-residual base):
  dev 0.0546 vs 0.0549 (16/30 better, mean diff -0.0003, 95% CI includes 0);
  gen-val 0.0732 vs 0.0569 (3/30 better, +0.0163, CI excludes 0);
  gen-val-s 0.0590 vs 0.0492 (6/30 better, +0.0098, CI excludes 0).
- FINDING: halving the residual authority removes r2's dev gain (0.0446 -> 0.0546, indistinguishable from pid_water)
  and does not fix the held-out loss. gen-val is worse than r2's own 0.0687. A smaller scale, trained in, is not the lever.

### 2026-10-16 20:40 IST plain PPO, once more, on the scaled observation
- The plain PPO runs already exist and failed. The first (20M, before the trainer fixes) scored 0.71 against pid_water
  0.055. The second (10M, fp32, per-head ratio, tanh means, desync, reward scale 0.1) had a healthy update
  (KL 0.003-0.007, clipfrac 0.02-0.1) and still got worse: mean |live reward| 0.56 in the first epochs, 0.76 in the
  last, full episodes -0.53 to -1.09. Both fed the network the raw observation (most fields mean 0 / var 1).
- Not repeated. The scaled canonical observation was only ever trained as a residual over pid_water. `ppo_canon` is
  that observation with the policy's own actions (`env.canonical=1`, `env.residual=0`), the same PPO settings as the
  ini, fp32 training solve, 960 agents, 20M steps. Evaluation stays on the fp64 binary.

### 2026-10-16 20:49 IST submitted r2 checkpoint, smaller residual at evaluation only
- Same weights (`r2_dev_cguard/ckpt/0000000019906560.bin`), fp64, cold guard, no retraining
  (`results/eval/r2_s0{25,10}_*.json`). pid_water is 0.0549 / 0.0569 / 0.0492 on dev / gen-val / gen-val-s.
  Scale 0.5 (the submitted policy) was dev 0.0446, gen-val 0.0687, gen-val-s 0.0522.
- Scale 0.25: dev 0.0439 (19/30 better, CI excludes 0), gen-val 0.0660 (11/30, still worse, CI excludes 0),
  gen-val-s 0.0503 (14/30, CI includes 0).
- Scale 0.10: dev 0.0438 (21/30 better, CI excludes 0), gen-val 0.0641 (15/30, still worse, CI excludes 0),
  gen-val-s 0.0492 (18/30, tied with the base, CI includes 0).
- FINDING: turning the trained residual down at evaluation keeps the dev win and brings the smaller held-out
  buildings to a tie. The large held-out buildings stay worse than pid_water at every scale tried. This is not a
  new training run.

### 2026-10-17 08:29 IST plain PPO on the scaled observation (`ppo_canon`): about 10x worse than pid_water
- `results/runs/ppo_canon/`, fp32 training solve, 960 agents, 19,906,560 steps in 23 min 34 s. The trainer again
  exited after the last epoch on the undeclared `episode_score` tag; the checkpoint `0000000019906560.bin` was already
  written. Deterministic fp64 eval (`results/eval/ppo_canon_*.json`), absolute actions, residual off, paired vs pid_water:
  dev 0.651 vs 0.0549 (0/30 better, mean diff +0.596);
  gen-val 0.516 vs 0.0569 (0/30, +0.459);
  gen-val-s 0.594 vs 0.0492 (0/30, +0.545).
  Every confidence interval excludes 0. Training already showed it: the last full episodes had a per-scenario score
  near 0.89, comfort deviation about 5 K.
- FINDING: giving the fixed PPO the scaled canonical observation, and letting it own the actions, does not learn a
  controller. It is an order of magnitude worse than pid_water on the buildings it trained on and on both held-out sets.

### 2026-10-17 12:20 IST r7 pure-RL chain started (`tools/run_pure_chain.sh`)
- Eleven segments, `pool2_s0` .. `pool2_s10`, 4M steps each (the per-segment budget used for the r4 rotation; the
  600k `pool2_s2` run was a smoke and is not this run). Canonical layout, absolute actions, residual off.
  Learning signal: per-scenario reward normalization (`env.rew_norm=1`), advantage normalization, action hold of 4
  steps. Checkpoints carry optimizer momentum. Log-std anneals from -1 to -3 across the eleven segments.
  gamma 0.999, gae_lambda 0.98, learning rate 1e-4 with no anneal. TensorBoard is archived per segment
  (`results/runs/r7_sNN/tb`); the live log is `results/runs/r7_live`.
- Environments per scenario: 16. The `pool2_s2` smoke at 32 (640 agents) died in `cudaGraphInstantiate`, the
  out-of-memory signature. 16 keeps a 5-weather group at 80 envs, under the layout that overflowed s2 and s10.
- Speed: `SB_F32=1` on the 3090 (the fast training solve; `tools/run_train.sh` now rebuilds when the binary's
  solve variant does not match, the same check as `tools/train.sh`). The chain is one sequence, so it stays on
  the faster card. The 3060 scores each finished segment on gen-val with the fp64 binary while the next segment trains.

### 2026-10-17 12:28 IST r7 first segment aborted on the action limit, then relaunched without the three buildings that cannot load
- `r7_s00` on `pool2_s0` (ENVS=16, fp32 solve) exited during setup. The log is
  `scenario needs 659 continuous / 30 discrete actions, the env pads to 256 / 16`. That is building_30028
  (659 continuous, 30 discrete, 2203 observation values). building_30025 in the same set (186 / 9 / 658) loaded.
  `SB_CONT_BIG` raises the continuous limit to 320 only, and the trainer build does not pass it. Two more buildings
  in the eleven sets are over a hard limit: building_30021 (308 continuous, 1060 observations) and building_0500
  (278 continuous; the documented `SB_CONT_BIG` case). The port's limits are 256 continuous, 16 discrete, 1024
  observations (`complete_port/README.md`, "New buildings").
- The second attempt kept only building_30025 in that first set. It loaded, then the step fell off the
  convection path: `conv launch: operation failed due to a previous error during capture (csmem=322240
  max_room=40280)`. The room has 40 280 cells. `convect2g_k` still keeps a `max_room` array of unsigned
  shorts in shared memory, and that footprint is 102 464 bytes, over this card's opt-in shared-memory
  limit, so the step used `convect_k`, which needs 322 KB and cannot launch. The other three buildings
  already excluded have rooms that fit; 30025 is the only one in the pool that does not.
- The chain that is running drops those four buildings and is ten segments: `pool2_s1`, `pool2_s2`,
  `r7_s3` (30021 omitted), `pool2_s4` .. `pool2_s9`, `r7_s10` (0500 omitted). Same 4M steps, hold, gamma
  and the -1 to -3 log-std anneal across these ten. ENVS=16. The heaviest remaining set is about
  0.16 Gcell·env.

### 2026-10-17 12:55 IST r7 env count steps down on failure, and the 3090 joins evaluation after training
- Segment 0 (`pool2_s1`, 16 environments, about 2.7M of 4M steps when this was decided) is left to finish.
  Restarting it would throw away that work, and 32 environments on this set is the size that already failed
  graph capture on the similar `pool2_s2`.
- From segment 1 on, `tools/run_pure_chain.sh` tries 32, then 30, then 28, and so on down to 8. A count is
  skipped when the agent total is not divisible by 8 (minibatch 2048 / horizon 256); the trainer rejects
  that before any step. For a 15-scenario set that means 32, then 24, then 16. For a 20-scenario set,
  30 and 28 are legal and are tried. Each segment starts the search again at 32, so a smaller later set
  is not stuck at the count a larger set needed. A graph-capture, out-of-memory, or convection-launch
  failure is what steps down. A building past the action or observation limit still stops the chain.
  (r8 starts that search at 20 and pins training to GPU 0; see the 19:20 entry.)
- The scoring is not a separate program. `tools/eval_policy.sh` skips the global GPU queue when `SB_GPU`
  is set (the 3060 score can run while training holds the 3090) and `SB_BIN=puffer_f64` selects the exact
  binary. `tools/run_pure_chain.sh` calls it: gen-val of each finished segment on GPU 1 during training,
  then dev and gen-val-s together on GPU 0 and GPU 1 once the chain is done.

### 2026-10-17 13:20 IST r7 resumed from the finished first segment
- Segment 0 (`pool2_s1`, 16 environments) reached 4.0M steps in 29 min 14 s. The final save
  `0000000003993600.bin` is empty (the process was stopped while that file was created). The previous
  save is intact, 2.9 MB weights and momentum: `r7_s00/ckpt/0000000003686400.bin` (3.69M steps).
  The copies under `results/runs` were also empty; the real files were still in the trainer checkout.
  The chain now ignores a zero-length checkpoint. Segment 0 is not retrained.
- The restart continues at segment 1 with that checkpoint. Every remaining segment tries 32 environments
  per scenario, then 30, 28, and so on. The 3060 scores segment 0 on gen-val while segment 1 trains.

### 2026-10-17 16:15 IST r7 finished: every later segment fit at 32 environments, scores stay far from pid_water
- Nine segments after the first, each about 4M steps, all at 32 environments per scenario (no step-down).
  Wall clock 13:26 to 16:15, including the scores. Final weights
  `results/runs/r7_s09/ckpt/0000000003956736.bin`.
- Deterministic fp64, absolute actions, paired vs pid_water (0.0549 / 0.0569 / 0.0492):
  dev 0.568 (0/30 better, mean diff +0.513, CI excludes 0);
  gen-val of the final checkpoint 0.689 (0/30, +0.632);
  gen-val-s 0.553 (0/30, +0.504).
  gen-val along the chain: 0.798, 1.031, 0.823, 0.814, 0.731, 0.712, 0.707, 0.713, 0.708, 0.689.
- FINDING: the pure-RL mechanics (reward norm, advantage norm, action hold, log-std anneal, rotation with
  momentum) on the scaled canonical observation do not learn a controller. The held-out score improved
  through the chain and is still about ten times pid_water on every split.

### 2026-10-17 19:20 IST honest split + trainer fix bundle: r8 prepared (six defects fixed, not tuned)
- Post-mortem extended to `research/new_renaissance/ppo_analysis/ppo_ceiling_verdict.md` §9-§10 and
  `research/DECISIONS.md` D10.
- **New defect found while building the bundle (all r7 runs affected):** `adv_stats_kernel` divided the
  Welford/Chan **mean** by the count a second time, so `adv_norm=1` reported `true_mean/n` and the update
  used `(A - mean/n)*inv_std ~= A*inv_std` - the advantage mean was never removed. Proof: lr=0 smoke,
  reported `norm/adv_mean` -0.0001232 = -0.2522/2048, `loss/policy` 0.7536 = 2.989 x 0.2522. Fixed
  (`out[0] = s_mu[0]`); with lr=0 both normalization modes now agree exactly on the single-scenario set
  and both policy losses are ~1e-8. Mode 2 was and is bit-identical (it never used that kernel).
  Mechanism D's measured dynamics (clipfrac 0.38 -> 0.48, KL 0.35 -> 0.54, value loss 1845) are therefore
  partly this bug, so r7 is not clean evidence about PPO's own ceiling.
- **Honest split** (`tools/build_split.py`, gated by `tools/check_split.py --deep`, clean): train = 36
  buildings in `sets/tr36_s1..s6` (171 scenarios, size-balanced), eval = 9 buildings in `sets/eval9`
  (45 scenarios), split for two-GPU scoring into `eval9a` (5 buildings) + `eval9b` (4) and merged by
  `tools/merge_eval_json.py`; `sets/evalcore3` (20003/20009/25014, 15 scenarios) monitors each segment.
  Sealed range 30000-30019; the old `pool2_*` sets (hardlinked eval buildings) are no longer used.
- **Fix bundle** (all previously verified live): canonical action mask written on the GPU path (measured
  per-building active heads 71-187 of 288, entropy ~62-65 not ~450); `train.rew_clamp=0` +
  `live/rew_z_gt{1,3,5}_frac` (measured: 0.91% of steps beyond 1 sigma, none beyond 3 - the clamp hid
  the worst ~1%, which is what the score is made of); `policy.logstd_anneal=0 init_logstd=-1` (constant
  sigma 0.37); `train.shuffle=1`; whole-rollout advantage standardization `adv_norm=2` (prepass mirrors
  the live-V GAE); EV/moment diagnostics.
- **Measured envelope** (tr36 on the 3090): 600 agents 22.9/24 GB at ~3.9K SPS, 432 agents 21.1/24 GB
  at ~2.6K SPS, 648 (27-scenario set) and 960 agents OOM. The chain starts at 20 environments per
  scenario and steps down per set; a 27-scenario set can only use 8/16/24/32 (agents must be divisible
  by 8), so it runs at 16.
- **Runner** `tools/run_pure_chain.sh`: 6 segments x 8M steps, chained checkpoints + momentum, split
  proof before every segment, training pinned to GPU 0 while GPU 1 scores each finished segment twice
  (held-out core3 and the segment's own training set - in-sample, so a flat held-out curve next to a
  rising in-sample one names the r7 failure directly), then `eval9a` (GPU 0) + `eval9b` (GPU 1) together
  and merged. Scorer is `puffer_f64` (the default fp64-solve build kept under its own name).

### 2026-10-18 07:10 IST r8 interrupted by a host power event at 22:51, resumed from segment 5
- The host went down mid-run: the previous boot ends 22:51:52 IST and every r8 log stops within the same
  minute (s05 train.log 22:51, s04 core3 eval placeholder 22:51). No run log shows a CUDA, docker or
  trainer error first; nothing was killed by a resource failure. The machine came back 07:05:41 IST.
- Salvaged: s04 finished at 22:44:40 with its checkpoint **and** optimizer momentum sidecar
  (`results/runs/r8_s04/ckpt/0000000007962624.bin` + `.mb`); chain file entries 0-4; the s00-s03 scores.
  Lost: ~7 min of s05 training (no checkpoint was written, so the segment restarts at the full 8M steps)
  and the in-flight s04 scores (the core3 JSON was still the 91-byte placeholder; in-sample never ran).
- Resume: `follow_evals` has no skip-already-scored logic (it starts at chain line 0 on every launch), so
  a plain relaunch would have re-scored s00-s03 on GPU 1 first (~2 h of redundant evals). The chain file
  was backed up (`results/runs/r8.chain.pre_reboot.bak`, 5 lines) and trimmed to the single pending
  segment-4 line, then relaunched: `FIRST=5 START=<r8_s04 ckpt> tools/run_pure_chain.sh r8 8000000
  "tr36_s1 .. tr36_s6"`. That scores s04's held-out + in-sample on GPU 1 while s05 trains on GPU 0, then
  s05's scores, then `eval9a` + `eval9b` and the merge with no other repeated work.
- Pre-flight after the reboot: docker GPU access works, `puffer_f64` is still the fp64-solve build (no
  cuSOLVER fp32 symbol), the four s04 periodic checkpoints all carry `.mb`. Started 07:09:11: GPU 0 at
  20.9/24 GB training s05 (tr36_s6, 30 scenarios, 600 agents), GPU 1 scoring s04's core3.
- Dead-run scalars (segments 0-4, held-out core3 every time): 0.7924, 1.0152, 0.9400, 0.7154 (s04 eval
  lost). s04 end-of-segment: entropy 42.7, clipfrac 0.131, EV 0.168, `norm/adv_mean` -0.671, z>1 sigma
  14.0%, worst_dev_K 18.4, value loss 7066. Held-out is recovering after the s01 hump, but EV was -0.072
  at s03 and value loss is O(10^3-10^4): the critic is the open concern, not the exploration schedule.

### 2026-10-18 09:30 IST r9a launched: the r8 diagnosis as a fix pack (fresh, canonical absolute)
- r8 closed at 48M steps flat (DECISIONS D11, post-mortem §11): held-out core3 0.7154-1.0152 vs bar
  0.0508, final `eval9` 0.768986 vs 0.0555 = **13.86x** (`results/eval/r8_final_eval9.json`).
- Root cause, all measured (post-mortem §12 for the per-segment table):
  1. the learning reward's z-score scale ramps inside every segment - `norm/reward_inv_std` starts at
     0.105 and climbs **x16.7 / x8.2 / x14.6 / x7.5 / x11.3 / x9.8** (s00-s05) while
     `norm/reward_mean` drifts from -0.01..-0.03 to -0.58..-1.31: the critic fits a moving target
     (s02 value loss 252 at 0.1M -> 14,240 at 1.5M is partly the target's own variance growing);
  2. `gamma=0.999` on 8640-step episodes (~14-day returns) makes that target large and high-variance;
  3. the policy starts chaotic: in absolute canonical mode run bits decoded as `on iff p > 0`, so
     zero-mean heads with sigma 0.37 flip every AHU/pump on a coin toss - the plant churns (energy
     3-6x the baselines) and the gradient has nothing stable to follow. Measured EV <= 0.222 in every
     segment (0.198/0.208/0.212/0.194/0.222/0.116).
- r9a = one fresh 8M-step segment on `tr36_s6` (GPU 0) with: reward-norm EMA initialised at the data's
  own scale (`g_rn_var` 100 -> 1.0), `gamma` 0.99, `gae_lambda` 0.95, `vf_coef` 1.0, sigma 0.37 -> 0.20
  constant (`init_logstd=-1.6`, `logstd_anneal=0`), run bits decoded as `on iff p > -0.5` (neutral
  output = running device; comment in `gpu/sb_gpu_base.inc`, header line 27 updated), and
  `policy.zero_init_actions=1` so the deterministic start is "valves mid-range, equipment running"
  (the decoder *weight* rows are zeroed; the bias rows stay ~+-0.06). Deploy + rebuild of both
  binaries verified (`puffer` and `puffer_f64` both the default fp64-solve build).
- Zero-init reference evals on GPU 1 (`evalcore3`, `tr36_s6`) give the start level on the same sets the
  segment is scored on.
- Pre-registered fingerprints (D12), live at 0.5M steps: `inv_std` 1.14 (**no ramp** ✓), value loss 64
  (O(10-100) ✓, not O(10^3-10^4)), EV 0.01 (3 updates in, too early), comfort_dev 5.6 K,
  out_of_band 0.92, energy 0.253 (the mid-valve start is deliberately wasteful: reheat 50% open,
  equipment always on - the policy has to learn to close it).
- Measured at 2.5M steps: EV 0.129 vs r8_s00's 0.034 at the same step (**4x**), `norm/reward_inv_std`
  1.814 (1.048 -> 1.814 over the segment: no x7.5-x16.7 ramp, but it is still climbing - the true
  fixed point is `1/sqrt(0.28)` ~ **1.89**, i.e. the r9a run started at 1.0 and spent the segment
  converging; the next segments start at the fixed point), value loss 105-200 (r8: 10^3-10^4),
  `live/energy_cost_norm` 0.253 -> 0.195 (**-23%**, the expected reheat-closing gradient), out_of_band
  0.92 -> 0.84, comfort_dev 5.6 -> 4.7-5.2 K. The start is no longer chaotic and the critic is no
  longer chasing a moving target.
- **Zero-init reference (`results/eval/r9a_init_core3.json`): core3 = 0.327246** on `evalcore3` vs the
  0.0508 bar (6.44x) - and 2.2-3.1x *better than everything r8's PPO ever produced* (0.7154-1.0152).
  The chaotic-start diagnosis is confirmed quantitatively: a policy with zeroed decoder weights and
  running equipment beats 48M steps of r8 training. In-sample `tr36_s6` reference started in parallel
  on GPU 1 (finishes ~09:57).
- Init fix for the following segments (staged in `gpu/smart_buildings.cu`, worktree source updated):
  `g_rn_var` 1.0 -> **0.28** (inv_std 1.89). Training-only statistic - the evaluator uses raw rewards,
  so `puffer_f64` scores are unaffected; verified live by the next run's first TB point at
  `norm/reward_inv_std` ~1.89.
- At 3.07M steps the runs directory holds a **rolling periodic checkpoint**; it was snapshotted to
  `/tmp/opencode/r9a_early.bin` (md5-verified) and queued as an early held-out read on GPU 1 (behind
  the zero-init reference, both score `evalcore3`) - the first *score* read of the pack's learning
  curve, ~2.5 h before the end of the full chain.
- The chain's own tail (`follow_evals`, identifies s00's core3 + in-sample on GPU 0) was cut to keep
  GPU 0 training: the per-segment held-out scores are now produced on GPU 1 by
  `research/tools/rl_loop.sh` (train on GPU 0, score the finished checkpoint while the next segment
  trains, one eval JSON per segment). r9a's own core3 + `eval9` are queued on GPU 1 behind the early
  read, so the pack's held-out verdict costs no training throughput.
- Continuation run **r9b** (fired from `/tmp/opencode/r9_post.sh` when r9a's checkpoint is archived):
  rebuild `./puffer` with the 0.28 init inside `puffertank:5.0`, then 5 continuation segments
  `tr36_s4, s1, s2, s3, s5` (the sets r9a/r8 did not use, 30/27-scenario mixes = 600/432 agents) with
  `--base.load_model_path` chaining the checkpoint + `.mb` momentum. Every segment starts from the
  measured init scale, so the critic never sees a reward-scale jump; the held-out curve is the
  decision metric (bar 0.0508). **Not launched** - the early read below reversed the decision
  (D14); the rebuild itself was still run (it verified the 0.217 init), only the 5 segments were cut.
- r9a endpoint: segment finished at 8M steps in 29m13s (~4.6K steps/s, 600 agents, 30 scenarios).
  Final scalars: EV 0.2085, value loss 1132, `adv_abs_mean` 8.14, clipfrac 0.1204, `norm/adv_inv_std`
  0.0698, `live/worst_dev_K` 16.6 K, `live/rew_z_gt1_frac` 0.19. Checkpoints archived to
  `results/runs/r9a_s00/ckpt/` (3.07M/6.14M/7.99M); the TB was copied to `results/runs/r9a_s00/tb/`
  by hand (`run_pure_chain.sh` does not archive it; `run_train.sh` does).
- **Held-out early read, 3.07M-step checkpoint (`results/eval/r9a_early_core3.json`): core3 =
  0.532577 vs the zero-init 0.327246 - the pack's learning made the scored metric 63% worse.**
  Component means over the 15 held-out scenarios (`0.6*p - 0.2*e - 0.2*c` reproduces the score):

  | term (weight) | zero-init | 3.07M | delta |
  |---|---|---|---|
  | productivity/comfort (0.6) | -0.3333 | -0.7487 | **-0.4154 (2.2x worse)** |
  | energy_cost (0.2) | 0.2815 | 0.2003 | -0.0812 (better) |
  | carbon (0.2) | 0.3549 | 0.2166 | -0.1383 (better) |

  Per scenario the regression is climate-dependent: `atlanta_7` -0.3445 -> -0.2925 (better),
  `denver_1` -0.2004 -> -0.2529, `fairbanks_3` -0.2388 -> **-0.6358** (cold climate: shutting
  conditioning down freezes the zones).
- Diagnosis (measured, not guessed): this is a **credit-horizon failure**, not a chaos failure. The
  mechanics the r9a pack fixed stayed fixed (no `inv_std` ramp - 1.048 -> 2.265 -> 2.147, value loss
  105-592, EV 0.13-0.21), but with `gamma=0.99`, `lambda=0.95` the GAE credit for a reward 144 steps
  (12 h) later is `(0.99*0.95)^144 = 1.5e-4`: the overnight setback/plant-shutdown saves energy
  *immediately* (credit ~1) and its comfort cost arrives at the next occupancy period (credit ~0), so
  the shutdown is the *optimal* policy under the training objective and catastrophic under the
  scored one (0.6-weight comfort traded for 0.2-weight energy/carbon - the exact measured shape).
  The live comfort loss (`live/comfort_dev_K` 3.6 -> 6.4 K) is partly an unoccupied-period artifact
  (productivity is occupancy-zeroed), but the training-regime `episode_score_scen_mean` (0.55-0.63 on
  `tr36_s6`) agrees with the eval - the regression is real in both regimes, and the scored 0.6-weight
  term is what the policy sold.
- Operational fix that cost two eval rounds: `tools/eval_policy.sh` runs the eval in a container that
  mounts `complete_port/results` as **`/res`** - the model argument must be the *container* path
  (`/res/runs/<name>/ckpt/<file>.bin`). A host path aborts with
  `puf_load_weights_into ... failed to open weights for reading` and leaves a `complete:false` JSON.
  `research/tools/rl_loop.sh` had the same bug (fixed; see D14).
- The 8M endpoint eval (`results/eval/r9a_s00_core3.json`) completes the r9a curve: **core3
  0.327246 (zero-init) -> 0.532577 (3.07M) -> 1.110974 (8M)**, with the components
  productivity -0.333 -> -0.749 -> **-1.792**, energy 0.281 -> 0.200 -> 0.130, carbon
  0.355 -> 0.217 -> 0.049. The endgame is "stop conditioning the building" (energy and carbon costs
  near zero, comfort collapsed), which is *not* the score optimum (0.6*1.79 >> 0.2*0.13) but is the
  optimum under the training objective with `(gamma*lambda)^144 = 1.5e-4`.
- `eval9` cannot run on the 3060 (12 GB): 45 scenarios at 1 env/scenario exceed 12 GB
  (`cuda err out of memory` after group 6/9; the docker died silently, `eval_policy.sh`'s grep
  filter hid it and its `set -e` does not see inside the pipe). The shipped `eval9a` (25) +
  `eval9b` (20) split sets are the workaround; the pooled score is the scenario-count-weighted mean.
- r9d prep (the next lever; implemented and built while r9c ran - see D15): a **train-only
  potential-based comfort shaping** in the env (`env.comfort_shape=K`, `env.shape_gamma=g`). The
  learning reward handed to the trainer gains `F_t = g*Phi(s_t) - Phi(s_{t-1})` with
  `Phi(s) = -K * mean_z shortfall_z(s)` - the SAME per-zone deviation curve the official comfort
  term uses, *without* the occupancy weight. Rationale: the two measured failures are the two ends
  of one trade-off - with `gamma*lambda = 0.94` the 12 h-later comfort cost is numerically absent
  (r9a), with `gamma*lambda ~ 1` the estimator's variance swamps the signal (r9c, EV 0.01-0.03).
  The shaping moves the credit to the step that causes it **without changing the objective**:
  summed over an episode it telescopes to `g^T*Phi_T - Phi_0`, a constant for any policy
  (`g^T ~ 1e-4` at 8640 steps), so the argmax is unchanged - only the credit assignment moves.
  It also lets the estimator go back to the low-variance regime (`gamma 0.99, lambda 0.95`) that
  r9a measured stable.
  - Placement: `reward_k` accumulates the occupancy-free shortfall mean and writes the bonus into a
    new per-env `d_shape` buffer (`d_phi` holds the previous step's potential; it starts NaN and
    `kv == 1` skips each episode's first transition, so no boundary bonus ever leaks);
    `finalize_k` adds it to `rewards[]` - the buffer the trainer reads. `rew_out`, `rp[]`,
    `ep_ret`, `ep_dret`, the live metrics and the eval dump keep the official reward, so the scored
    metric is untouched by construction.
  - Flags: `env.comfort_shape` (K, default 0 = off, so the gate and every existing path are
    unchanged) and `env.shape_gamma` (default 0.99; set to the trainer's gamma). Both keys are
    declared in `config/smart_buildings.ini` (PufferLib binds `--env.*` against that base config;
    without the keys a run dies with `missing key [env] comfort_shape`).
  - K = 2.0 is a calibration estimate, not a measurement. In the *linear* region of the official
    curve (`dev < 5 K`, where a night drift starts) `Phi` charges `K * drift_K/5` per K-step, i.e.
    ~`K*0.06` per step for the measured ~0.3 K/step night drift, against the ~0.02-0.06/step the
    shutdown buys back in energy+carbon (0.2 * the measured 0.257 -> 0.130 / 0.355 -> 0.049 levels).
    K = 1 is a wash there (a coin flip at the very transition that decides the run); K = 2 charges
    the drift 2-6x the saving, and in the quadratic region (`dev > 5 K`, the r9a/r9c endgame) it
    charges ~`2*2*(dev-5)/25*0.3` = 0.14-0.7/step. In-band (`sbar = 0`) `F_t` is exactly 0, so the
    energy/carbon optimization inside the band is untouched - the shaping only acts on the way out
    of and back into the band, which is the behaviour that was missing.
  - Regenerated `gpu/sb_group_state.inc` (`d_phi`, `d_shape` are per-group buffers; the two scalars
    are batch-wide and excluded in `tools/gen_group_state.py`). The SB_TEST gate build and the
    trainer build both compile clean.
- r9c_s00 finished (10:38, 8M steps) and its held-out core3 read (10:45) closes the second branch:
  **core3 0.788374** (zero-init 0.327246, r9a@3M 0.532577, r9a@8M 1.110974). Components: productivity
  **-1.1881**, energy 0.2622, carbon 0.1152. `gamma = lambda = 0.999` did not fix the trade - the
  policy still sold the 0.6-weight comfort term for the 0.2-weight pair, and unlike r9a it barely
  bought anything (energy 0.262 vs the zero-init's 0.281): the same basin, entered by a different
  route. Live fingerprint at 6M: productivity -0.685, energy 0.230, carbon 0.151, EV 0.042, SPS 4441
  (vs the zero-init start p -0.560 / e 0.257 / c 0.289). So neither end of the `gamma*lambda`
  trade-off works - with 0.94 the 12 h credit is `1.5e-4`, with ~1 the estimator cannot resolve it
  (EV <= 0.04) - which is the measured case for moving the credit itself instead of the horizon
  (D15). D14's rule fires: r9c is not continued (s01/s02 dropped).
- Required verification for the r9d change (`gpu/*`, `tools/gen_group_state.py`, `config/*.ini`):
  **`tools/verify_all.sh --trainer` = ALL CHECKS PASSED** (54 PASS, 0 FAIL, 10:55-11:00, GPU 0 while
  the r9c core3 read ran on GPU 1), including the trainer rebuild from patch + complete_port and the
  trainer smoke tests, and `multi-scenario: sb_group_state.inc matches the sources`.
- r9d launched 10:59:31 (fresh, `zero_init_actions=1`, tr36_s6 first, then tr36_s4/tr36_s1):
  r9a's recipe (gamma 0.99, lambda 0.95, vf_coef 1.0, sigma 0.2, lr 1e-4, action_repeat 4,
  adv_norm 2, shuffle 1, rew_clamp 0, checkpoint_optim 1, reward-norm init 0.217) **+ the shaping**
  (`--env.comfort_shape=2.0 --env.shape_gamma=0.99`).
- Operational hazard (cost one silent eval): `tools/verify_all.sh --trainer` **deletes `./puffer_f64`**
  from the shared worktree. Its trainer section starts with
  `( cd "$PL" && git checkout -q -- . && git clean -fdq )` (verify_all.sh line 91): `puffer_f64` is
  untracked and not git-ignored (`git check-ignore` says so; the dry run also lists `puffer`,
  `config/smart_buildings.ini`, `ocean/smart_buildings/`, `src/tblog.h`), so `git clean -fdq` unlinks
  it, and the next line rebuilds only `./puffer`. `logs/` and `checkpoints/` *are* git-ignored, so the
  live TB/rolling checkpoints survive that step (the TB wipe is the separate container-side
  `rm -rf logs/smart_buildings/tb`). Any `SB_BIN=puffer_f64` eval started afterwards dies in seconds
  with `bash: ./puffer_f64: No such file or directory` *inside* the container, which
  `eval_policy.sh`'s grep filter hides and its `set -e` cannot see through the pipe - the only visible
  symptom is a missing JSON plus the summarize traceback (occurred at ~10:54; the 10:38 r9c eval had
  already exec'd its copy and completed). No copy survives anywhere (`find /data`, other mounts, no
  process holds the inode). `tools/run_pure_chain.sh` guards against this (copies `./puffer` to
  `./puffer_f64` when missing, and refuses an fp32-solve copy); `research/tools/rl_loop.sh` does not
  (restored by hand: `cp puffer puffer_f64`, same default fp64-solve build, verified by the absent
  `EEv5LuDevPf` symbol and by re-running the r9c core3 with the replacement - identical JSON).
- r9d_s00 finished (11:31, 8M steps) and its held-out core3 read (11:37) **fails the pre-registered
  kill rule** (D14: any read >= the zero-init 0.327246 ends the chain): **core3 1.058289** (mean
  reward -1.058289; productivity **-1.6872**, energy 0.1485, carbon 0.0814). The ladder on the same
  held-out set: zero-init 0.327246 -> r9a@3M 0.532577 -> r9d@3.07M 0.744992 (mid read) -> r9c@8M
  0.788374 -> **r9d@8M 1.058289** -> r9a@8M 1.110974. The shaping is inert (A/B-identical objective,
  gate bit-exact) and insufficient: at the mid read it was *worse* than r9a at the same step
  (0.745 vs 0.533); at 8M it lands in the same basin with marginally less comfort damage (p -1.687
  vs -1.792) but still ~3x the zero-init cost. The K = 2 charge (~0.02-0.07/step for the measured
  night drift) versus the ~0.044/step the shutdown buys back is a wash, as the pre-launch
  calibration predicted - credit assignment was not the binding constraint.
  Chain **stopped 11:38**: the `rl_loop.sh r9d` driver and the s01 (tr36_s4) trainer killed one
  minute into s01 (started 11:30 with the same flags; no s01 read taken, the rule fired on s00).
  GPU 0 free. The r9 line (a: reward-norm scale fix + run-bit threshold + zero-init actions,
  c: gamma = lambda 0.999, d: potential shaping) is now a **measured failure in all three variants**:
  every run makes its own training objective worse than the policy it started from.

### 2026-10-18 12:40 IST r10 started: one scenario, PufferLib PPO, no baseline in the action
- Standing rule written into `AGENTS.md`, `complete_port/README.md`, `research/DECISIONS.md` (D16) and
  `Experiments.md`: the policy learns only by acting. No residual on pid_water, no action head
  initialized as that controller, no comfort guard, no comfort shaping. The public target is 0.00603.
- r10 is PufferLib 5.0 PPO on `building_train_1` / `fairbanks_1` only, 256 environments, absolute
  canonical actions. Learning rate 0.0004 held constant, entropy coefficient 0.05 held constant,
  discount 0.99, GAE 0.95, clip 0.2, value coefficient 0.5, four epochs, horizon 256, minibatch 2048,
  advantages standardized over the rollout, reward normalization off, reward scale 1, reward clamp off,
  action hold 1. 50M steps requested. fp32 solve on the 3090 (4.8 GB). The layout line is one group,
  256 envs, 8 per lane, value mode 0 (one weather). 97 of 288 heads are active.
- First minute of the trainer clock: 1.4M steps in 1 min 29 s. A paused viewer container (`sb_watch`,
  an r9d checkpoint) was holding the 3090 and was stopped so this run could take the card.
- The 3060 scores each checkpoint on fairbanks_1 and on atlanta_2 of the same building with the fp64
  binary. After the run, gen-val is scored on the 3090 and gen-val-s on the 3060. TensorBoard is the
  server already on http://127.0.0.1:6006 (`results/runs/r10`). `tools/run_train.sh` copies the event
  file and the checkpoints out while the run is going, so that server updates during training.

### 2026-10-18 14:05 IST r10 finished: the score curve is the comfort term
- 49.9M steps in 56 min 1 s. Final weights `results/runs/r10/ckpt/0000000049938432.bin`.
  gen-val 0.261153 (pid_water 0.0569). gen-val-s 0.247430 (pid_water 0.0492).
- The reward is 0.6 × productivity − 0.2 × energy − 0.2 × carbon. On the training episodes
  0.6 × productivity matches `env/score` to within 0.05 at 2.2M, 5M, 15M, 20M, 27M, 30M, 40M
  and 50M. Energy's contribution stays near −0.02. The productivity plot and the score plot
  are the same curve.
- The comfort improvement at 30M (mean deviation 31.8 K → 21.7 K) is the gas term rising from
  12.6 to 68.7 thousand kWh per episode. By 50M deviation is back to 26.6 K and gas to 44.2.
  Out-of-band fraction stayed 0.989–0.993. Worst zone stayed 52–57 K.
- Deterministic fairbanks score 8.10 at 1.3M and 2.47 at 47M, and it was already 2.58 at 30M.
  The TensorBoard decline after 30M is the sampled episodes: entropy went 61 → 342.
  atlanta_2 of the same building scored 0.095. On gen-val the mean productivity is −0.389
  against pid_water's −0.072, and energy is 0.089 against 0.042. The policy spends more
  energy and is less comfortable on buildings it did not train on.

### 2026-10-18 14:51 IST r11 started: entropy cosine from 0.01 to 0.001
- Same scenario and PPO knobs as r10 (`building_train_1` / `fairbanks_1`, 256 envs, absolute
  canonical actions, learning rate 0.0004 constant). The only change is the entropy bonus:
  `ent_coef=0.01`, `anneal_ent_coef=1`, `min_ent_coef_ratio=0.1` (cosine down to 0.001).
  30M steps. fp32 solve, value mode 0, 97/288 heads, 4.5/24 GB. First frame: 65.5K steps,
  uptime 7 s, `loss/entropy` 51.9 (inside the translated healthy band of about 40–120).
- No gen-val. The 3060 scores each checkpoint on fairbanks_1 and atlanta_2 with `puffer_f64`.
  The follower stops the trainer if three logged points past 5M steps have summed entropy
  above 180 with KL under 0.001 and clip fraction under 0.01. TensorBoard:
  http://127.0.0.1:6006 (`results/runs`, run `r11`).

### 2026-10-18 15:51 IST r11 finished: entropy stayed healthy, the mean did not move
- 29.9M steps in 33 min 42 s. Final weights `results/runs/r11/ckpt/0000000029949952.bin`.
  Summed entropy stayed in 52–100 (final 98.3, log-std −0.41). End KL 0.003,
  clip fraction 0.033, policy loss −0.037, explained variance 0.998. The freeze
  rule did not fire.
- Deterministic fp64 fairbanks 7.66 at 1.3M and 8.12 at 29.9M. Energy cost on
  that episode fell from 0.060 to 0.018; productivity stayed near −13.5.
  atlanta_2 0.225 → 0.210 (best 0.141 at 15.7M). r10's mean reached 2.47 on
  fairbanks and 0.095 on atlanta.
- The sampled training episodes did heat: score −8.0 → −4.0, gas 13 → 60
  thousand kWh, deviation 32 K → 24 K, out-of-band fraction still 0.987.
  The Gaussian samples found the heating. The mean that evaluation plays
  settled further into the low-energy action. A start of 0.01 was too small
  to move that mean. r10's constant 0.05 moved it, then the noise ran away.

### 2026-10-18 16:06 IST r12 started: r10's 30.1M weights, entropy bonus 0.001
- Loads `r10/ckpt/0000000030146560.bin` (the checkpoint where deterministic
  fairbanks had just reached 2.58). `ent_coef=0.001` constant, `anneal_ent_coef=0`,
  `checkpoint_optim=0` so the 0.05-run momentum is not loaded. Same scenario,
  learning rate 0.0004, absolute actions. 20M further steps. fp32, value mode 0.
- The load took: first frame summed entropy 235.4, not the fresh-init ~52.
  3090 at 4.9/24 GB. No gen-val. The 3060 scores fairbanks_1 and atlanta_2.
  TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r12`).

### 2026-10-18 16:34 IST r12 finished: fairbanks 2.58 → 1.25, entropy stayed near 222
- 20.0M further steps in 22 min 27 s. Final weights
  `results/runs/r12/ckpt/0000000019988480.bin`.
- Deterministic fairbanks 2.58 → 1.25. r10, continuing with ent_coef 0.05,
  went 2.58 → 2.12 → 2.47 over the same stretch. Atlanta 0.110 → 0.096,
  matching r10's 0.095. Training-episode deviation 19 K → 5.2 K, out-of-band
  fraction 0.982 → 0.756, worst zone 52 K → 34 K. Gas stayed near 70 thousand
  kWh. Final fairbanks energy cost 0.061 against 0.133 on the loaded checkpoint.
- Summed entropy 235 → 222. The standard deviation did not return to the
  healthy band. KL was near 0 for long stretches while the score kept falling.
  The fairbanks score was still falling at the last checkpoint.

### 2026-10-18 16:51 IST r13 started: r12's mean, log-std pinned at −1.6
- Loads `r12/ckpt/0000000019988480.bin` (fairbanks 1.25, summed entropy 222).
  `logstd_anneal=1`, `init_logstd=-1.6`, `logstd_final=-1.6`, so the standard
  deviation is held at 0.2. `ent_coef=0.001` constant, `checkpoint_optim=0`.
  Same scenario, learning rate 0.0004, absolute actions. 20M further steps.
- The load and the pin both took. First frame: entropy −6.2 (the continuous
  heads at log-std −1.6 are about −18, and the discrete heads add about +11).
  The first update's KL was 73 because that rollout had been sampled at the
  old standard deviation of 2.4; by 262K steps the KL was 0.019. fp32, value
  mode 0, 4.9/24 GB. No gen-val. The 3060 scores fairbanks_1 and atlanta_2.
  TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r13`).

### 2026-10-18 17:30 IST power cut: r13 resumed from 6.55M as r13b
- r13 died at 8.0M steps, trainer uptime 8 min 51 s, with about 14 min left.
  `0000000007864320.bin` was 0 bytes and was removed. Last intact weights
  `r13/ckpt/0000000006553600.bin`. Fairbanks before the cut: 1.34 at 1.3M,
  1.23 at 2.6M, 1.45 at 5.2M. Atlanta 0.133 and 0.179 at 1.3M and 2.6M.
  The 5.2M atlanta eval did not finish.
- r13b loads that checkpoint and trains the remaining 13.5M steps with the
  same flags, log-std still pinned at −1.6. First frame entropy −6.5, KL 0.031
  (the loaded policy was already narrow, so there is no KL spike). fp32, value
  mode 0, 4.8/24 GB. TensorBoard restarted on http://127.0.0.1:6006.

### 2026-10-18 17:49 IST r13b finished: pinning sigma at 0.2 undid the fairbanks gain
- 13.4M steps in 15 min 14 s. With r13's 6.55M that is the planned 20M.
  Final weights `results/runs/r13b/ckpt/0000000013434880.bin`. Entropy stayed
  at −6.5 to −6.3.
- Deterministic fairbanks 1.25 → 1.23 at 2.6M → 1.45 at 5.2M → 1.80 at 20M.
  Atlanta 0.096 → 0.137. Training-episode score 1.59 and deterministic score
  1.80, so the two controllers now agree. Mean deviation 5.2 K → 13.2 K,
  out-of-band 0.76 → 0.89, worst zone 34 K → 43 K. Electricity 18 → 4.7
  thousand kWh. The narrow policy walked the mean off the comfort it had
  under the wide samples.

### 2026-10-18 18:03 IST r14 started: both weathers from r12's mean
- r13 showed that pinning the standard deviation at 0.2 walks the mean off the
  fairbanks gain. r12's deterministic score was still falling with the noise
  left wide, so r14 continues from `r12/ckpt/0000000019988480.bin` with
  `ent_coef=0.001` and `logstd_anneal=0`.
- Trains `building_train_1` on fairbanks_1 and atlanta_2 together: one group,
  128 envs each, 256 total, value mode 2. Learning rate 0.0004, absolute
  actions, 40M steps. First frame entropy 221.7, so the wide noise loaded.
  fp32, 5.0/24 GB. The 3060 scores both weathers with the same fp64 episodes.
  No gen-val. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r14`).

### 2026-10-18 19:13 IST r14 finished: fairbanks 1.25 → 0.92, atlanta 0.096 → 0.080
- 40.0M steps in 47 min 59 s. Final weights
  `results/runs/r14/ckpt/0000000039976960.bin`.
- Deterministic fairbanks 1.49 at 1.3M, 0.99 at 7.9M, 0.86 at 32.8M, 0.92 at
  40M. Atlanta 0.159 at 1.3M, 0.066 at 15.7M, 0.080 at 40M. The loaded r12
  policy was 1.25 and 0.096. The 32.8M pair (0.856 and 0.090) is better than
  the final pair on the sum of the two scores.
- Fairbanks productivity −2.02 → −1.44. Energy 0.061 → 0.092, carbon
  0.126 → 0.184. Comfort is 94% of that penalty. Sampled fairbanks episodes
  score 0.38. Sampled atlanta episodes score 0.04, against a deterministic
  0.08, and atlanta's training deviation stayed near 0.2–0.5 K.
- Entropy 222 → 254, standard deviation 2.4 → 3.3. End KL 0.0015, clip
  fraction 0.007. Fairbanks over the last 8M stays inside 0.86–0.97.

### 2026-10-18 19:52 IST r15 started: entropy bonus off, from the best r14 pair
- A longer r14 would keep the entropy term pushing log-std up. The trainer
  clamps log-std at 2 (standard deviation about 7.4). Gamma and comfort
  shaping stay closed: both were measured failures, and shaping is barred.
- Loads `r14/ckpt/0000000032768000.bin` (fairbanks 0.856, atlanta 0.090).
  `ent_coef=0`, `logstd_anneal=0`, same two weathers, learning rate 0.0004,
  40M steps. First frame entropy 248.0, unchanged at the second frame, so the
  noise did not collapse on load. fp32, value mode 2, 4.9/24 GB. The 3060
  scores both weathers. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r15`).

### 2026-10-18 20:49 IST r15 finished: noise held, fairbanks did not improve
- 40.0M steps in 48 min 11 s. Final weights
  `results/runs/r15/ckpt/0000000039976960.bin`.
- Entropy stayed in 246.7–249.7. Standard deviation 3.12 at the start and at
  the end. Removing the bonus stopped the climb. The policy-gradient term on
  log-std did not shrink it.
- Deterministic fairbanks 0.886 at 1.3M, 1.190 at 11.8M, 0.927 at 40M, against
  0.856 on the loaded weights. Atlanta 0.090 → 0.072. Pair sum 0.946 → 0.999.
  Sampled fairbanks stayed near 0.36. KL near 0.002, clip fraction near 0.02.

### 2026-10-18 21:06 IST r16 started: scheduled standard deviation, bonus off
- r15 held the standard deviation at 3.12 and the mean did not beat 0.856.
  The mean gradient scales as one over variance, and the samples at that width
  are the clipped rails. r13 showed a jump to 0.2 walks the mean backward.
- Loads the same r14 pair, `r14/ckpt/0000000032768000.bin`. `ent_coef=0`,
  `logstd_anneal=1`, `init_logstd=1.14` to `logstd_final=0` (standard deviation
  3.12 → 1.0) over 40M steps. Same two weathers, learning rate 0.0004. The
  3060 scores both weathers. No gen-val. TensorBoard: http://127.0.0.1:6006
  (`results/runs`, run `r16`).

### 2026-10-19 08:44 IST r22 restarted fresh, no automatic stop
- The 10M guard fired on fairbanks 0.232 and `norm/logit_outside` 0.953 while
  fairbanks was falling (0.337 at 5M) and the outside fraction had turned down
  to 0.944 by 11M. Same spec again from `r21/ckpt/0000000091750400.bin`.
  The guard is not armed.

### 2026-10-19 08:05 IST r22 started: chain floor 0.25, five weathers, 40M
- Forward location stays `3*tanh(logit/3)`. The loss floors the chain at 0.25
  so a head past the rail can come back. Log-std stays at the checkpoint's 1.14.
- 320 agents, 64 per weather, building_train_1 atlanta/denver/fairbanks/phoenix/
  singapore. 488 epochs, 39,976,960 steps, from `r21/ckpt/0000000091750400.bin`.
  Scores those three: fairbanks, atlanta, denver. Stop near 10M if
  `norm/logit_outside` is still above 0.95 and fairbanks is no better than 0.21.
  TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r22`).

### 2026-10-19 07:51 IST r21 stopped at 286M: improved, then saturated
- Fairbanks 0.193 at 85.2M (loaded 0.856), pair best at 91.8M (0.210 / 0.061).
  End of run 0.373 / 0.117. Comfort moved, then `norm/logit_outside` reached
  0.998 and the mean gradient died. The commit dropped entropy 259 → 99 and
  did not make a better action. Best weights:
  `r21/ckpt/0000000085196800.bin` and `0000000091750400.bin`.

### 2026-10-19 01:25 IST r21 started: 300M, hold the loaded width, then commit
- r20 finished at fairbanks 1.571 and atlanta 0.082. KL stayed near 0.02 and
  `norm/logit_outside` near 0.77, so the hard clamp froze the heads that had
  learned. The location is now `3*tanh(logit/3)`.
- Log-std is held at 1.14 for the first 80% of 4578 epochs (300,023,808 steps),
  then moves to −1. 256 agents, horizon 256, minibatch 2048, replay 4.
  Checkpoint every 100 epochs. Both weathers, from the r14 weights. The 3060
  scores with `tanh_squash` on. No third weather unless one score improves
  while the other is worse by ≥0.3 for three checkpoints. TensorBoard:
  http://127.0.0.1:6006 (`results/runs`, run `r21`).

### 2026-10-19 01:20 IST r20 finished: the clamp held, fairbanks walked to 1.57
- KL near 0.02 and explained variance near 0.99 through 40M. Fairbanks 0.921 at
  9.2M, 1.571 at 40M (loaded 0.856). Atlanta best 0.063 at 13.1M, 0.082 at 40M
  (loaded 0.090). Best weights remain the r14 checkpoint.

### 2026-10-19 00:02 IST r20 started: Gaussian location clamped to ±3
- r18 and r19 left the squashed ratio at 1e4–1e6 at two widths. The Gaussian
  location is now clamp(logit, ±3), artanh(0.995), with no mean gradient
  outside the box. `norm/logit_outside` records how many active heads that
  binds.
- Loads `r14/ckpt/0000000032768000.bin`. `tanh_squash=1`, log-std pinned at 0,
  `ent_coef=0`, centering off, learning rate 0.0004, 40M steps, both weathers.
  The 3060 scores with `tanh_mean` and no squash. No gen-val. TensorBoard:
  http://127.0.0.1:6006 (`results/runs`, run `r20`).

### 2026-10-18 23:52 IST r19 stopped: σ = 1 still left the ratio at 1e6
- Entropy sat at 148.7, so the pin took. Explained variance stayed 0.80–0.96.
  KL was 3 at 0.07M and 1e4–3e6 after that, through 1.3M.
- Deterministic fairbanks 0.916 at 1.3M, 0.866 at 2.6M, against 0.856 loaded.
  Atlanta 0.101 then 0.096, against 0.090. Stopped at 2.6M.

### 2026-10-18 23:47 IST r19 started: squash with standard deviation pinned at 1
- r18's ratio sat on the ±20 clamp at the loaded width. This run overwrites
  log-std with 0 every epoch (`init_logstd=0`, `logstd_final=0`), keeps
  `tanh_squash=1` and the ratio clamp, and turns centering off.
- Loads `r14/ckpt/0000000032768000.bin`. `ent_coef=0`, learning rate 0.0004,
  40M steps, the same two weathers. The 3060 scores both on the raw reward.
  No gen-val. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r19`).

### 2026-10-18 23:44 IST r18 stopped: the ratio sat on the clamp
- Stopped at 0.72M. The ±20 clamp kept the critic finite: explained variance
  0.67–0.98, entropy stayed 248, value loss peaked at 180. KL was 1.3e4 at
  0.20M, 1.8e7 at 0.39M, and 7.3e7 at 0.72M. The loaded width puts the
  squashed ratio on the cap. No checkpoint was scored.

### 2026-10-18 23:42 IST r18 started: same squash, log-ratio clamped
- r17's ratio overflowed at 0.26M and the value head went NaN. The log-ratio
  is now clamped to ±20 before the exponential, and a non-finite policy or
  value gradient is dropped. `puffer_f64` was copied from the fp64 `./puffer`
  before this run's fp32 rebuild.
- Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1`,
  `env.rew_center=1`, `env.center_beta=0.01`, `ent_coef=0`, `logstd_anneal=0`,
  40M steps, the same two weathers and learning rate 0.0004. The 3060 scores
  both weathers on the raw reward. No gen-val. TensorBoard:
  http://127.0.0.1:6006 (`results/runs`, run `r18`).

### 2026-10-18 23:16 IST r17 stopped: the ratio overflowed and the critic went NaN
- Stopped at 15.7M. Explained variance is 0 from 0.39M because the return
  sums are non-finite (the logger writes 0 when return variance is not
  positive). `norm/adv_inv_std` is exactly 10000 from 0.46M, i.e.
  `1/sqrt(1e-8)` after `max(NaN, 0)`. Entropy is bit-identical at 148.72845
  from 0.46M on, KL about 1e-11. No update landed after that.
- The break is at 0.26M: entropy still 248, centering mean −0.006, KL 8.4e12,
  policy loss 2.7e13. The logit ran until the importance ratio overflowed.
  An infinite ratio with a negative advantage is not cleared by the clip, so
  the value head went NaN. Training episodes at the stop are fairbanks ~3.4
  and atlanta ~0.14. No deterministic score: `puffer_f64` was absent.

### 2026-10-18 22:54 IST r17 started: squash the sample, center the learning reward
- r16's advantage mean sat around zero once explained variance was ~0.99, so
  stepping the average reward by the advantage (the paper's PPO appendix,
  equation 10) would not remove the constant. The on-policy running mean
  (equation 4) is the update, per scenario, with no division by std, μ starting
  at 0. r13 and r16 both walked fairbanks off the heat by shrinking the noise:
  the plant scores the clipped rail and the gradient updates the unclipped
  sample. `tanh_squash` puts the noise inside tanh and leaves the loaded
  log-std alone.
- Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1`,
  `env.rew_center=1`, `env.center_beta=0.01`, `ent_coef=0`, `logstd_anneal=0`,
  40M steps, the same two weathers and learning rate 0.0004. The 3060 scores
  both weathers on the raw reward. No gen-val. TensorBoard:
  http://127.0.0.1:6006 (`results/runs`, run `r17`).

### 2026-10-18 22:02 IST r16 finished: narrowing the noise walked the mean off the heat
- 40.0M steps in 48 min 32 s. Final weights
  `results/runs/r16/ckpt/0000000039976960.bin`. Entropy 259 → 149, so the
  schedule reached log-std 0. Training-episode score 2.0 → 0.36. Value loss
  spiked to 3254 on the flatten and ended under 10.
- Deterministic fairbanks 0.836 at 1.3M (noise still wide), 1.229 at 19.7M,
  0.846 at 34.1M, 1.113 at 40M, against 0.856 on the loaded weights. Atlanta
  0.083–0.139, final 0.093, against 0.090 loaded. Fairbanks productivity
  −1.31 → −1.79, energy 0.113 → 0.074. Same direction as the σ = 0.2 pin.

### 2026-10-19 r22 completed; r23 campaign handed to Codex
- r22 completed 39,976,960 steps in trainer uptime 3054.8 s. Best monitored
  three-weather mean is 0.085699151 at 24,985,600 steps (fairbanks 0.183536658,
  atlanta 0.041155705, denver 0.032405089), versus matched pid_water 0.088564368.
  Final mean 0.096129327. This is training-scenario evidence, not a leaderboard
  or generalization claim. r21's original pair improves 17.2% at the selected r22.
- The final critic remains finite (EV 0.9738). Entropy 259.43 -> 277.86 shows
  trainable exploration widths grew despite zero entropy bonus. The logit-outside
  fraction alone cannot establish that the actor's derivative vanished.
- User authorizes efficient use of both GPUs, careful experiments, no removal of
  important files, and updates to Experiments.md and this history. Target is now
  0.00598. Keep the original reward and pure learned absolute actions; no baseline
  residual, controller initialization, guard, comfort shaping, or sealed data.
- Historical fp64 evaluator/config/source/patch and checkpoint hashes preserved
  under results/campaign_r23. GPU 1 starts matched five-weather and evalcore3
  development evaluation of r21@91.8M, r22@25M, and r22-final. GPU 0 is reserved
  for loss verification and the required full port/trainer gate before training.
- First hypotheses: masked padding gradients, surrogate chain derivative, and
  nonstandard ratio diagnostics. Reproduce defects before fixing; record failures
  and isolate changes with controlled runs. No claim of a new algorithm yet.

### 2026-10-19 r23 loss audit and scheduling
- Corrected the focused test harness's initial buffer-layout error before drawing
  conclusions. Actual unmodified CUDA kernels then fail three cases: padding
  categorical gradients/entropy, the intentional non-exact saturated chain floor,
  and counting only the first categorical class for the geometric-ratio divisor.
- Fixed masking consistently in sampling, cached likelihoods, loss gradients, and
  ratio counting. Exposed squash_chain_floor (default exact 0, r22 control 0.25),
  guarded freezing of learned widths with cold momentum, and diagnostic counters.
  Canonical sources live in patches/pufferlib_core.diff; no checkout edited directly.
  All 80 focused scalar checks pass after the repair. Full port/trainer verification
  is in progress; the focused test is now included in verify_all.sh --trainer.
- Matched five-weather scores: r21 0.148698754, r22@25M 0.079433892, final
  0.085689033, pid_water control 0.086294041. Matched unseen evalcore3: r21
  0.257756722, r22@25M 0.183006974 (other controls pending). Different batch-layout
  results are kept separate. Pareto fronts use identical scenario sets only.
- Prepared exactly the declared 30 shipped scenarios as tr23_shipped30, without
  copying blob contents. Deep check against eval9: 30 train / 45 eval, clean.
- User reiterates time efficiency and GPU roles: 3090 for training, 3060 for eval;
  distribute pending eval across both once a training phase finishes. New runs
  use unique logs/checkpoints and immutable fp64 runtime snapshots. First planned
  screen is lr=0 identity, then matched 5M-step lr 0.0004 versus 0.0001 pilots.

### 2026-10-19 r23 gate passed; first pilot and shared-policy prototype
- Full gate `results/verify_20261019_102139.log`: ALL CHECKS PASSED. Immutable
  corrected fp64 runtime archived. lr=0 identity checkpoint exactly matches r22@25M.
- r23a finishes 4,997,120 steps in 461.28 s trainer time (10.83K physical SPS),
  finite ratios and zero clamps. r23b lower-lr comparison running on 3090;
  3060 evaluates full episodes. Corrected parent five-weather score exactly
  reproduces historical 0.07943389249.
- Completed evalcore3 baseline 0.05078483742 dominates historical learned
  candidates in all three components. r22-final 0.1966018673. Generalization
  remains the main failure; no winning or leaderboard claim.
- Optional shared local actor / pooled context prototype added to canonical
  patch sources. It excludes PID-derived columns, shares actuator parameters,
  and leaves physics/reward unchanged. All correctness checks and full gate
  must pass before its fresh 30-scenario training screen. Dependency/isolated
  checkout setup failures retained; existing checkpoints untouched.
- A provenance-recording full-episode evaluator tests CUDA graph replay
  against the unchanged reference before adopting faster evaluations.

### 2026-10-19 05:35 UTC — user requests stop
- Stopped all campaign processes, containers and queued work; both GPUs have
  no campaign compute remaining. Preserved checkpoints and logs. Do not resume
  without user instructions. Automatic review blocked a post-stop queued launch.
- Both flat-policy pilots completed 4,997,120 steps. r23a deterministic five-
  weather score 0.08447111271 regresses 6.34% from parent 0.07943389249 and is
  dominated in all components. r23a OOD eval was stopped incomplete. r23b has
  no valid full evaluation; preserve its trained checkpoints for later review.
- Fixed graph score export at synchronized puf_log, reading a diagnostic symbol
  only. Verification-only binary reproduces the entire parent JSON exactly.
  88.18 s measured process time; no demonstrated speedup. Initial graph export
  failure and a live shell-runner edit/read race are retained as failures.
- Shared actor focused checks pass: 60 finite differences, permutation
  equivariance, PID-column independence, padding and forward CUDA graph.
  It has not been trained. Its full port/trainer gate was stopped before
  completion; must finish before new-architecture training or promotion.
- Pareto JSON/CSV/PNG/SVG exported under results/campaign_r23. No Kaggle
  submission or score below 0.00598; generalization remains unresolved.

### 2026-10-19 — research protocol correction; permission required
- User requires hypothesis/control/budget/evaluation criteria proposed first,
  then explicit permission before experimental GPU work. No autonomous restart.
  Use 20M/30M longitudinal runs, retain complete curves and report dips/recovery.
- r23a's 5M regression is preliminary endpoint evidence, not a rejection of a
  method's long-run behavior. pid_water is not our research baseline; learned
  controls and the official target 0.00598 define the research comparison.
- Proposed r24 (NOT APPROVED): two matched 30M continuations from r22@25M,
  same architecture/runtime/settings/960-agent batch; change only the training
  mixture (five weathers/current building versus 30 scenarios/six buildings).
  Fixed learning rate 0.0001, full episodes every ~5M, primary last-three-point
  development mean, full curve/components/worst-building report. Approximately
  60M total training transitions; no score-based early stopping or extensions.
- 3090 training / 3060 concurrent evaluation, then both evaluate. Full pending
  verification precedes the proposed study. All work remains stopped awaiting
  explicit user approval; shared architecture is outside this proposed ablation.

### 2026-10-19 — r26 approved and started
- User approved the continuation from r25 at 30,044,160. Training batch is the
  six r25 buildings plus building_train_1, five weathers, 35 scenarios.
  560 agents (16 per scenario), same r25 settings, about 30M steps. Bar:
  new buildings <= 0.071, held-out <= 0.127, original Fairbanks <= 0.213.
  No score-based stop.

### 2026-10-20 — r32 scoring on the 3060 stopped
- The search on the 3090 keeps running. Further checkpoints are not scored
  on the 3060; that GPU is reserved. The random-start score, 9.14, stands.

### 2026-10-20 — r32 restarted after a power cut
- Verification had passed and the fp32 binary was already built. The cut
  fell during the first update, after the random-start checkpoint (fp64
  9.14). That score is in `results/study_r32/eval/poweroff/`. The search
  was started again from the same seed. No setting changed.

### 2026-10-20 — r32 started: paired search in the weights
- Flat policy from scratch on building_train_1__fairbanks_1. Both signs of
  one random weight change, scored by the whole month's official return.
  16 directions, 8 updates, noise 0.02, step 0.05. The action during a
  trial is the mean. Scores are that scenario on the fp64 binary.

### 2026-10-20 — r31 finished, best Fairbanks point is 1.23 at 20M
- 29,884,416 learning steps, 66 min, exit 0. fp64 scores 7.26, 6.65, 1.30,
  1.23, 1.57, 2.11. Best is 20M: productivity −1.96, energy 0.084, carbon
  0.172. The early points were a fuel cut. The drop to 1.30 is the rooms
  warming. After 20M comfort worsened again. r10 is 2.47, r30 ended at 9.76.

### 2026-10-20 — r31 started: paired official return on the plant setpoints
- Flat policy from scratch on building_train_1__fairbanks_1. r10 settings.
  256 learning envs plus 256 probes. A probe repeats the action with supply
  air and hot water raised by 0.25 for the 256-step window. That official
  return difference moves those two setpoints. Zone dampers keep λ = 0.95
  and standardized advantages. 29,884,416 learning steps. Scores are the
  same scenario on the fp64 binary. The building starts at 21 °C; the cold
  is produced by the policy.

### 2026-10-20 — r30 finished, January Fairbanks stayed near 10
- 29,884,416 steps, 35 min, exit 0. fp64 scores 11.03, 10.21, 10.31, 10.37,
  10.21, 9.76. Final productivity −16.22, energy 0.099, carbon 0.025.
  Comfort did not recover; fuel fell. r10 on this scenario is 2.47.

### 2026-10-20 — r30 started: one Fairbanks scenario, λ = 1, no advantage standardization
- Flat policy from scratch on building_train_1__fairbanks_1 only. r10 settings
  except the 256-step return is the advantage and it is not standardized.
  29,884,416 steps. Scores are that same scenario on the fp64 binary.

### 2026-10-20 — r29b distillation stopped during pass 6
- User asked to save and stop. Passes 1–5 are `results/study_r29b/student_pass01.bin`
  through `student_pass05.bin`. Best action match is pass 4 (3.70, down from
  5.65). Value loss was still falling (20.3 to 12.7). The five teacher
  rollouts are kept. The gate and the 200M had not started.

### 2026-10-20 — r29b gate relaxed to the near-0.50 band
- Fairbanks on `building_train_1` passes at or under 0.60, and the 35-scenario
  mean at or under 0.10. A much worse score gets one distillation repair before
  any fallback. The year-shaped run starts only if that repair does not move
  the action loss. The 200M still waits for a passing gate and for verify.

### 2026-10-20 — r29b approved: distill r26, then 200M if the gate passes
- Teacher rollout records squash locations with log-std fixed at −1. `SB_RECORD`
  is off unless that directory is set, and it refuses a CUDA graph. The patch is
  in `patches/pufferlib_core.diff`. Thirty-five scenarios do not fit in one
  process on the 3090, so the same 9,676,800 steps are recorded as five groups
  of seven. The 200M does not start unless the fp64 gate passes.

### 2026-10-20 — r28 finished, shared-from-scratch is not the lever
- 29,982,720 steps, all six fp64 evals complete. Panel mean 1.435 at both
  15M and 30M. Fuels fell, panel productivity worsened. January Fairbanks
  ended at 5.04 with energy 0.007. The other three cold windows fell to
  0.464, 0.130 and 0.108. No further run started.

### 2026-10-19 — r28 cold windows started
- User approved training the shared actor on cold scenarios only, at the
  fastest setting. Four windows: January Fairbanks, October Fairbanks, and
  February Denver on two buildings. fp32 solve, 160 envs per scenario.
  Scores stay on the fp64 binary. No source change, so the full gate was
  not re-run. Random init. About 30M steps on the 3090.

### 2026-10-19 — r27 stopped near 21.8M
- User asked to cancel after explained variance locked and the policy traded
  comfort for lower electricity. Stopped at checkpoint 0000000021790720.
  Explained variance 0.98 since 14M. Electricity about 79 kW to 18 kW, gas
  about 227 kW to 62 kW, productivity about -0.3 to -1.8, scenario mean
  stuck near 1.10. Shoulder scenarios improved slightly; hot scenarios and
  January Fairbanks got worse. No further training started.

### 2026-10-19 — r27 restarted without pid_water
- User said start the run instead of waiting for the gate. The shared binary was
  rebuilt with SB_SHARED_POLICY, so base_pidw_k is not compiled in. SHA256
  b2b1ed04268ae1feec19743d013737a0b258d80a8ad64451c74c1a27d6fb4bc2.
  Training is r27_shared30m on the 3090, random init, hidden 256, two layers,
  560 agents, about 30M steps. The earlier checkpoints stay in
  results/runs/r27_shared30m_pidkernel and are not loaded. The full gate
  re-ran on the 3060: results/verify_20261019_212146.log ended ALL CHECKS
  PASSED. The training run was left running.

### 2026-10-19 — r27 stopped so pid_water is not executed
- The first r27 start reached about 2.9M steps. Its action was absolute, and the
  shared features skip the pid_water columns, but the step still ran the
  controller and wrote its proposal into the observation. Stopped at the user's
  instruction. Checkpoints kept in results/runs/r27_shared30m_pidkernel.
  The shared build no longer calls that kernel.

### 2026-10-20 — r32 held, r33 is the reinforcement-learning run
- The weight search is not resumed after the second power cut. Its checkpoints
  stay in results/runs/r32_fairbanks_ars. r33 trains the shared actor from
  scratch on January Fairbanks, on the 3090, and writes the per-step reward
  parts that earlier runs did not keep. No evaluation during that training.
- r33 finished 30,081,024 steps in 46 min 42 s, exit 0. Six checkpoints are in
  results/runs/r33_fairbanks_rl/ckpt/. The trajectory is 574 MB. Last-epoch
  value loss 5364, entropy 74.8. No fp64 score yet.
- r34 trains the same actor on the same scenario. The value and the advantage
  are the competition weights on the raw terms. The potential is not added,
  and the terms are not standardized. Entropy coefficient 0.001, constant.
  fp64 scores of every checkpoint follow the run.
- r34 finished 30,081,024 steps in 46 min 48 s, exit 0. fp64 scores on this
  Fairbanks month: 5.220, 5.268, 5.329, 9.780, 10.138, 11.619. The best point
  is 5.0M. By 15M the fuels were almost off and the rooms were not back.
  Explained variance ended at 0.09. The advantage was still three separate
  returns, and the value did not see the delay features.
- The next run uses one scalar return. The value head reads the delay
  summary, component heads stay off, and the feature trace is bound for that
  scalar path. Gate verify_20261020_200341 ended ALL CHECKS PASSED. r35 is
  the 50,135,040-step Fairbanks run on that build.
- r35 finished in 58 min. fp64 scores from 5M through 50M: 5.065, 5.071,
  5.071, 5.071, 5.072, 5.072, 7.368, 7.385, 5.340, 5.072. Best checkpoint is
  the first, plant off. The heated month at 2.2M scored 3.96 and was not saved.
- r36 is the same month from scratch on the r35 binaries. The one change is
  gae_lambda 1. 39,813,120 steps, a checkpoint every finished month.
- r36 was stopped near 19M at the user's request. Occupied-day ev_ret was
  still falling. r37 is r35's flags with env.desync=1, 50,135,040 steps.
- r37 finished in 62 min. fp64 scores 5M through 50M: 1.483, 2.373, 1.201,
  0.540, 0.429, 0.330, 0.473, 0.452, 0.440, 0.605. Best is 30.1M, productivity
  -0.297, energy 0.227, carbon 0.534, below r14's 0.856 on this month.
- r38 is r37 from scratch. The entropy bonus is the entropy of the executed
  tanh action. Entropy coefficient stays 0.001. Continuing the 30.1M weights
  with entropy coefficient 0 is pending and starts only if r38 fails.

### 2026-10-21 — joint KL gate, two weathers
- r40's per-dimension clip does not see the 97-action policy. Joint KL median is
  19 nats, and Muon steps the actor by the same amount on every minibatch.
  ent_coef stays 0.001. r44 trains the shared MinGRU value from scratch on
  Fairbanks and Atlanta, and stops the actor once joint KL exceeds 1 nat.
- Stopped near 37.6M. Per-head KL was 0.00014. Fairbanks training episodes
  stayed near 5 from 9M; Atlanta eased to 0.09. The 1 nat cap is not how r40
  learned: that descent happened at about 20 nats of joint KL.

### 2026-10-21 — PopArt on the separate critic
- The critic value is σ(w·h + b) + μ. μ and the second moment come from the
  GAE return, once per epoch. The last row, the bias, and their momentum are
  rescaled so the value does not jump. van Hasselt et al. 2016.
  `patches/src/sb_shared_policy.cuh`. r42 trained 50,135,040 steps. Best Fairbanks
  month 0.926 at 30.1M, productivity −1.339, energy 0.213, carbon 0.400.

### 2026-10-21 — separate value network
- The shared-policy critic no longer reads the trunk. It is a two-layer tanh MLP
  on the fixed building summary and the delay summary. Its gradient is not added
  to the trunk. One global clip remains. Andrychowicz et al. 2020, C47.
  `patches/src/sb_shared_policy.cuh`. The finite-difference check covers the new
  weights. r41 trains this, from scratch, on the r40 flags.

### 2026-10-19 — r27 approved, gate passed, training started
- User approved the shared zone actor after the r26 scenario breakdown.
  `verify_20261019_200556.log` ended ALL CHECKS PASSED. The independent read
  agreed with the shared actor and rejected a smaller network as a second
  factor, so the run uses hidden 256 and two layers, matching r26. Same 35
  scenarios, random initialization, r26 loss settings, about 30M steps. The
  r26 checkpoint is the control and is not loaded. Decision is the mean of the
  20M, 25M and 30M checkpoints: held-out cold at or under 0.150. In-batch
  still above 0.10 is inconclusive.

### 2026-10-19 — r25 restarted after power loss
- The first start died near 2M steps. Checkpoints kept in
  results/runs/r25_acquire30m_poweroff. 960 agents had already failed
  cudaMalloc, so the restart uses 720 agents (24 per scenario), same parent
  and settings. Fresh 30,044,160 steps, no score-based stop.

### 2026-10-19 — r25 approved and started
- User approved one 30M acquisition run from r24b at 24,576,000 steps.
  Training buildings are 10001, 10004, 10012, 15002, 15021, 15025, five
  weathers each. Shipped six, sealed seeds 30000-30019, and eval9 stay out.
  Same r24b settings and fp64 runtime. Curves: the new buildings, the original
  five weathers, and evalcore3. No score-based stop. The equivariant prototype
  stays untrained.

### 2026-10-19 — r24 approved
- User approves the fixed two-arm 30M training-distribution study. Full gate
  precedes training; 3090 trains, 3060 evaluates, then both finish evaluation.
  Reuse existing code; keep Markdown short and detailed artifacts in results/study_r24.

### 2026-10-21 — r40 audit, no trainer changes
- User assigned the 3060 for audit diagnostics; two existing-binary fp64 frozen
  replays (65.2M and 70.2M) exactly matched their stored score fields, 85.1/84.8 s.
  The 3090/r42 runner was left untouched. No canonical source/tool was edited.
- Late occupied-zone regression occurs without the 50M value/KL explosion.
  CPU probes expose stale-sample entropy, thresholded-run-head entropy mismatch,
  changing GAE with stale normalization, and geometric-surrogate logging scale.
  The old Adam/fp64-training labels are inaccurate: Muon, fp32 training/fp64 scoring.
- All 14 checkpoints, training metrics, checked time/zone/action/fuel decomposition,
  limits and an unlaunched single-factor protocol: results/audit_r40_20261021/REPORT.md.

### 2026-10-21 — r43 entropy comparison approved
- User assigns the free 3090 to the audit's first proposal: two fixed 20,054,016-step matched restarts from r40's best weights + Muon state. Continuous entropy sampling is the only changed learning factor; run-head regularization stays unchanged.
- Add an explicit r40 shared-critic compatibility path without changing r42 defaults. Fresh-noise derivative/mask/floor/graph checks: 3,520 checks, zero failures. Full simulator/trainer verification and parent/first-update parity precede study training.
- Preserve both curves, all checkpoints, terms and fuels; GPU1 performs deterministic fp64 Fairbanks evaluations and checked diagnostic replays. Detailed artifacts: results/study_r43.
- Full gate `results/verify_20261021_200003.log`: **ALL CHECKS PASSED**. Exact parent replay; byte-identical first-epoch control weights + Muon state; fresh-noise end-to-end graph check finite. r43 fixed-budget controller PID 133572 launched GPU0 control at 14:51:26 UTC; treatment follows automatically, GPU1 scores.

### 2026-10-21 — all experiments stopped; from-scratch RL rule
- User cancelled every run and rejected starting classical RL experiments from saved checkpoints. Stopped r43 controller/process group and training container; treatment was never launched. Saved checkpoints/logs remain. r43 is incomplete and provides no completed control/treatment conclusion.
- AGENTS.md now requires random initialization and fresh optimizer state for new experiments; checkpoint resumes need an explicit user exception. Cancelled the stale r37 warm-start instruction and marked old live-run guidance historical. No replacement GPU work is authorized.

### 2026-10-21 — r46 zone advantage, five interleaved weathers
- User approved the advantage split. Zone reheat and damper train on that zone's productivity share plus the building's energy and carbon. Plant heads keep the full return. The five official `building_train_1` weathers are round-robin across env index, so each 8-env minibatch holds all five. r40's update otherwise: shared MinGRU value, `ent_coef` 0.001, minibatch 2304, no KL stop, from scratch, seed 73, 330,891,264 steps.
- Default off (`env.zone_adv`, `train.zone_adv`, `env.interleave`). The scored return is unchanged. Sunday fp64 of all five is on the 3060. Keep the checkpoint whose worst weather is best. Driver: `results/study_r46/launch.sh`.

### 2026-10-22 — r47 Adam against Muon
- PufferLib 5.0 steps Muon. `train.adam=1` selects Kingma Adam (β1 0.9, β2 0.999, ε 1e-8, bias correction, weight decay 0) after the same global grad clip. Default 0 is Muon, and that path's allocation is unchanged. The Adam step counter is a device integer so a captured train graph advances it.
- Both arms are from scratch on this binary, seed 73, Fairbanks only, shared value, zone advantage off, 30,081,024 steps, no evaluation. Adam on GPU 0, Muon on GPU 1. r40 is not the control: its binary is older. Driver: `results/study_r47/launch.sh`.
- Both exited 0 at 16:58 UTC, six checkpoints each. Adam 37 min, Muon 73 min. Deterministic fp64 months, desync off, every checkpoint: Adam 0.797, 5.160, 7.655, 6.962, 9.825, 8.578. Muon 1.525, 1.481, 1.152, 0.631, 0.561, 1.008. Muon is lower at five of the six points. Its best is 25.1M, productivity −0.321, energy 0.541, carbon 1.298, electricity 13.4 MWh, gas 913 MWh. Adam after 5M is nearly plant-off. Checkpoints and sidecars kept. Curve: `results/study_r47/curve.json`.

### 2026-10-22 — r48 fresh entropy and the sibling block
- Arm a finished, exit 0, epoch 408, six checkpoints. fp64 scores 8.697, 5.939, 9.774, 3.814, 2.197, 0.512. The last step is 0.512, productivity −0.644, energy 0.228, carbon 0.399, electricity 32.2 MWh, gas 269 MWh. r47 Muon is lower at the first five points. The 25.1M to 30.1M score kept falling. `logit_outside` peaks lower, 0.79 against 0.97, and ends higher, 0.51 against 0.34. Curve: `results/study_r48/curve_a.json`.
- Gate before launch: the fresh-noise kernels match the r43 source. The r43 check set re-ran at 3520 checks, 0 failures. A three-epoch CUDA-graph smoke of the r47 binary with both flags exited 0, finite, and printed the fresh-entropy banner. No source change.
- 2026-10-23 — fresh-entropy battery regenerated from the deployed source (base commit + live patch via `puffer_patch_source.py`). The entropy branch is byte-identical to the r43 snapshot; the harness appends the two new trailing `PPOKernelArgs` fields. Result: 3520 checks, 0 failures; deployed `algo.cu` sha256 `0a8e9926e7e777c58fd213a6523304abd13216711a6658b35e8b7f3694c7ae6b`. Artifacts: `results/study_r48/checks_r48/entropy_v3/`.
- Arm b finished, exit 0, six checkpoints, 416,864 floats. fp64 scores 7.339, 7.626, 11.827, 10.904, 0.290, 0.402. Best registered point 25.1M at 0.290, productivity −0.249, energy 0.209, carbon 0.493, electricity 5.6 MWh, gas 346 MWh. 30.1M rose to 0.402. Arm a is lower at 10, 15, and 20M. Curve: `results/study_r48/curve_b.json`.
- 2026-10-23 — joint-KL gate pre-flight (no run): sized the proposed `--train.joint_kl=60` arm from r48-A's stored SB_UPDATE per-visit rows calibrated to the TB epoch totals. Threshold 60 closes 61% of epochs, 98% closure over 15.0–22.6M (arm A's descent window), ~48/128 visits frozen per epoch. Arm declared dead; p99 fence ≈ 400 nats would touch 5% of epochs. Full entry in Experiments.md.
- 2026-10-23 — arm B (sibling) evaluated: best registered point 0.2897 @ 25.07M, current-recipe best on Fairbanks; morning-ramp concentration absent at that checkpoint (worst bin 0.060); blows up worse mid-curve (10.0–20.1M) and lands one window later than arm A; ends 0.4018 @ 30.08M vs arm A 0.512. Full entry in Experiments.md. Artifacts: study_r48/curve_b_bins.json, replay records.
- 2026-10-23 — entropy extension to 70.19M scored: wave (3.22/2.31/0.27/4.92 excursion/0.22/0.36/**0.1617 @ 65.18M**/0.24). New lineage-best dev point 0.1617, late basin holds five of six checkpoints under 0.36 at ~⅓ arm A's gas. Continuation caveats apply; one seed, dev only. Full entry in Experiments.md; sibling extension training next on GPU 0.
- 2026-10-23 — audited bins+zone census on the 0.1617 @ 65.18M replay: morning ramp still 60% of the score (bin 0.555, 42.5 of 96.9 MWh gas); every bin below arm A's 30.08M by 3-4x; loss is a distributed tail over 25-35 zones with zone 27 the standing worst in both arms (32.9 K / 41.0 K). decomposition.json in the replay dir.

### 2026-10-24 — r53 room value
- The 90-row zone critic on the pooled trunk does not name a room (probe correlations under 0.12 on r51 and both r50b months). The scalar's remaining-return slope, once `rew.bin` is put in official units, is 0.44–0.73. The scalar stays.
- `train.room_value=1` adds one 12-parameter head on that zone's 11 features. Damper and reheat use the whitened zone advantage. The gradient does not enter the trunk. `g_valve_zone` lives in `algo.cu`, beside the other PPO kernel arguments.
- From scratch, Fairbanks, 50,135,040 steps, GPU 1, run `r53_room_space`. Gate `results/verify_20261024_171509.log`: ALL CHECKS PASSED. First checkpoint and sidecar 1,667,504 bytes. Driver `results/study_r53/run_r53.py`.
- Stopped 2026-10-24 15:05 UTC at the user's request, epoch 298, dashboard about 22.0M steps. Exit 137 is that stop. Four checkpoints, 5.0M–20.1M, each 1,667,504 bytes with a sidecar. The room credit was moving the valves (zone KL 1.55 at epoch 60, against 0.013 on the sibling scratch) and the desync dashboard had left its best of 2.233 at 11.7M for a band of 5.93–8.42. Last box 7.59. At 20.9M the sibling scratch was 1.22 and the entropy scratch was 5.46. No fp64 month. r54 was left running.

### 2026-10-24 — r52 shipped-30 stopped
- Stopped `r52_shipped30_entropy_50m` on GPU 0 at the user's request, after six checkpoints (50.0M–300.3M file steps). fp64 of the 150.1M and 300.3M files on all 30 months: score 1.106 and 1.106. Energy under 0.0003 and carbon under 0.0007 on every scenario. `building_train_1__fairbanks_1` is 5.072. The dashboard low of 0.079 was a desync episode mean. Scores: `results/study_r52/eval/`.
- Verdict (analysis on the scored months): this is the plant-off basin, saturated. The two files agree to three decimals on every scenario, so the policy's fp64 behavior did not move across 150M steps; there is no descent surface left in this recipe. The weather-ranking of the thirty rows is the building's conditioning load alone (per-weather scenario means: Phoenix 1.810, Fairbanks 1.399, Singapore 1.370, Atlanta 0.732, Denver 0.220, against pid_water 0.078/0.079/0.047/0.030/0.038 on the same splits), a factor of 1.5–24 worse than pid_water per scenario — the six- worst rows are the extreme climates, not building quirks. One seed, not a leaderboard result.
- Measurement value: the run doubles as the all-30 no-control bound (fuel floor ≈ 0 everywhere, so any future batch arm has a re-ignition gate: gas > 0 with score under this table's row). It dates a collapse window inside the first 150M steps; scoring the 50.0M and 100.1M files (~2.5 h each) would close that window. The 0.079 dashboard low is kept as the recorded instance of the desync-mixing trap — never quote it beside month scores.
- Implication logged for the batch: "more steps" is ruled out on measurement. The batch needs the room-value recipe (if r53 shows the head does its job), the structured zone action, or the schedule/trust-region rungs to compete with plant-off; Fairbanks 0.134 @ 70.19M stays the only above-pid learned month. Full entry in `Experiments.md` (workspace root), section "r52 stopped — the full months are plant-off".

### 2026-10-24 — r54 return frozen at epoch start
- `train.freeze_return=1` computes GAE once from the stored rollout values, normalizes that advantage once over the rollout, and holds both for every minibatch visit. Default 0 keeps the recompute. It requires `adv_norm=2` and refuses the zone critic, the room value, zone advantage, and component values.
- Run `r54_return_at_start`: Fairbanks only, the entropy recipe, from scratch, 50,135,040 steps, ten checkpoints, fp32 and CUDA graphs, GPU 0. The sibling block is in this actor. fp64 months follow training. Gate `results/verify_20261024_195122.log`: ALL CHECKS PASSED. Driver `results/study_r54/run_r54.py`.
- Started 2026-10-24 14:40 UTC. Epoch 3 is 13.4K steps/s (evaluate 5.5 s, train 2.9 s). First checkpoint and Muon sidecar are both 1,667,456 bytes.
- Finished 2026-10-24 15:59 UTC, exit 0, ten fp64 months. Best is 0.227 at 50.1M (productivity −0.168, energy 0.240, carbon 0.391, gas 260 MWh). The 20.1M month spikes to 0.927 and the next save is 0.377. Not under 0.134.

### 2026-10-24 — r55 zone command, started, 80,216,064 steps
- `env.zone_command=1` turns the squashed damper-slot sample into damper and reheat. Heat opens reheat at the 0.1 floor, then the damper. Cool opens the damper with reheat shut. The pump runs when any present zone is heating. Hot-water temperature stays the learned sample. Reheat, pump head, and pump run are masked. Default 0 is the independent heads. Canonical absolute actions only. The header guard is `SB_ZONE_COMMAND_CUH`, because the env and the policy check include it through two paths.
- r55 keeps `train.freeze_return=1` and adds `env.zone_command=1`. Fairbanks, from scratch, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid, GPU 0. Started 2026-10-24 16:07 UTC. Gate `results/verify_20261024_211242.log` ended ALL CHECKS PASSED. Banner `zone command on (one heat curve per zone, pump follows heat)`, 56/288 active heads, first checkpoint and sidecar 1,667,456 bytes, epoch 3 at 13.6K steps/s. The comparison is r54's months.
- Finished 2026-10-24 18:12 UTC, exit 0, epoch 1088, sixteen fp64 months. Best is 0.217 at 55.1M (productivity −0.168, energy 0.227, carbon 0.354, electricity 36.1 MWh, gas 234 MWh). The next two saves stay heated, at 0.239 and 0.645. Plant-off through 35.1M. The 65.2M month burns 1016 MWh of gas. Not under 0.134. Curve: `results/study_r55/run.log`.
- The zero of that curve is the off state, and any positive zone sample slams the pump. A zero-mean policy starts plant-off. The next decode puts half reheat and a running pump at `u = 0`, closes the coil without opening the damper for `u < 0`, opens the damper to cool only when the supply is colder than the zone and the zone is above its cooling setpoint, and ties pump head and the hot-water cap to the average heat request.

### 2026-10-25 — r57 value trunk cap stopped (failed)
- `train.value_trunk_cap=1` clamps the value path's share of the shared trunk gradient at the policy path's magnitude once per minibatch visit: `scale = min(1, ||g_pi||/||g_V||)`. The value head's own gradient, the actor heads and the policy gradient are untouched; `max_grad_norm` stays 0.5; off is the untouched trunk, bit for bit; it cannot bite on the separate-critic path, where the value gradient into the trunk is zero by construction. Canonical sources `patches/src/sb_shared_policy.cuh` (the three kernels and the call in the decoder backward), `patches/pufferlib_core.diff` (flag, buffers, per-epoch telemetry) and `config/smart_buildings.ini`; working copies in `results/study_r57/sources/`. Telemetry is `SB_VTC <epoch> visits <n> capped <n> frac <f> min_scale <s>` per epoch plus TensorBoard `norm/vtc_capped_frac` and `norm/vtc_min_scale`.
- Arm `r57_decode_value_trunk`: r56's config plus that flag, from scratch, Fairbanks, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid, GPU 0. Started 04:04 UTC; first checkpoint and Muon sidecar 1,667,456 bytes. Gate `results/verify_20261025_091402.log` ended ALL CHECKS PASSED; `policycheck` reports 108+108+60+60 finite differences, 0 failures, so the off path is unchanged. Driver `results/study_r57/run_r57.py`.
- Stopped 05:25 UTC at the user's request: 860 epochs, 63,406,080 steps, twelve of sixteen checkpoints through 60,162,048, each with its sidecar. Marked failed; no fp64 month was scored and the arm is not extended.
- Reason: the ratio is standing, not a spike. The cap bit on 849 of 860 epochs; the per-epoch tightest scale was 1.000 once (epoch 1) and then 0.034, 0.019, 0.043, 0.040, 0.028, 0.105, 0.067, 0.072, 0.046 at epochs 101 to 860 (median 0.044, minimum 0.0144). So the clamp permanently removes the critic from the trunk, keeping 2–10% of its share, while the value head's own gradient is left uncapped by design — the target it was meant to protect is not protected. At 63.4M against r56's control at 63.1M: episode score 3.945 vs 0.369, comfort deviation 18.5 K vs 6.24 K, productivity −6.37 vs −0.52, out-of-band 0.917. `loss/ev_ret2` grew monotonically 1.19 → 1.5e4 → 4.7e4 → 3.4e5, failing the pre-registered excursion test by orders of magnitude. Steady SPS 13.2K against r55's 13.1K on the same shape, so throughput is not the cause.
- Verdict: a per-visit magnitude cap is the wrong instrument for this ratio; it changes the acting point permanently instead of clamping a spike. Do not extend r57, do not score its twelve files as a curve of this recipe, and do not load them. Control stays r56 (best 0.219 at 50.1M). Preserved: `results/runs/r57_decode_value_trunk/` (checkpoints and sidecars, `train.log`, `train.docker.log`, `FAILED.md`), `results/study_r57/` (`vtc.jsonl`, `sources/`, `run_r57.py`).

### 2026-10-25 — r58 value spike guard, started, 80,216,064 steps
- `train.value_spike_guard=1` guards the trunk-side value tangent against its own history instead of against the policy. Per minibatch visit `s = min(1, q*M/||g_V||)` with `q = 3`, `||g_V|| = ||grad_value_h||_2`, and `M` the trailing geometric mean of `||g_V||` (an EMA of `ln||g_V||` whose halflife is one epoch of visits, 128, so a short storm barely moves it and a real shift is followed). Applied to `grad_value_h` only; the policy path into the trunk, `wgrad_value`, the value loss and the actor heads are untouched, and a chronic ratio below q passes at s = 1 by construction. `q = 3` is grounded on r56's quiet `ev_ret2` fluctuating at most 2x epoch to epoch against measured surges of 14x-250x.
- Canonical sources `patches/src/sb_shared_policy.cuh` (three kernels and the call in the decoder backward), `patches/pufferlib_core.diff` (flag, buffers, telemetry) and `config/smart_buildings.ini`; working copies `results/study_r58/sources/`. r57's `value_trunk_cap` stays in the binary at its default 0, so this arm's only active factor against r56 is the guard. The r57 buffer is a pass over `grad_value_h` (B*H), not over the role gradients, so the guard is nearly free.
- Telemetry, fixed for r57's gap: per epoch the bite fraction, the applied scale (mean and minimum), the `||g_V||` geometric level and the histogram of `log2(||g_V||/M/q)`. `SB_VSG` per epoch, TensorBoard `norm/vsg_bite_frac`, `norm/vsg_scale_mean`, `norm/vsg_scale_min`, `norm/vsg_ratio_mean`, `norm/vsg_ratio_max`, `norm/vsg_gv_geo`, `norm/vsg_frac_lt_q`, `norm/vsg_frac_ge_8q`, and `results/study_r58/vsg.jsonl`.
- Arm `r58_value_spike_guard`: r56's config plus that flag, from scratch, Fairbanks, seed 73, 80,216,064 steps, sixteen saves, GPU 0. Started 2026-10-25 05:53 UTC; banner `value-spike guard on (q 3.00 x the trailing value-tangent scale, trunk value tangent only)`, first checkpoint and Muon sidecar 1,667,456 bytes. Gate `results/verify_20261025_110318.log` ended ALL CHECKS PASSED; `policycheck` 108+108+60+60 finite differences, 0 failures. Driver `results/study_r58/run_r58.py`. The first logged epochs read bite 0.000, s_mean 1.000, r_mean about 1.0-1.4 and r_max at most 2.6: inert below the threshold, as constructed. Sixteen fp64 months run after training.

### 2026-10-24 — r56 neutral half heat, 80,216,064 steps
- `r56_zone_command_decode` keeps `train.freeze_return=1` and `env.zone_command=1` with that decode. Fairbanks, from scratch, seed 73, 80,216,064 steps, sixteen saves, GPU 0. The 19:20 UTC container stopped at nine checkpoints; those files are `results/runs/r56_zone_command_decode_partial`. The 0.70 banner rejection is removed. Restarted from scratch 2026-10-25 00:58 UTC. Gate `results/verify_20261025_003118.log` ended ALL CHECKS PASSED. Banner `zone command on (neutral half heat, pump follows the average, water capped)`, 56/288 active heads. First checkpoint and sidecar 1,667,456 bytes. That process died during epoch 580. The last complete file is 40,108,032. Continuation `r56_zone_command_decode_cont` loaded that file and its Muon sidecar at 2026-10-25 02:12 UTC and trained the remaining 40,108,032 steps in a new directory. It exited 0 at 03:04 UTC. The two segments are separate. Best fp64 month is 0.219 at original 50.1M (productivity −0.269, energy 0.105, carbon 0.185, electricity 13.0 MWh, gas 125 MWh). Heated by 10.0M. That point is under r54's 0.227 and above r55's 0.217, and the next month is 0.582. Not under 0.134. Curve: `results/study_r56/curve.json`.
### 2026-10-25 — r58 read: the guard is inert; r59 credit window proposed, not approved
- r58 live read at epoch 351 of 1088 (25.9M of 80.2M steps, 11:47 local). `SB_VSG` over 325 epochs: bite 0 in 314; the 11 that bit are the EMA warm-up (epoch 5, bite 0.98). After epoch 10, `r_mean` 1.05, `r_max` at most 2.5, `s_min` 1.0, `vsg_gv_geo` 0.43–1.27. So the guard is inert for the whole run, not only its first epochs.
- Over the same epochs `SB_UPDATE` shows `grad_shared`/`grad_actor_heads` going 32× (epoch 25) → 52× (100) → 350× (315), with the trunk's Muon step 5.0× the heads'. The value's hold on the shared trunk is chronic and grows 7×; a guard that normalizes `||g_V||` by its own trailing mean cannot see it, which is r57's lesson from the other side. The arm cannot answer its own question; its only difference from r56 is binary noise.
- State at 25.9M: cost 2.58, productivity −4.47 (0.6×4.47 = 2.68 of 2.76, 97% comfort), comfort deviation 12.4 K, worst zone 41.7 K, out-of-band 0.81, gas 177 MWh, electricity 17.0 MWh, explained variance 0.991, value loss 312, joint sampled KL 8.0, entropy −92 (min −190 at 21M). Under-heated basin, not plant-off.
- Same seed, same flags: r56 and r58 agree to ~5% through epoch 110, then r56 sits at comfort deviation 4–6 K while r58 stays at 14–16 K. Dashboard-to-month calibration from the 33 measured pairs of r54/r55/r56: ≤6 K → 0.22–0.74; 11 K → 2.1; 16–20 K → 1.6–6.3; 24–33 K → 5.4–10.9. Two runs of one recipe differ 4–5× in the month by 20M steps, larger than every effect the five arms since r50 were chasing.
- r59 proposal written to `results/study_r59/PROTOCOL.md` with driver `results/study_r59/run_r59.py` (refuses to start without an `APPROVED` file; nothing was launched). One change from r58: `train.gae_lambda` 0.95 → 1.0, guard off. The effective credit window goes 16.8 steps (84 min) → 100 steps (8.3 h); a comfort bill six hours after the heat enters the advantage at 0.485 instead of 0.012. Binary reuse: r58's two binaries, hash-checked (`ce7d8061…` train, `cd8a87d5…` score), gate `results/verify_20261025_110318.log` ALL CHECKS PASSED; no source change, so no new gate.
- Pre-registered: at the 40.1M and 45.1M saves gas > 200 MWh and productivity > −0.5; best month < 0.174, the current actor's best measured month (r48 sibling, 80.2M). Plant-off past 40M falsifies the credit window as the constraint. Explained variance < 0.5 at 5.0M invalidates the run (λ = 1 makes the value target about 5× larger). 80,216,064 steps, sixteen saves, seed 73, from scratch, GPU 0, about 2 h 5 min with scoring.
- Queued, each needing approval: horizon 576 with λ = 1 (one rollout holds a night and the following morning); a per-zone value whose night target is that zone's future comfort (r53's `r_z` was 0 all night); per-epoch `||g_V||` and `||g_pi||` at the trunk logged separately; and a two-seed 30M control pair to pin the spread before another 80M arm.- r58 stopped 2026-10-25 12:15 IST at epoch 573, 42,254,336 steps, exit 137, at the user's instruction after a read that showed the plant-off drift (productivity −6.42, gas 96 MWh, out-of-band 0.92, comfort deviation 19.8–23.5 K over 36–41M). Eight checkpoints remain, 5,013,504 through 40,108,032, each with its Muon sidecar; no fp64 month was scored. The read was one step early: epochs 569–573 were the heated mode (comfort deviation 10.1, 6.16, 9.54, 8.39, 8.06 K; cost −1.80, −0.99, −1.73, −1.43, −1.39; gas 96 → 195 MWh), and epoch 570 sits in the band that maps to months of 0.22–0.74. That is the third observation of the recipe's bistability (r56 epoch 120, r55 at 45M, r58 at 41.7M), and the transition fell between the 40.1M save and the stop, so those weights are not on disk. Preserved: `results/runs/r58_value_spike_guard/`, `results/study_r58/`.
- r59 started 2026-10-25 06:50 UTC on GPU 0 as approved by that instruction. Banner: no value guard, Muon, frozen return, advantage normalization, zone command (neutral half heat), sibling block, 56/288 heads, no loaded weights, `gae_lambda=1`, `value_spike_guard=0`, 8.35 GB, 13K SPS. Visit-0 value-row gradient 11.5 against the heads' 4.7 (r58: 2.2 against 8.0), which is the larger λ = 1 target. The first launch failed on a missing `config/default.ini` because the runner copied only the binary; the runtime tree is copied now, and the empty directory is kept as `results/runs/r59_credit_window_lambda1_badconfig`. Sixteen fp64 months follow training.- r59 at 5.9M steps (epoch 80): score 2.87 → 0.82 between 4.7M and 5.9M, gas 55 → 138 MWh, comfort deviation 7.1 K, value loss 52,349 (3.7M) → 228 (5.9M), trunk/head ratio 10.9×. The 5.0M explained-variance gate fired (0.006); recorded as a waiver in the PROTOCOL addendum with four new stop conditions, because the loss falls 230× while the diagnostic goes negative under `freeze_return=1` and the policy is heating from the first millions.- r59's first process was cut by a host reboot, not by a training failure: `last -x reboot` shows 12:53 IST (also 06:25 and 07:25), docker came back 12:53:13, and the container went with the boot. `train.log` ends mid-dashboard at 14.6M steps with no `status.json`; the last complete save is 10,027,008 with its sidecar. The policy was healthy at the cut (score 0.49, comfort deviation 5.1–6.1 K, gas 203 MWh).
- The user set the budget at 50,135,040 total. `results/study_r59_resume/run_r59_resume.py` continues from that save as `r59_credit_window_lambda1_cont` for 40,108,032 local steps with `--base.load_model_path=/res/runs/r59_credit_window_lambda1/ckpt/0000000010027008.bin`. Gate line present and clean: `loaded optimizer momentum from …/0000000010027008.bin.mb`; eight saves on the 68-epoch grid, original 15,040,512 through 50,135,040. Segments reported separately; splice point original 10,027,008; ten months on the 5,013,504 grid. Launched detached with `setsid` so a shell or harness death cannot take the container with it.- r59 finished: 50,135,040 steps over two segments (host reboot at 14.6M; continuation 10,027,008 → 50,135,040). Ten fp64 months: 0.620, 0.492, 0.446, 0.314, 0.517, 0.522, 0.515, **0.237**, 7.618, 5.421. Best 0.237 at 40.1M is a single checkpoint between 0.515 and 7.618 and is not held; median 0.516, mean 1.670. Comfort/fuel split of the best: 0.121 / 0.116 on 290 MWh of gas and 6.7 MWh of electricity.
- The credit window is confirmed as a timing lever: the 5.0M month is 0.620 on 99 MWh where r56 was 1.581, r54 4.068 and r55 9.551, and 20.1M has the best comfort term any arm has measured at that step (0.135). It is not a month lever: 10M–35M burn 410–498 MWh at a 0.30 comfort term (r56's best: 125 MWh at 0.161), the r55 65.2M delivery signature, and 45.1M falls into plant-off (gas 40 MWh, comfort 7.600).
- Frontier: no arm since r48 has entered comfort < 0.10 with fuel < 0.10. r48's 0.134 (0.069, 0.065) dominates r56 (0.161, 0.058), r54 (0.101, 0.126), r55 (0.101, 0.116) and r59 (0.121, 0.116); the arms trade along the frontier, they do not cross it.
- Scoring: the launcher was killed by a harness restart during the scoring pass, so the last two months were scored afterwards by `results/study_r59_resume/score_all.py`; the continuation names files by local step (original = 10,027,008 + local), and the first pass mislabelled six months before that was corrected. `results/study_r59_resume/curve.json` is the corrected ten-point grid.





- 2026-10-25. `r59_lambda1_logstd_const` — r59's λ=1 base + `policy.freeze_logstd=1`
  (arming: `checkpoint_optim=0`, weights-only saves), from scratch, seed 73, Fairbanks,
  65,175,552 steps on the 68-epoch grid (13 saves), then 13 fp64 months, GPU 0.- r61_mod_band finished (exit 0, 65,175,552 steps, 13 saves) and its pre-registered mechanism gate failed: sustained pins at 6.8–10.9M (4.2M steps, gas 337/399 MWh) and 12.4–15.5M (3.2M, 411/417) both exceed the 250 MWh bound, so the 2 K proportional band does not bind a saturated policy.
- Months: 1.522, 0.296, **0.287@15.0M**, 0.560, then plant-off 8.385 → 7.667 across the last nine saves; final − best +7.380, max 1M pinned share 1.00, entropy range 249 nats. Best comfort term of any arm (0.112) at 3× r48's fuel (412 MWh gas).
- Measured baseline row in our format (`study_r61_mod_band/baseline_row.json`): fixed_setpoint 0.1491 (comfort 0.033, fuel 0.116, gas 253), pid_water 0.2139 (comfort 0.070, fuel 0.144, gas 295). r48's 0.134 beats both on fuel.
- Register extended with the Sutton/Javed/Silver programme: P28 per-feature step sizes with optimization (SwiftTD), P29 update bound + decay, P30 plasticity maintenance (L2-to-init + redo), P31 let the agent rank its own features (OaK steps 2–4), P32 options, P33 plant-authority bound as fallback. Wave plan in PRIORS.md §8: C1/C2 on both GPUs now (flag-only), then one build carrying C14 (P28+P29) and C15 (P30).- r62 started 2026-10-25 22:06 IST: the learning-rate anneal, base r54's exact recipe, one change `anneal_lr` (arm 1, cosine 1e-4 → 0; control 0), seeds 73 and 74 each with its own control, 65,175,552 steps, 13 saves, 13 fp64 months per run, decision on the final checkpoint.
- Parity before launch: the r61 binary on r54's 50.1M checkpoint reproduces r54's month exactly (0.2267760469, all components identical), so `zone_command=0` gives r54's map — no rebuild, no gate.
- Restarted once for scheduling; the partial directories are kept as `r62_*_restart1`. Seed 73's pair on one card, seed 74's straddles both.- r62 result: **refuted**. Arm finals 8.583 (s73) and 4.653 (s74), both far above their own best (3.119, 0.285); controls 0.363 and 7.210. Arm final mean 6.618 vs control 3.786. The control reproduces r54 at 50.1M to 0.001 (0.2259 vs 0.2268) and then drifts to 0.363 by 65.2M. The anneal had spent only 6% of the step size by 10M, where the two seeds had already split into opposite basins (11.19 venting vs 0.282 heating) — a schedule applied after the basin is chosen only freezes it. C1 closed; C14 (per-feature step sizes, bound, decay) moves up.



  Approved by the user; protocol `complete_port/results/study_r59_logstd_const/PROTOCOL.md`.
  No code change (existing, never-tested flag); covers under the r59 gate
  `results/verify_20261025_110318.log`. Tests the r59 collapse anatomy: freezing σ removes
  the entropy-bonus churn that precedes every observed storm.
- 2026-10-25. `r59_lambda1_logstd_const` stopped at 56.4M steps (epoch 777), 10:44 UTC, on
  the pre-registered M-gate substance; verdict **premise refuted** — no fp64 months scored,
  the 5.0M–55.1M checkpoints stand for diagnostics only. The live curve entered the extinction
  regime from ~13–15M (KL 2.5e-7–1e-5, `logit_outside` 0.9999, `loss/entropy` pinned −227)
  into a permanent full-hot pin: productivity −0.21…−0.30 against gas 1,234–1,428 kW
  (~900–1,030 MWh/month), stationary ≥5M steps. The same-seed twin fingerprint (r59, identical
  except frozen logstd): bit-identical to 0.9M, plant-off by 5M (prod −5.5…−6.5, gas 30–56 kW),
  then the full-hot pin by 10M — both extreme pins inside 10M steps. σ-creep refuted as the
  necessary cause of the extinction regime; the regime is structural to the dead location-map
  chain and r48-style re-entry evidently needed a live logstd channel. This motivated r60
  (squash amplitude 3→6). Checkpoints preserved in `results/runs/r59_logstd_const/`;
  study evidence in `results/study_r59_logstd_const/`.
- 2026-10-25 r60_alive_mean launched: squash_location amplitude 3→6 (chain `1−(loc/6)²`,
  logit_outside rail raw>6), r59 base with frozen σ reverted, seed 73, 65.2M, GPU 0.
  ppo_math reference updated to amplitude 6 (80/80); verify_20261025_164809.log ALL CHECKS
  PASSED. Study dir results/study_r60_alive_mean/ (PROTOCOL.md, APPROVED, driver).
- 2026-10-25. Corrections to the `r59_lambda1_logstd_const` stop record, read from the TB
  series: the M-gate's entropy-climb branch DID fire (entropy −231 → −67 at 25.1M, with
  later excursions to −83 at 44.4M and −160 at 53.2M), and the run re-entered the hot rail
  without a live logstd channel (gas 870–883 MWh / `logit_outside` ~0.86 at 50.1–50.3M,
  984–999 MWh / 0.96–0.99 by 55.4–56.4M, after a 15–57 MWh plant-off detour through 30–45M).
  "r48-style re-entry required a live logstd channel" is refuted; the frozen-σ policy still
  wobbled between attractors, so σ-creep and σ-freeze are both retired as handles and the
  wobble is structural to the decode's authority shape (Experiments.md has the full series).
- 2026-10-25. r61_mod_band proposed; NOT yet approved. One factor on the r60 base: the
  zone-command decode opens coil+damper together proportional to u and caps the water fill
  at a 4 K demand reset band (constants from the shipped pid_water: REHEAT_FULL_K 2.0,
  zone_damper_full_k 2.0, RESET_FULL_K 4.0), open-loop only. Mechanism gate (saturation
  windows hold gas ≤ 250 MWh) is read before any fp64 month; parity vs r59's months at
  5.0/10.0/20.1M; two-sided band gates; target a sub-0.1475 month held two saves.
- 2026-10-25. Protocols agreed and written for `r61_mod_band`:
  `complete_port/results/study_r61_mod_band/PROTOCOL.md`. Single factor: the zone-command
  decode goes proportional (coil and damper open together by u, the 0.5 split and
  coil-first ordering removed) with the water fill capped by a 4 K mean-deficit demand
  reset (constants from the shipped pid_water: 2.0 K valve/damper bands, 4.0 K reset).
  Open-loop only. Metric bar equivalence to REFERENCE.md is measured at an oracle
  pre-flight (pid_water + fixed_setpoint decompositions, positive-step count included),
  not argued. Gates in read order: mechanism (sustained saturation ⇒ gas ≤ 250 MWh, read
  before any fp64 month), parity vs r59's months, too-tight (worst_dev_K > 5 with
  gas < 100 MWh → widen in the next arm), too-loose (gas > 500 with comfort ≤ 0.10, read
  beside comfort_dev_K), target a sub-baseline month held two saves. Independence stated:
  r60 owns the chain/amplitude (live), r61 runs amplitude 3 and owns the authority band;
  frozen σ retired. Not launched; queued after r60_alive_mean's verdict.

- 2026-10-25 (r61 build + pre-flight). Pre-flight oracle row for r61_mod_band
  written: pid_water 0.21392 (elec 40.30 MWh, gas 295.0 MWh), fixed_setpoint 0.14910
  (elec 23.54 MWh, gas 252.9 MWh), zero positive-reward steps in both — the |R| = −R
  equivalence is measured, and the bar for this scenario stood at fixed_setpoint's
  0.1491 in our own pipeline (REFERENCE.md's 0.1475 row reproduces +1.1% relative).
  Canonical sources updated: squash amplitude reverted to 3 in the patch
  (map/chain/rail/banner/check_ppo_math reference), zone command decode rebuilt as
  the joint proportional stroke with the 4 K mean-deficit water reset
  (sb_water_reset_cap replaces sb_water_cap; contract cases updated). Binaries
  sha-pinned in a fresh plr61 worktree; gate verify_20261025_183001.log run on GPU 1;
  launch queued behind r60_alive_mean via the armed waiter (user-approved order).

- 2026-10-25 (r60 verdict, r61 launch). r60_alive_mean finished: best fp64 month
  0.3035 at 50.1M; entry extinction at 5.0M; five-month 0.479 plateau with gas pinned
  404–424 MWh under a ≥0.99 logit_outside window (14–31M); hot-rail re-entry at
  60.2–65.2M (gas 891→988 MWh). The alive-mean map closed the dead-chain family and
  confirmed the fuel-blind pin: r61's decode-authority hypothesis, with amplitude 6
  retired. r61_mod_band launched on GPU 0 at 13:29 UTC after gate
  verify_20261025_183001.log (ALL CHECKS PASSED, ran on GPU 1 by user assignment);
  from-scratch banners confirmed; r61 gates live (mechanism: saturation ⇒ gas ≤ 250
  MWh; parity vs r59; too-tight/too-loose; target < 0.1491 held two saves).
- 2026-10-26 (r63 verdict complete; C14 next). r63_ent_anneal s74 finished (exit 0, 04:11 UTC,
  884 epochs; 13 fp64 months by 04:29 UTC): best 0.2307 @30.1M, final 5.1221 with gas 0.5 MWh
  (fully off), `final − best` +4.891. Mechanism identical to s73 despite the 10× entropy
  premium: entropy −316.6 by 9.5M, logit_outside 0.80 @10M (case B), pin never appeared
  (0/884 points > 0.9). New fact: **30.1M–40.1M is the longest held row in the campaign** —
  three consecutive saves at 0.2307/0.2454/0.2384, all heated (gas 89–174 MWh) — and the run
  then walked off the plateau from inside it, unsaturated (entropy −150…−185, logit_outside
  0.43–0.55 at 40→45M). Seed direction inconsistent (anneal outlived its control on s74, died
  earlier on s73), so C4/P11 is refuted on both seeds. Outcome log and P11 updated; per the
  pre-declared case-B routing, **C14 (P28+P29, per-feature step sizes in the actor's last
  layer) is the next arm**, its gate sharpened by the held row: the win condition is
  `final − best ≈ 0` from a plateau entered by 30M. C14's protocol and APPROVED are still
  outstanding; nothing launched.
- 2026-10-26 (r62 verdict, r63 first leg). r62_anneal_lr: arm refuted (arm final mean
  6.618 vs control 3.786; both arm finals plant-off/venting; no arm beat its control
  on the best month). Positive: control s73 reproduces r54's 50.1M month to 0.001
  (0.2259) — binary, flags and parity hold end to end — but its final month drifts
  to 0.363, so r54's own recipe fails the hold test at the longer budget too, and
  s74's control died at 20M: seed variance now qualifies every single-seed read.
  r63 (C4, ent anneal 0.01 → 0.001) s73: mechanism gate failed (entropy −179 @8M at
  the 10× premium), never pinned, best 0.3756 @30M after a heated rail, then
  plant-off from 35M (final 9.35): **P11 refuted — the collapse is not the standing
  ent_coef.** Dose probe s75 (0.005, host-killed at 46.1M, scored to 45.1M on GPU 1)
  died earlier, by 20M. s74 relaunched from scratch 08:18 IST after two host kills;
  its 13 fp64 months score automatically behind training. Per r63's protocol, C14
  (per-feature step sizes) moves up, pending its protocol and approval. Registers:
  PRIORS.md outcome rows C0/C1/C4 filled; P11 and P16 verdicts recorded.
- 2026-10-26 (C2 launched, GPU 1). r63_vf_clip at 11:54 IST on the user's assignment
  ("RUN THIS IN 3060" — the 3090 belongs to C14): single flag `vf_clip_coef=0.2`
  (r54 base recipe, seed 73, r61 binary hash-pinned, no build, no gate) into
  `study_r63_vf_clip/` (PROTOCOL.md + APPROVED; runner `run_r63_vf_clip.py` adapted
  from the r62 runner, want-list checks the 0.2 flag). Control is the standing
  r62_ctrl_s73 (final 0.363). Training ~2.5 h on the 3060 (~7.2K SPS), then 13 fp64
  months (~15 min). Mechanism read: value-loss spikes should end under 0.2.
  Timeline: C14/r69 lives on GPU 0 (06:11 UTC); r63_vf_clip lands ~14:35 IST;
  r69 lands ~13:05 IST + scoring.

- 2026-10-26 (C14 implemented and launched). r69_lastlayer_stepsize: IDBD (Sutton 1992 Fig. 2)
  per-weight step sizes + SwiftTD's bound/decay (official repo kjaved0/swifttd) on the three
  role readout matrices, flag `train.lastlayer_stepsize`; Muon keeps the rest and the role
  gradient is zeroed after muon's clip norm so the trunk is bit-identical to the control. One
  ifdef defect caught by the gate's plain-build check and fixed. Gate
  verify_r69_20261026_113615.log: ALL CHECKS PASSED (57/57, incl. the new idbd contract case).
  Launched 06:11 UTC on GPU 0 (r62-control recipe, seed 73, 65.2M, 13 saves + 13 fp64 months);
  alpha adapting by epoch 21 (0.001 → 0.0079). Driver refuses without APPROVED; verdict gates
  pre-registered in study_r69_lastlayer_stepsize/PROTOCOL.md.
- 2026-10-26 (r70 powered probe; r71 zone credit). **r70 powered probe:** 72 CPU months (24 per regime; white,
  1 h hold, 4 h hold; σ 0.3) on the frozen r62_ctrl_s73@50.1M, measured from the raw draw ε. No credit or regime
  passes |t| > 3. Under white noise, building credit leans toward less reheat (t −2.35 / −2.68), own-zone credit
  does not (+0.81), and the 0.70/0.30 blend keeps most of the lean (−1.52). The C18 one-month verdict is superseded.
  **r71:** zone critic arm vs matched control, both freeze_return=0, seeds 73/74, r61 binaries hash-pinned, no build.
  - **Training:** all four ran 884 epochs. r71_ctrl_s74 was moved to GPU 0 by user instruction via
    move_ctrl_s74.py; run_r71.py was not edited.
  - **Scoring:** stopped by user order at 20:20 IST. Seed 74's pair has only the 5.0M month; the 10.0M replays are
    partial.
  - **Seed 73 months:** arm best 0.2743 @50.1M, final 0.2774; control best 0.3222, final 0.7504.
  - **Live last-20M means:** arm 0.337 / 0.276, control 0.412 / 0.266.
  - **Zone credit not shown.** All four freeze_return=0 runs had 0 pinned epochs and no second-half plant-off.
    r71_ctrl vs r62_ctrl is a single-flag freeze_return comparison: s74 collapse avoided, s73 over-fires gas.
  - P19 annotated as under test. live_analysis.py / live_analysis.json in the study directory.
  - PRIORS.md corrections the same day: the anneal is a cosine; the family recipe is r54's decode; C2 vf_clip
    "plant-off from ~45M" and "s74 stopped at ~16M" corrected (heated at 45.1M; s74 stopped at 30.6M with six
    saves); r69's "off" months restated with their gas.
- 2026-10-26 20:30 IST (r72 launched, A3 evidence repair). The user's external analysis proposed A1 (λ=1 +
  horizon 576 + zone night target + ev gate). It was not adopted, for three reasons:
  - It stacks four factors.
  - The powered probe shows λ=1 does not flip the building-credit sign (−2.68 vs −2.35).
  - Night r_z still carries discounted Monday comfort through γ.
  Approved instead: seeds 75/76 for fr1 (r62 recipe), fr0 (r71 control) and zone (r71 arm), six runs from scratch,
  with flags identical to the recorded args.txt. fp64 months are scored on the last four saves only.
  `study_r72_seeds/` (PROTOCOL.md, APPROVED, run_r72.py). r72_zone_s75 is on GPU 0 and r72_fr0_s75 on GPU 1, both
  confirmed 15:01 UTC.
- 2026-10-27 (r72 complete). All six runs exited 0. Collapse runs: fr1 2/4, fr0 1/4, zone 0/4.
  - **freeze_return:** not supported (fr0_s76 cold collapse; fr1_s75 collapsed unpinned).
  - **Zone credit:** supported as stability. Finals 0.277 / 0.324 / 0.286 on the scored seeds; one collapsed month,
    zone_s76 at 50.1M, 3.818.
  - **Best month:** fr0_s75 0.1691 @60.2M (gas 137.7 MWh). Nothing under 0.1491 held.
  - PRIORS C20 added and P19 revised. live_analysis.py was rerun over all twelve runs (study_r72_seeds/).
- 2026-10-27 (r72 dynamics). dynamics.py over the twelve runs found three things:
  - Zone collapses are reversible: every zone episode recovered, in 1–46M.
  - fr1 plant-off is absorbing: two runs are flat for 20M at gas 0–21 MWh.
  - Four runs were still improving at 65M (zone_s75, zone_s76, fr0_s76, r71_ctrl_s73) and two were worsening
    (r62_ctrl_s73, r71_zone_s74).
  Leading indicators of a flip 5M ahead: value-loss rise, logit_outside rise, entropy fall (|ρ| ≤ 0.27). The verdict
  is revised from "zone is stable" to "zone makes collapse reversible; 65M ends mid-descent on two of four seeds".
- 2026-10-27 04:41 IST (r73 / E1 launched). r72_zone_s75 extended to 130.4M as one from-scratch process (same binary,
  flags and seed; no budget-dependent schedule). First-half checkpoints are byte-identical to r72 at 5.0–25.1M.
  `study_r73_zone130/`. Months scored on GPU 1 as saves land.
- 2026-10-27 (r74 / E2 built). `train.valve_zone` added to `patches/src/sb_shared_policy.cuh`, `config/smart_buildings.ini`
  and `pufferlib_core.diff` (`pufferl.cu`: zone advantage alone on damper and reheat; aborts without `zone_critic=1`).
  verify_20261027_044522.log ALL CHECKS PASSED. Flag-off parity with the r61 binary is byte-exact (weights and Muon
  sidecar, two epochs). Not launched: four seeds do not fit the 4 h budget alongside E1.
- 2026-10-27 05:23 IST (r74 / E2 launched). No earlier run used `valve_zone=1`, so seed 75 trains from scratch on
  GPU 1, 65.2M, train only (user order, 3 h window). r73 / E1 is also train only; its runner is stopped before month
  scoring by `study_r73_zone130/stop_after_train.sh`.
- 2026-10-27 05:32 IST (plan). E2 (`valve_zone`) marked URGENT. Planned, not run, continuing 2026-10-27 18:00 IST:
  - E2 seeds 73, 74 and 76.
  - E1, zone s75 and s76 from scratch to 130M on the 3090 (same binary and seed; about 5.7 h). Pass: the late descent
    continues and holds, with no absorbing collapse.
  The s75 half is in progress now as r73_zone130_s75 (train only).
- 2026-10-27 ~08:50 IST: the user stopped r73 / E1 (~107.9M, last save 105.3M) and r74 / E2 (~39.6M, last save 35.1M).
  Power is off at home; work moves to CPU analysis on the laptop.
- 2026-10-27 (r75, CPU analysis). The comfort wall is the zone spread. No training episode in five studies beats
  productivity −0.148 at any fuel. Every recorded month applies one valve command to all zones (across-zone std
  ≤ 0.006), and the zone head's own-temperature weights never leave initialization (r73 to 105M, r74 to 35M). See
  PRIORS C23.
- 2026-10-27 (bug). The per-zone reward used the VAV index as a zone index (38/39 wrong on building_train_1; every
  shipped building affected). Fixed in `gpu/smart_buildings.cu` (zV via vav_zone_idx); verify pending. The zone-credit
  results of r51, r71, r72, r73 and r74 are void as tests of zone credit.
- 2026-10-27 (r76). The zV fix is verified: SB_ZCHECK finds 0 wrong zones, the gate `verify_20261027_153152.log` ALL
  CHECKS PASSED, and zone-critic-off parity with the r74 binary is byte-identical. Approved re-test of zone credit:
  zone cell arm `valve_zone=1` against control 0.70/0.30, 80M from scratch, seeds 75/76 (`study_r76_zone_fix`).
- 2026-10-27 18:21 IST (r76 stopped). The host rebooted ~18:14 and killed ctrl_s75 (~43.7M, last save 40.1M)
  and arm_s76 (~50.9M, last save 50.1M); arm_s75 had finished 80.2M. Stop recorded at the user's order. CPU read in
  `results/study_r76_zone_fix/ANALYSIS.md`.
- 2026-10-27 (r76 read). arm_s75 held 0.17–0.20 (training score) at 68–72.5M. Then gas was cut and comfort fell,
  and the value loss spiked at 74.3M. The zone head's own-temperature weight never moved. Trunk gradient tracks value
  loss (corr +0.74 to +0.96 in five runs). Raw occupancy (pool f = 69) carries ~47% of the encoder's input signal.
- 2026-10-27 18:42 IST (r77). New trainer flags, default 0, in `patches/`:
  - `train.trunk_log`: read-only `SB_TRUNK` and `norm/trunk_*`.
  - `train.head_adam`: the zone head on Adam, frozen in Muon.
  - `policy.occ_asinh`: pool f = 69 through asinh.
  - Keys added to `config/smart_buildings.ini`. Flags-off two-epoch parity with the r76 binary is byte-identical.
  - Launched C (3090), then D to 80.2M (3090), and B (3060, after `verify_all --trainer`). A dropped (r41/r42/r57).
  - `results/study_r77_trunk/`.
- 2026-10-27 ~20:00 IST (r77 stopped, audit). Three defects, all fixed:
  1. The `head_adam`/IDBD sum-of-squares reducer had swapped arguments and no grid-stride loop (r69 and r77 C void).
  2. The async zone trace had one slot (a race; the r76 first-visit KL of about 1e-15 shows no effect).
  3. Rollout zone values were never captured (`g_zv_roll` was allocated after graph capture), so the zone value clip
     read stale memory.
  - Plus a plain-build guard outside `#ifdef`.
  - Gate `verify_20261027_200752.log` ALL CHECKS PASSED; audit sanitizer runs 0 errors. `results/study_r78_fixed/`.
- 2026-10-27 (r79, Beta policy). New flag `policy.beta` (default 0, in `patches/` and `config/smart_buildings.ini`):
  - Two outputs per continuous head (α, β = softplus + 1), env action 2x − 1, deterministic = mean.
  - The v outputs are a second decoder block after the usual rows. Their gradient rides `grad_logstd`; `logstd` is
    unused and gets a zero gradient.
  - Beta math in double: lgamma log-prob, closed-form entropy, digamma/trigamma, Marsaglia–Tsang gamma draws.
  - Update diagnostics report the Beta KL.
  - Aborts with tanh_squash, tanh_mean, entropy_current, logstd_anneal, freeze_logstd, plant_pair, head_adam or
    lastlayer_stepsize.
  - `check_ppo_math.py` and `policycheck` gained Beta cases.
  - A Beta checkpoint is 1,772,960 bytes on the zone-critic recipe; evaluate it with `--policy.beta=1`.
  - `results/study_r79_beta/`.
  - r79_beta_s75 result: best fp64 month 0.1221 @40.1M, final 0.1430 @80.2M (`final − best` +0.021). Five
    consecutive saves held in 0.124–0.136 from 50M to 70M. r76's best at the same seed was 0.1784. Gas 109–253 MWh
    after 20M. Trains at 8.3K SPS, against 12.8K for the Gaussian.
- 2026-10-28 (recorder). Under `SB_RECORD` the recorder now also writes:
  - `parts.bin`: float64, per agent per step, the eight `d_rparts` values (productivity, energy_cost, carbon,
    comfort_dev_K, worst_dev_K, out_of_band, elec_kwh, gas_kwh). It comes through the new env hook
    `sb_rec_parts` (`gpu/smart_buildings.cu`), one group only.
  - `conc.bin`: under `policy.beta=1`, α + β per head.
  - On a recorded month, `parts.bin` reproduces `rew.bin` (0.1 × R to 1e-8), `score.json` (1e-7, the float32 episode
    log) and the eval log's fuels.
  - Training kernels are unchanged. Gate `results/verify_20261028_012921.log` ALL CHECKS PASSED.
- 2026-10-28 (speed build). Beta head math is mixed precision: lnB (three double lgamma) and the logp/entropy sums
  stay double, digamma, trigamma, logs and gradient products are fp32 (`beta_*_f` in `algo.cu`); the KL and the
  diagnostics keep the double functions. The Beta sampler draws Marsaglia–Tsang in fp32. `sbs_diag_visit_k` is
  replaced by a row-parallel kernel plus a fixed-order sum; `sbs_zone_stat_k` reduces over 256 threads.
  `check_ppo_math.py` gained fp32 special-function and 800-point head-versus-double checks (986 checks, 0
  failures). Gate `results/verify_20261028_104356.log` ALL CHECKS PASSED. 3090, r79 flags: 8.05K → 11.9K SPS.
  Binaries in `results/speed_beta_20261028/` (fp32 train sha 256cf4c2…, fp64 score sha c3039efd…).
- 2026-10-28 (speed2). `zm_update_k` (`gpu/sb_zone_memory.cuh`) runs one block per env and one thread per zone
  slot; thread 0 adds the building sums in slot order. `sample_logits` runs one warp per agent: lane 0 walks the
  heads (RNG order unchanged) and defers the Beta log-prob, which the warp evaluates; lane 0 sums per-head
  log-probs in head order. Both are bit-exact: a 2.95M training run reproduces the previous build's SB_UPDATE
  log, weights and momentum byte for byte. 3090, r79 flags: 12.0K → 13.3K SPS. Gate
  `results/verify_20261028_124502.log` ALL CHECKS PASSED. Binaries `results/speed2_20261028/final_runtime`
  (fp32, sha 907d0f50…) and `final_score_runtime` (fp64, sha 64b00c9b…).
- 2026-10-28 (exact-r79 proof, not deployed). The speed builds diverge from r79 through three numeric changes of
  the earlier speed build: the fp32 Beta head and Marsaglia–Tsang sampler, and the 256-thread
  `sbs_zone_stat_k` sum (it feeds the zone-advantage scale, `zone_critic=1`). A copy `/data/puffer-work/plr79x`
  restores those three to the r79 code and keeps the two speed2 kernels. Seed 76, r79 flags, 2.95M: all 5,121
  SB_UPDATE lines equal stored `runs/r79_beta_s76`, and `ckpt/0000000002949120.bin` + `.mb` equal the old
  binary's (`runs/bench_r79x_exact`). 3090: 12.2K SPS (old binary 8.05K, speed2 13.3K). Binary
  `results/exact_r79_20261028/runtime` (sha d025266c…); `patches/` unchanged, no gate run.
- 2026-10-28 (r79 review, CPU only). Stored logs and recordings of r79_beta_s75/s76
  (`results/study_r79_beta/review_20261028/`). Four findings:
  1. The Beta head's softplus + 1 floor is a dead corner: floored parameters sit at u ≈ −12…−21. On s76's
     50–55M months the eval damper is exactly 0.5 on 88–99% of occupied steps (uniform in training).
  2. With the oracle's shortfall and the true occupied mask, a fixed per-zone offset could fix 50–74% of the comfort
     loss and one building-wide shift 23–42%; zone identities are constant through the month.
  3. The reheat own-temperature gain is learned with the right sign and was still accelerating at 80M (s75 −0.36);
     the damper's per-zone gain stays ≈ 0.
  4. Both seeds slide into the cold basin identically at 2.5–5.5M with KL 0.002 and clip fraction 0 (energy first,
     comfort from 4M); after escape the stochastic training score is flat.
  A squared-deviation proxy does not track productivity (r −0.2…0.15 on the weekday-hours mask); the oracle shape on
  the occupied mask does (r 0.70–0.84). Jan 2 and Jan 16 are holidays on this month; comfort starts Tue Jan 3.

## 2026-10-28 (r80: Beta floor 0.5)

- New flag `policy.beta_floor` (patches/pufferlib_core.diff, `config/smart_buildings.ini`): α, β = softplus(u) + floor; shapes
  below 1 draw G(k + 1) U^(1/k). At floor 1 the trainer is byte-identical to speed2 (513 SB_UPDATE rows, checkpoints and
  momentum, `study_r80_beta_floor/parity.json`). A first version wrapped the gamma draw in a branch; ptxas then stopped
  fusing d·t into the caller's G1 + G2 FMA and floor 1 diverged at update 1 (`diag/`). The draw now ends in one product on
  both paths.
- `tools/check_ppo_math.py`: floor-0.5 loss cases, head-vs-double at floor 0.5, sampler shapes < 1; numeric references
  integrate after t = s^(1/a) (the density is singular at a < 1). 1833 checks, 0 failures.
- Power cut 16:06 IST: killed the first gate attempt and the control r79_beta_speed2_s76 at 12/14 saves (not resumed).
  Gate rerun `results/verify_20261028_161103.log` ALL CHECKS PASSED; r80_beta_floor05_s76 started 10:59 UTC on GPU 0.
- `results/stop_all_runs.sh` stops every driver and puffertank container on both GPUs.
- r80 result: best 0.1121 @65.2M, final 0.1626, held 1 (alternating months from 55M); 30–60M mean 0.171 vs control
  0.140. Floor moved, not removed (damper β on the 0.5 floor 65–97% of occupied steps). No early cold basin on this seed.
  Review of all five Beta runs (old s75/s76, speed2 s75/s76, r80): `study_r80_beta_floor/review_r79_r80/`
  (curves_all, floor reads, holding.txt, instability_vs_month.txt).
- r81 (n = 3 per arm, 60.2M): floor 0.5 loses (30–60M mean 0.403 vs 0.199; lower in 1 of 3 pairs; basin 2/3 vs 3/3).
  Floor 1 stays the default. Best month on record r81_floor1_s77 0.0939 @50.1M, held 1. Entropy < −100 marks the
  cold basin in all 8 Beta runs (`entropy_vs_state.txt`). AGENTS.md: "Let the agent look into every possibility".

## 2026-10-28 (r82: entropy target, built and gated, awaiting launch approval)

- New flag `train.ent_target_on` (patches/pufferlib_core.diff, `config/smart_buildings.ini`, off by default): once per
  epoch on the host, c <- clamp(c * exp(ent_coef_lr * (ent_target - H)), ent_coef_min, ent_coef_max), starting at
  ent_coef; H is the epoch's mean loss/entropy read as the change of the loss accumulator across `train_impl`.
  Prints `SB_ENTFLOOR epoch H coef_used coef_next`, TensorBoard `loss/ent_coef`. Gate
  `results/verify_20261028_221109.log` (57 PASS, 0 FAIL) ALL CHECKS PASSED. Study `results/study_r82_ent_target/`.
- r82 launched 17:01 UTC (s75 GPU 0, s77 GPU 1; s76 stopped before start by user order). Windup at the 1e-6 floor,
  then 22 / 37 epochs at the 0.05 cap with H < −100 (cap rule tripped, values finite). s75: 30–60M mean 0.148 vs
  control 0.265, but cold to 20M (10 cold bins vs 5). Total entropy hid narrow plant heads (pump, water) behind
  uniform dampers; escape follows the pump head (`review/role_entropy.py`).
- r82_ent80_s77 stopped by user order at ~55.1M. r83 (`train.ent_role_on`: per-role zone/AHU/plant coefficients on
  mean entropy per head; new loss slots `LOSS_ENT_*`/`LOSS_NH_*`, TB `loss/ent_head_*` on every build) launched
  2026-10-28 19:25 UTC, seeds 75/76 GPU 0, 77 GPU 1. Gate `results/verify_20261029_003405.log` ALL CHECKS PASSED;
  `tools/check_ppo_math.py` now extracts `ent_head_role` and passes three coefficients.
- r83 landed (target −1.0, 3 seeds): 30–60M arm 0.198 vs control 0.199, no difference; basin 3/3 with plant
  entropy held ≥ −2. Recordings of all 12 Beta runs: warm needs dampers open AND recirculation at the 0.7 cap;
  cold is dampers shut, or dampers open on outside air (`study_r83_ent_role/review/cold_signatures.txt`).
- r84: `policy.beta_cn_tau` (AR(1) normals as the first Marsaglia–Tsang proposals in `sample_logits`; per-buffer
  latent state, `cn_state_init`). `check_ppo_math.py` gains `beta_cn_check`. Gate
  `results/verify_20261029_061946.log` ALL CHECKS PASSED. Launched 01:08 UTC, seeds 75/76 GPU 0, 77 GPU 1.
  Result (Fairbanks, 3 seeds): 30–60M fp64 mean 0.110 vs control 0.199, lower in 3/3 pairs, no month ≥ 1 in any
  seed (controls 3/3), cold training bins 5 vs 23; held 5/1/1. s77 is two segments (user-approved resume from
  40.1M with momentum, `runs/r84_cn12_s77_seg2`); its second segment's stdout log stops at 1.8M (driver banner
  check killed the wrapper, the container finished), TensorBoard and the checkpoints are complete.
  Transfer eval (126 fp64 months, six other scenarios): seed-mean finals 0.13–0.35, comfort-limited in warm months.
- r85 (2026-10-29): four-scenario batch (`artifacts/sets/r85_four`: b1/fairbanks_1, b4/denver_2, b5/atlanta_6,
  b2/phoenix_6), `env.groups=4`, 64 envs each, r84 binary and flags otherwise; `study_r85_four/` (`run_r85.py`
  trains, `score_r85.py` scores saves as they land beside the trainer, `summarize_r85.py`). Four groups of 64 train
  at ~8K SPS on the 3090 and ~4.5K on the 3060 (r84's one group of 256: ~13K).

## 2026-10-29 r86 counterfactual (eval only)

`results/study_r86_counterfactual/`: `SB_FORCE=<logit col>:<env value>` in a copy of the r80 score tree
(`/data/puffer-work/plr86cf`, diff `src/sb_force.diff`; not in `patches/`) holds named continuous heads after the
sampler and records the executed value. 24 months on r85_four_s75 80.2M. Hotter water is worse in all four scenarios
(overheating; reheat does not back off enough), so the water head's bottom setting is not the defect. r85's losses to
pid_water are energy and carbon (2–16×) with equal or better comfort; occupied supply air sits at ~13 °C in every
climate with reheat 0.6–0.84.
Joint/recirculation follow-up (same day): holding every AHU's recirculation at the 0.70 cap lowers the score in all
ten scored scenarios (mean 0.0629 → 0.0527, comfort unchanged). r85's recirculation head starts at 0.55–0.82 and is
walked to its floor in every climate by 30–60M while its concentration grows; r84 on Fairbanks alone held it at cap.

## 2026-10-30 r87 credit replay (diagnostic, r85_four_s75)

`SB_CREDIT=64` in `/data/puffer-work/plr87diag` (diff `results/study_r87_credit/src/sb_credit.diff`, not in
`patches/`). The replay was bit-identical to r85 until a power cut at ~17.9M. The resumed segment
(`runs/r87_credit_s75_resume`, from 15.04M, new random stream) is reported separately. On the identical segment the
recirculation head's advantage correlation is positive in three of four groups while r85's recirculation mean fell
(phoenix_6 0.55 → 0.24 at 10 → 20M). The same holds for supply air in fairbanks_1 and denver_2. Movement against
the head's own policy gradient points at the shared readout and trunk (`value_shared=1`, where r57 measured the
critic's trunk gradient at 10–50× the policy's), not at credit. `credit_read.txt` holds the table.

## 2026-10-30 r88 readout swap (eval only)

`results/study_r88_swap/run_swap.py`: role[1] of r85_four_s75 swapped between the 10M and 20M saves (layout: encoder,
decoder [value rows, logstd, role0, role1 8×288, role2 6×272], 2 MinGRU layers 768×256 last). The readout carries
the recirculation fall (phoenix_6 0.548 → 0.239; 20M body with 10M readout 0.638, 10M body with 20M readout 0.337;
atlanta_6 likewise). `readout_walk.txt`: the mean direction moves persistently over 5–20M, opposite the r87
advantage correlation. Remaining suspects, in that update path: the PPO clip under the joint ratio, the entropy
term, or the Muon preconditioning on the readout matrix.

## 2026-10-30 r89 update-path replay (diagnostic)

`SB_PPD` in `/data/puffer-work/plr89ppd` (diff `results/study_r89_ppd/src/sb_ppd.diff`, not in `patches/`): two
side passes of `ppo_loss_compute` into scratch (entropy off; entropy and clip off), first-order projection of each
gradient and of the applied Muon step on each AHU/plant Beta mean via the readout input rows, joint-ratio clip
statistics, readouts saved per epoch (`runs/r89_ppd_s75/readout_weights.bin`). Bit-identical to r85 through 30.08M.
The AHU heads' gradients are sign-noise (41–65% negative epochs per bin); supply air, recirculation and water fall
together at 19–22M in all scenarios and lock. The geometric-mean joint ratio clip almost never binds.

## 2026-10-30 r90 per-AHU advantage (`train.ahu_critic`, `env.ahu_critic`)

- Env (`gpu/sb_zone_memory.cuh` `zm_rz_k`): under `env.ahu_critic=1` the per-step return export is CN_ZS + CN_K wide.
  Column CN_ZS + k is AHU k's return: the productivity of its own slots, minus the whole building's energy and carbon.
  When the building is empty it is the building reward; an absent AHU gets 0. `sb_rz_width()` reports the width.
- Trainer (patch):
  - The zone critic's row width is runtime (`g_sbs_zw`). Value rows are 1 + CN_ZS + CN_K.
  - GAE, transpose and value loss run at that stride. The AHU rows are averaged separately, so they weigh one head.
  - `sbs_zone_stat(..., ahu=1)` whitens the AHU rows over present AHUs into `d_astats`.
  - `ppo_loss_compute` gives heads CN_AA + 4k .. + 3 the whitened AHU-k advantage.
  - The SB_CREDIT (r87) and SB_PPD (r89) diagnostics are now in `patches/pufferlib_core.diff`. Both are env-gated and
    read-only. SB_CREDIT uses the AHU advantage for types 0–3 under the flag.
  - Both flags default to 0. The trainer aborts if the env width and the critic rows disagree.
- Gate `results/verify_20261030_123632.log` ALL CHECKS PASSED.
- Result (2026-10-30): rejected; see `results/study_r90_ahu_critic/review/decision.md`.
  - Seeds 75 and 76 reached 30.08M; seed 77 was stopped by the user.
  - The AHU gradients are still noise.
  - The recirculation collapse is gone at 20M, and returns by 30M in s76 on occupied hours.
  - Comfort at 30M is worse than r85 in all 8 pairs.
  - r_k overweights energy by the AHU count, so Σ r_k ≠ R. The flag stays default 0.

## r92 readout Adam (2026-10-30, `results/study_r92_role_adam/`)

- Trainer (patch): `train.role_adam=1` (default 0) runs the head_adam Adam kernel on role[0..2] as one contiguous
  range (asserted), allowed under `policy.beta`; it conflicts with `head_adam`, `lastlayer_stepsize` and `adam=1`.
  Muon gets that range frozen (zero update and momentum). The SB_PPD weight copy now runs before the Adam step, so its
  applied step includes Adam.
- New diagnostic `SB_ROLESTEP=1` (needs `update_diagnostics=1`; read-only, `sb_shared_policy.cuh sbs_rl_*`): per
  epoch and role, Σ‖g‖ pre-clip, Σ‖ΔW‖ applied, ‖W_end − W_start‖, visits.
- The canonical patch also holds r91's `beta_cn_tau_ahu` (another session). With that flag at 0 the sampler is not
  bit-identical to r85 (first-visit advantage differs at 1e-6, a codegen change), so r92's binary is built from r90's
  sources + `src/role_adam_*.patch` only. Parity: `checks/parity.txt` PASSED.

## r95 Bernoulli run heads (2026-10-31, implemented; launched the same day — see the correction below; `results/study_r95_binary_run/`)

- `policy.binary_run` (default 0, needs `policy.beta=1`): heads 183..215 step 4 and 218 are Bernoulli,
  p = σ(u + c0) with c0 = logit P(Beta(floor + ln2, floor + ln2) > 0.25) computed on the host (continued fraction).
  `algo.cu binary_run_head` (lp, entropy, g = {x − p, 0, −z p(1 − p), 0}) is used by `cache_imp_and_v` and
  `ppo_loss_compute` on those slots; the v row gets zero gradient. Sampler: stored 1/0, env ±1, deterministic p > 0.5,
  coherent on iff Φ(z1) < p. Diagnostics: KL Bernoulli, SB_CREDIT m = p, PPD dm/du = p(1 − p). Record: loc executed ±1
  in deterministic months (2p − 1 otherwise), conc = p; `meta.txt` says so.
- `tools/check_ppo_math.py`: loss mode 4 (Beta + one binary head, six cases), c0 against the numerical Beta integral,
  coherent binary marginal and hold. Guarded by `HAVE_BINARY_RUN`, so it still compiles on a tree without the head.
- Merged into the canonical patch by a three-way merge (r90 base, canonical, r95); r91/r92/r94 hunks kept.

### 2026-11-05 — r95 ran; results recovered from the on-disk scorer artifacts; verdict

Two register errors corrected: the heading above said "not run", and PRIORS P41 said "implemented, gate
pending, not run" — both stale. r95 **launched on Modal 2026-10-31 09:12 IST** (approved; Experiments.md has
the launch entry), and the study directory holds a completed scoring pass that was never written up.

- **Run status**: `live/epochs.tsv` logs 360 epochs (~83M of 576M steps, ~13K SPS) and stops — the run is
  **truncated**, no stop record. Exactly one checkpoint (save 1, 57,600,000) was scored: 15 months on
  2026-10-31 07:09–07:31 UTC, full recordings (`scores/r95_train25_s75_attempt1/0000000057600000/`).
  **r96, the protocol's control, never launched** (second Modal account had no payment method).
- **Held-out correction**: `artifacts/sets/train25_holdout_b7/` contains the six gen6-named scenarios, so
  PROTOCOL.md's "plus the existing gen6 held-out set" is wrong — they trained. Only building_train_7's five
  weathers are held out.
- **Save-1 scores vs pid_water** (`study_r95_binary_run/ANALYSIS.md` has the full table): in-batch 10 months
  0.1117 vs 0.0828 (worse, better 4/10); held-out building 7 **0.0246 vs 0.0427, better 5/5 (−42 %)** — the
  strongest unseen-building result on record, confounded by the new 25-scenario batch (r25→r26 precedent);
  all 15: 0.0827 vs 0.0694 (worse). Losses are comfort-heavy in the extreme months (prod −0.23…−0.39);
  phoenix_4 wins at −0.130.
- **Mechanism**: the run switches hold commanded, climate-differentiated states (hot: fan subsets + water off,
  p ≈ 0.01; cold: water on, p ≈ 0.73–0.80) and toggle where toggling pays (302–401 transitions/month in
  fairbanks_1/fairbanks_2/phoenix_6/atlanta_6). AHU recirculation holds high (0.85), aligned with the rc70
  counterfactual. Empty-period run share exceeds occupied-period in fairbanks_2/phoenix_6 (pre-heating).
- **Verdict: inconclusive-positive on mechanism, no attribution to the head, not adopted.** One save, one
  seed, 10 % of budget, no control, batch confound. Follow-ups: pull saves 2–10 from the Modal volume and
  score them; the clean single-factor arm is `r94b + binary_run` vs `r94b` on `r85_four`.

## r94 Beta concentration cap (2026-10-31, `results/study_r94_conc_cap/`)

- `policy.beta_conc_max` (default 0 = off): cap on the Beta concentration α+β, both shapes scaled alike in sampler and
  loss, mean preserved exactly. Applied to r93's recipe (`beta_cn_tau_ahu=48` + `role_adam=1`). Gate: `chain.log` GATE
  PASSED. First launch died at 02:36 (rc 137, externally killed); partial kept as `runs/r94_conc_cap_s75_interrupted_20261031`,
  relaunched from scratch. Amended order 2026-10-31: 50.14M per seed, each seed's 50 fp64 months score on GPU 1 after
  (or, after the user's pipelining, alongside) its training on GPU 0; s77 checkpoints score on the 3060 as they land
  (user order; driver score-only mode).
- Result, s75, all 50 months scored (`scores/s75/`, recorded):
  - Ten-month means 10.03M 0.161 → 20.05M 0.100 → 30.08M 0.124 → 40.11M 0.101 (training-4 0.057) → 50.14M 0.338
    (training-4 0.257); `final − best` +0.237, not held. Control r93-s75 same grid: 0.141 → 0.164 → 0.205 → 0.268.
  - At 40M the training-4 mean beats `pid_water` (0.057 vs 0.091), both fuels down (phoenix_6 elec 78 / gas 10 MWh vs
    r93's 108 / 189; occupied comfort deficit 1.4 K vs 10.9 K); the r91/r93 SAT-lock signature is absent there
    (SAT 293 K, recirc 0.24). Held-out 40M 0.101 vs `pid_water` 0.077 — generalization still behind the controller.
  - Concentration (`conc.bin`, raw, uncapped): ~39 at 30M, ~7.5 at 40M, ~34–40 at 50M. The cap holds the sampler's
    support under 12 during training while the weights still sharpen; raw re-concentration returned with the 50M corner.
  - The 50M corner (all six warm/hot months, only): static pressure pinned 1.85–1.95 kPa, recirculation 0.96, hot
    water 347–348 K, run bits 100% on; elec 100–360 MWh, overshoot counted as bad comfort. Cold months hold to the
    last save (fairbanks_1 0.079 monotone best, denver_2 0.023 flat). Last-10M training log: episode mean 0.072 →
    0.189, total entropy −181 → −235 with `anneal_lr=0` and `ent_coef=0.001` flat.
  - Cause settled (SB_CREDIT sign audit, `review/credit_audit.txt`): **the corner is credit-aligned, not a defect** —
    in the 45–50M window the hot groups' advantage is resolved positive on the very heads that walked up (phoenix_6
    recirc t +3.8, atlanta_6 recirc t +4.0, water +2.5, sp +1.8/+2.0, pump +1.8/+1.4) while the cold groups' credit
    points the other way and their heads stayed low; applied step followed credit sign on 0.62–0.85 of epochs (mean
    0.70 over 48 cells), no-clip descent 0.70–0.93. The desync-1 per-epoch training credit genuinely rewards the
    fuel-heavy commands the full desync-0 month bills for — a local optimum of the estimator. Next arm therefore stays
    **r94b** (r94 + linear lr anneal over the real 50.14M;PRIORS §4), not a credit or exploration prior.
- Result, s77 (checkpoint scored on the 3060 as it landed, driver score-only mode; all 50 months scored, recorded):
  - Ten-month means 10.03M 0.380 → 20.05M 0.174 → 30.08M 0.165 → 40.11M 0.208 (training-4 0.190, HX6 0.220; warm-month
    bump phoenix_6 0.396) → **50.14M 0.0926 (training-4 0.072, HX6 0.106)** — final save is its best; every month
    improved from 40.11M. Only the final save sits at the level.
  - Control r93-s77's scoring leg was interrupted (10M 0.2165 and 20M 0.4167 clean, HX6 1.23 with phoenix_5 2.79 /
    phoenix_4 2.19 / atlanta_5 1.69 — the hot-month lock; 30M 1/9, 40M 3/9, 50M unscored, left as-is). r94-s77 ran
    0.165–0.174 through 20–40M where the control locked.
  - Records, 40.11M → 50.14M: warm months cut water 330/324 → 297/294 K and SAT 295/290 → 281/278 K; cold months raise
    water to 310 K and recirc to 0.59/0.60 — a per-weather re-balanced operating point, not s75's corner (s75 50M on
    the same decode: water 335/334 K, recirc 0.76–0.79). Median head α+β 9.3–12.8; no head pinned at the cap.
  - Pre-registered tests both seeds: lock-at-any-save PASS (s75 20M `atlanta_6`, the only flagged cell, is a 0.048
    near-perfect month), recovery PASS on s77, training-4 below `pid_water` on each seed once (0.057 / 0.072 vs
    0.0911). Holding unmet on both seeds; seeds disagree on which save is best (s75 40.11M, s77 50.14M).


- 2026-11-01, approved and launched per user order: **r94b — the stopping term** (`study_r94b_lr_anneal/`): r94
  recipe exactly + `--train.anneal_lr=1 --train.linear_lr_anneal=1 --train.anneal_steps=50135040` — a linear lr
  anneal 1e-4 → 0 over the real 50.14M (total_timesteps untouched). Source arm: `train.linear_lr_anneal` +
  `train.anneal_steps` added to `patches/pufferlib_core.diff` (default-off; branch never entered without the
  flags, so the binary acts as r94’s when they are absent). Seeds 75/76/77 scratch; per seed:
  train to 50.14M then its 50 fp64 months score on the same GPU (amended 2026-11-01: seeds 75/76 on the
  3090, seed 77 on the 3060 after the r94-s76 arm clears). Same study dir also took **r94's third seed (76)** (`run_r94_s76.py`): seed 76 scratch on GPU 1,
  50 months scored as they land on the same GPU, fresh `bin_s76_train`/`bin_s76_score` copies. Gate:
  `results/verify_r94b_chain.out` (both chains wait on it before launching).

## r96 speed pass and the tile's learning-rate rule (2026-11-01)

Stopped r94b on both cards on the user's order and measured the step instead of guessing. All numbers below are
from 10-epoch runs with r94b's flags and every diagnostic on (`results/study_r96_speed/measure.py`, `sps.tsv`);
`SPS = agents*horizon/(Evaluate+Train)` on the last full board, because PufferLib's own `SPS` field is wrong.

| setting | agents | solve | minibatch | SPS 3090 | VRAM |
|---|--:|---|--:|--:|--:|
| r85_four 4x64 (the old r94b shape) | 256 | fp64 | 2304 | 5,720 | 11.0 GB |
| r96_light_four 4x128 | 512 | fp32 | 2304 | 11,157 | 16.0 |
| r96_light_four 4x160 | 640 | fp32 | 2304 | 11,594 | 19.9 |
| **r96_light_four 4x160 (default)** | **640** | **fp32** | **9216** | **14,044** | **19.9** |
| r96_light_four 4x192 | 768 | fp32 | 9216 | 14,109 | 22.8 |
| r96_light_four 4x192 | 768 | fp32 | 18432 | `cudaGraphInstantiate failed` | >24 |

1. Layout saturates at 4 envs per lane (64 -> 128 per group 1.67x; 128 -> 160 nothing; 160 -> 192 <1% of the
   rollout). Four concurrent groups already put 512 columns in flight, so the startup ">= 160 envs per group"
   warning does not apply to a four-group batch.
2. The recurrent update was starved by its tile: `rows = minibatch_size/horizon` was 8, so a 288-step scan ran 8
   sequences wide. 9216 gives 32 rows, 1.8x on the update, same samples, same `replay_ratio`.
3. The update diagnostics cost 1.17 s/epoch (2.41 -> 1.24 s without `SB_CREDIT`/`SB_PPD`/`SB_ROLESTEP`/
   `update_diagnostics`). They stay; the price is measured.
4. The fp32 training build is 1.25x on the rollout and halves the solve state, which is what buys the wider batch.
   Scoring stays on fp64 (`bin_score` has 0 `EEv5LuDevPf`).

**The trap this pass found.** The tile is a learning knob, not only a speed knob. `steps/sample =
replay_ratio/minibatch_size` and the parameter movement is `steps/sample x lr`. r96 was launched with the fast tile
at r94b's `learning_rate=1e-4`, i.e. 4x fewer optimiser steps per sample, and the policy stopped moving without any
numerical failure: median `loss/kl` 0.0 against r94b's 1.1e-02, `loss/value` stuck at ~3,521 against 47.7, score
panel 1.40 -> 1.81 -> 2.22 while r94b went 2.32 -> 0.26, and fp64 months 0.5896 vs r94b's 0.3254 at matched steps
with energy 5.0x and carbon 4.4x. No NaN, clipfrac 0, logits inside the squash. Stopped at 25.07M. **Rule: never
change `minibatch_size` without matching `replay_ratio` or the learning rate** (mb 9216 needs lr 4e-4; mb 2304 pairs
with 1e-4).

`artifacts/sets/r96_light_four` replaces the batch's largest building, `building_train_2__phoenix_6` (14.84M solve
terms, 1.5x the next), with `building_train_7__phoenix_3` (4.03M): same climate in the slot, not a held-out
scenario, set mean state per agent-step 77.6 -> 56.0 MB (-28%). Three arms now run in the r96 category
(`results/study_r96_light/run_r96_tile.py`): r96a_ctrl_tile2304 (3090, mb 2304 lr 1e-4, 11,594 SPS),
r96b_fix_tile9216_lr4e4 (3090, mb 9216 lr 4e-4, same `steps/sample x lr`), r96c_g1_tile2304 (3060, 5,465 SPS),
each 19,906,560 steps from scratch on r94b's save grid, decided on fp64 months against r94b@20.05M = 0.3254.

## r96 tile A/B: two runs, one factor (`results/study_r96_light/`)

The r96 speed pass produced 14,044 SPS on the 3090 and lost the learning, because the wide tile divides the number of
optimiser steps per collected sample (`steps/sample = replay_ratio/minibatch_size`: 4/2304 = 1.74e-3 for r94b against
4/9216 = 4.34e-4). The parameter movement is `steps/sample x lr`, so with the LR left at r94b's 1e-4 that movement fell
4x and the policy stopped moving without any numerical failure. This is the A/B that asks whether the speed can be
kept by raising the LR to match. Protocol `PROTOCOL_tile.md`, driver `run_r96_tile.py`.

| run | GPU | agents | `minibatch_size` | `learning_rate` | steps/sample x lr | SPS |
|---|---|--:|--:|--:|--:|--:|
| r96a_ctrl_tile2304 | 3090 | 640 | 2304 | 1e-4 | 1.74e-7 | 11,594 |
| r96b_fix_tile9216_lr4e4 | 3090 | 640 | 9216 | 4e-4 | 1.74e-7 | 14,044 |
| r96c_g1_tile2304 | 3060 | 256 | 2304 | 1e-4 | 1.74e-7 | 5,465 |

- **r96a is the control**: r94b's optimiser on the new batch and the new solve. It has to reproduce r94b's curve if the
  tile is the only cause of the collapse.
- **r96b is the treatment**: the wide tile's 1.18x speed with the movement restored by the learning rate rather than by
  more steps per sample. One change against the stopped r96 arm, `learning_rate` 1e-4 -> 4e-4.
- **Both on GPU 0 with the same seed (76)**, in sequence, so the tile and its LR are the only differences. r96c keeps
  the 3060 on the control tile, which its 12 GB and its 5,013,504-step grid were built for; it is not part of the A/B.

Changed against r94b, each measured on its own (`results/study_r96_speed/sps.tsv`): the batch's largest building
(`train_2__phoenix_6`, 14.84M solve terms) replaced by `train_7__phoenix_3` (4.03M, set mean state -28%); layout
4x64 -> 4x160 (5,720 -> 11,594 SPS, the layout saturating at 4 envs/lane); fp64 -> fp32 training solve (rollout
11.08 -> 8.88 s, and half the solve state is what makes the wider batch fit); `minibatch_size` 2304 -> 9216 (update
4.33 -> 2.41 s).

Fixed in both 3090 arms: 19,906,560 steps = 108 epochs = r94b's save grid, from scratch, every r94b flag including the
linear LR anneal to 50.14M, `update_diagnostics=1`, TensorBoard and the `SB_CREDIT`/`SB_ROLESTEP`/`SB_PPD` lines, fp32
train / fp64 score (0 `EEv5LuDevPf` in `bin_score`). Every save is scored deterministic fp64 and recorded on the three
months r94b also trained on, against the measured reference **r94b s76 @ 20,054,016 = 0.3254** (the stopped arm at
18.80M was 0.5896).

Decision: adopt the wide tile if r96b is within 10% of r96a and both within 10% of 0.3254, with median `loss/kl` back
near 1e-2 (r96 read 0.0) and `loss/value` under ~200 by 20M (r96 read 3,521). r96b is stopped if the first five
epochs show `old_kl > 0.1` or `clipfrac > 0.2`; there is no LR warmup knob in this harness. If r96b loses, the tile
stays 2304 (11,594 SPS, 1.95x the r94b shape) and the middle point (mb 4608 + lr 2e-4, same product, ~13,200 SPS) is
the next arm. One seed each, not a leaderboard result.

## r96 split arms: which change cost the learning (2026-11-01, running)

The tile A/B was stopped on the user's order before its treatment produced a board, because three things changed at
once against r94b — the batch, the solve, the tile — and an underperformance could not be attributed. Kept:
`r96a_ctrl_tile2304` completed 19,906,560 steps at 11,081 SPS with four saves; `r96c_g1_tile2304` stopped at 10.03M with
two; `r96b_fix_tile9216_lr4e4` six minutes in with none. Nothing deleted.

Two from-scratch arms now settle it by elimination, **r94b exactly** (256 agents, 4x64, `minibatch_size` 2304,
`learning_rate` 1e-4, replay 4, the linear LR anneal, every flag and diagnostic) with **one factor each**, matched to an
existing r94b arm on the **same card and seed**:

| arm | run | GPU | batch | solve | seed | control | the one change |
|---|---|---|------|---|--:|---|---|
| F | `r96f_fp32_check_s75` | 3090 | r85_four | **fp32** | 75 | `r94b_lr_s75` | the fp32 training solve |
| B | `r96b_batch_check_s77` | 3060 | **r96_light_four** | fp64 | 77 | `r94b_lr_s77` | the batch swap (train_2 -> train_7) |

12,533,760 steps each (170 epochs), 2,506,752-step saves, in parallel, no fp64 months — the read is the live curve,
because every r94b seed has separated by 10-12.5M. Measured from the controls: `r94b_lr_s75` 56.6 / 0.753 at 10M and
185 / 0.355 at 12.5M (value loss / score panel), `r94b_lr_s77` 43.6 / 1.174 and 86.4 / 0.386; a stuck run reads value
loss 1,400-6,700 and a panel of 1.4-2.2, which is what the stopped r96 arm did between 10M and 25M. The driver marks
`OK` or `STUCK` on value loss < 1,000 and panel < 4.

Arm F stuck means the fp32 training solve is the cause, and the fp64 ceiling for this layout is 9,432 SPS at
4x160/2304 against 11,594 on fp32 (a wider minibatch on fp64 is unmeasured). Arm B stuck means the batch swap is the
cause; fp32 stays and the slot takes a different building than train_7. If both learn, both changes are innocent, the
tile is the only remaining suspect, and the minibatch test follows on the setup that reproduces r94b: 9216 at lr 4e-4
plus 2e-4 as a middle point. Protocol `results/study_r96_light/PROTOCOL_tile.md` Part 2; driver
`run_r96_split.py`. One seed each, not a leaderboard result.

## r96 split arms: fp32 cleared, the batch convicted (2026-11-01)

Two from-scratch arms, `r94b exactly` with one factor each, each matched to an existing r94b arm on the same card and
seed, 12,533,760 steps, read on the live curve.

| arm | GPU | the one change | 12.02M value | 12.02M panel | control | verdict |
|---|---|---|--:|--:|--:|---|
| F `r96f_fp32_check_s75` | 3090 | fp32 training solve | **3.2** | **0.382** | 237.9 / 0.305 | **LEARNS** |
| B `r96b_batch_check_s77` | 3060 | batch train_2 -> train_7 | **2,262** | **1.589** | 140.9 / 0.362 | **STUCK** |

**fp32 is cleared.** The first four value boards are identical to the third decimal (0.85 / 17.4 / 246.8 / 1,669-1,672);
a 2e-3 rounding difference then amplifies through the closed loop, the critic lags ~3M steps and arrives better than the
control's (3.2 against 237.9), the policy lands at 0.382 against 0.305, gas descends in lockstep, and fp32 is **7,090
SPS against 5,882 (+21%)**. Consistent offsets: entropy −119 vs −176, `old_kl` 0.065 vs 0.014 — a worse-fitted critic
makes larger advantage-driven steps. It is a perturbation of a chaotic loop, not a systematic breakeven.

**The batch swap is the cause, and it is not the building.** The critic never fits (value 2,496 at 8.0M, 1,857 at 10.0M,
2,262 at 12.0M, against 2,152 / 43.6 / 140.9) and the panel stalls at 1.52-1.68 against 0.36-0.49. Per-scenario at
12.5M: `train_1__fairbanks_1` **26.8 K / −5.963** against the control's 12.3 K / −1.584, while `train_7__phoenix_3`
(4.03 K / −0.294) sits next to the building it replaced (`train_2__phoenix_6`, 4.60 K / −0.170). So train_7 is fine;
the swap tipped Fairbanks into the deep cold basin, and its per-step cost is 24x the next building's. With one shared
value head (`value_shared=1`) the critic's squared error is dominated by that tail, the value loss cannot descend, and
every building inherits the mis-fit: worse critic -> worse policy -> worse comfort -> bigger cost -> harder critic.

**Provenance.** All four buildings are organizer-shipped: `starter-kit/smart_buildings/buildings/` holds exactly
train_1, 2, 4, 5, 7, 8 and nothing else; every generated building this project made (0500, 10001, 30008, 20001, ...) was
produced at run time by `run_floor_generator.py` and is not in the repo. `building_train_7__phoenix_3` is one of the 30
scenarios in `training_set.yaml` (train_7 pairs with atlanta_8, denver_5, fairbanks_7, phoenix_3, singapore_6), so the
swap exchanged one official scenario for another. Nothing here is ours to blame, and nothing is defective.

**Where the score stands** (`r94b_lr_s75`, best save 40.11M, ten scored scenarios): mean **0.0340**, decomposed into
comfort 0.0153 (45%), energy 0.0093 (27%), carbon 0.0094 (28%). Per scenario it runs 0.0102 (atlanta_6) to 0.1022
(fairbanks_1): warm scenarios 0.010-0.026, both Fairbanks months 0.07-0.10. So it is scored in every weather, cold is
5-10x worse, and the best single scenario is still 1.7x the 0.00598 target while the mean is 5.7x it. No single term
carries the gap, which is why fixing one term alone cannot reach a target that needs all three near zero at once.

**The persistent issue.** One value head and one advantage stream for climates whose per-step costs differ by up to 24x,
with unbounded comfort, means the run's fate is set by whichever building lands in the worst basin — 5 of 10 scenarios
under 0.026 on `r94b_lr_s75`, with the mean set by the two cold ones; a stuck critic for all four in Arm B. That is a
property of our estimator, not of the organizer's buildings, and it explains why one recipe reads 0.034 on one batch and
stuck-at-1.6 on another. `adv_norm=2` whitens the advantage per agent block, but the **value target is still raw and
shared** — that asymmetry is the next thing to attack (per-group value normalization or per-group critics), not yet
proposed or approved. One seed each, not a leaderboard result.

### 2026-11-01 — r97, the actor may not read its own command

- Audit of the r94b recordings (`results/audit_r94b_20261101/REPORT.md`, CPU only): zone reheat 0.933/0.935/0.935 on
  cold/in-band/hot and 0.833 at night with across-zone spread 0.005; measured supply water at 311.3–312.0 K while
  occupied on all ten scenarios; 44–55% of the cooling scenarios' electricity at night with the AHU running; the same
  head's own-command weight is 3.0–3.5 in the checkpoint against ≤0.5 on every state column, and `obs reheat(t+1)` equals
  `loc.bin(t)` bit for bit. `r79_beta_s75` (same layout, no `beta_cn_tau`) has +0.009 there and the correct temperature
  sign, which isolates the coherent-exploration sampler as the cause rather than a correlate.
- Source: `[policy] no_cmd_echo` (1 zone valves, 2 every head) in `sbs_local_feature`, read at policy creation in
  `pufferl.cu` so train and eval share it; `sbs_beta_params` applies `d_beta_conc_max` (diagnostics only). No action
  shaping, no reward or observation-width change, no new parameters (443,240), so older checkpoints still load.
- Gate `results/verify_20261101_232028.log` ended ALL CHECKS PASSED; a 64-step eval at levels 0/1/2 on the r94b 40.11M
  checkpoint confirms the mask reaches the forward pass in eval as well as train.
- Launched 18:09 UTC: `r97_noecho_zone_s75` on GPU 0 (level 1, 6.0K SPS) and `r97_noecho_all_s75` on GPU 1 (level 2,
  3.2K SPS), both fp64 `r85_four` seed 75 from scratch, 80,216,064 steps, 16 saves, linear anneal over the full budget;
  the 8 even saves score 80 fp64 months each, recorded, on their own cards. Driver
  `results/study_r97_no_echo/run_r97.py` behind `chain_r97.sh`; protocol `study_r97_no_echo/PROTOCOL.md`; prior P42.

### 2026-11-03 — r108 proposed zone actor

User approved a fresh 50.14M-step seed-75 experiment on the 3090: one 32-unit tanh zone-actor branch plus the affine skip (P43), r94b settings and stored control. The run's protocol, verification, training output, recordings and analysis are under `results/study_r108_local_mlp/`.

- r108 gate `results/verify_20261103_122053.log` failed its new smoke check: branch weights moved, but the diagnostic read gradients after the optimizer cleared them. No r108 run started. Gradient capture now occurs before the optimizer; the full gate remains required.

- r108 initial launch exited before initialization: the frozen runtime was missing `config/default.ini`; no checkpoint or training data was produced. The verified default configuration has been copied into the frozen runtime. The failed run directory is preserved; any fresh relaunch awaits the user's choice.

- r108 retry completed 50,135,040 steps; the full gate passed. The user stopped further scoring after the ten 10.03M months and the Fairbanks 20.05M month. r108 shows a 20–25M training advantage that reversed, then recovered to an inferior late training level; the 20M Fairbanks recording shows much better comfort at 2.6× gas. Deterministic late holding and the registered ten-month goal were not measured. Evidence and next hypothesis: `results/study_r108_local_mlp/POSTMORTEM.md`.

### 2026-11-03 — r109 global valve credit, approved

User approved P44 against the completed r108 control: retain the nonlinear zone actor and zone critic but use the building advantage for the two zone-valve heads. One full `verify_all.sh --trainer` gate is chained to a fresh 50.135M-step seed-75 run on GPU 0; no scoring or parity run. The gate and launch artifacts are in `results/study_r109_global_valve/`.

- r109 stopped on direct user instruction at 34.65M logged steps. Six checkpoints and all logs are preserved. Median 30–34.65M training panel 0.354 vs r108 0.131 and r94b s75 0.113; deterministic result unmeasured.

### 2026-11-03 — r110 conditional raw-action likelihood, approved; not launched

User approved a fresh seed-75 r94b-based 3090 experiment and selected PPO likelihood on the raw sample before the existing action clamp. CPU references check the conditional head, action rails, asynchronous-slot prefix and recurrent derivatives. The sampler, replay and prefix backward are not yet integrated in the trainer; no trainer gate or r110 training has run. `results/study_r110_conditional_beta/IMPLEMENTATION_STATUS.md` records the remaining work and the guarded launch refuses until the implementation and its gate exist. Prior P45 remains unmeasured.


### 2026-11-04 — r110 result and evaluation defect

The gate passed and r110 s75 completed 50.135M on GPU 0. The 10 x 30-scenario scoring job ran on GPU 1, but the conditional sampler ignores `base.deterministic=1`; all 300 scored episodes used stochastic actions despite the startup banner. The driver also printed negative ten-panel scores although its summary JSON stores the correct positive cost. Thus no deterministic r110 control comparison is available. Training shows a faster early escape but 40–50M stochastic panel median 0.1181 versus r94b s75/s76 0.0546/0.0597. The recorded analysis, score components, temporal slices, and proposed corrected mean-action evaluation are in `results/study_r110_conditional_beta/review/POSTMORTEM.md`; no additional run has been launched.

### 2026-11-04 — r110 mean-action rescore completed

The user approved the 3090 fp64 marginal-Beta-mean rescore. All ten checkpoints x ten previously defined panel months completed and were recorded. r110 s75 scored 0.06136 at its 20M best, 0.19284 at 30M, and 0.06947 final; r94b s75 scored 0.03400 at 40M and 0.03969 final. The earlier r110 win over r94b at 10–20M was not held. Full curve, P/E/C, fuels, weather, timing, actions, limitations and an unapproved next hypothesis are in `results/study_r110_conditional_beta/review/MEAN_POSTMORTEM.md`.

### 2026-11-04 — r111 approved

The user approved P46, a fresh seed-75, 3090 r110-based run with independent Beta mean/concentration and unchanged reward, physics, coherent likelihood and schedules. The single trainer gate is chained to training and to the ten-checkpoint recorded mean-action evaluation; a gate failure prevents training. Protocol and status are under `results/study_r111_mean_precision/`.

- Gate passed and training completed. Ten recorded panel saves have best 0.04377 at 30.08M and final 0.13366 (+0.08989); only the best save is within 10% of best. Original scoring used both GPU devices: 30–40M are all on the 3090, and other saves are on the 3060 or mixed. The late oscillatory AHU behavior, energy/carbon increase and CPU-only policy reconstruction are documented with component and scenario breakdowns in `results/study_r111_mean_precision/review/POSTMORTEM.md`. The initial shared-pool masking proposal was withdrawn after further analysis; `review/REASSESSMENT.md` records a conditional-sampler episode-reset defect, its limitations, and an unapproved single-factor night-credit proposal based on r94b.

### 2026-11-04 — r112 P47 result

The trainer gate passed, r112 s75 completed 50.135M fp64 steps, and all 55 registered recorded months landed on GPU 0. Its 40/50M panel is 0.09494/0.05898 against r94b's 0.03400/0.03969. Final Fairbanks 1 is 0.13117 vs 0.10345; lower gas and carbon were outweighed by poorer productivity. Reject the measured P47 improvement. Recorded curve, scenario and time slices, credited-reward audit, sampling and prior-run caveats, and an unapproved team-return-baseline proposal are in `results/study_r112_night_credit/review/POSTMORTEM.md`.

### 2026-11-04 — r114 team valve baseline approved

The user approved P48 from `results/study_r112_night_credit/review/POSTMORTEM.md`. One trainer gate is chained to a fresh seed-75 r94b-based 50.135M-step run and the registered 55 recorded fp64 months on GPU 0. Protocol and live status: `results/study_r114_team_baseline/`.

- Gate initially failed because the short Beta smoke retained `tanh_mean=1`; no training started. After correcting the smoke flags, the complete trainer gate ended `ALL CHECKS PASSED` and r114 ran from scratch to 50.135M. Training took 8,059 s, about 6,221 steps/s; ten policy, Muon and team-baseline sidecars are present. At 35/40/45/50M the training episode panel was 0.1722/0.2458/0.2689/0.2649 versus r94b s75 0.1117/0.0539/0.0651/0.0461. After the user said evaluation was not needed, the r114 scorer and its current partial episode were stopped; completed recordings remain intact. Remaining scores were not produced. Training-only comparative evidence: `results/review_r94b_r114_training_20261105/analysis.txt`.

### 2026-11-05 — r115 headwise PPO ratio approved

P49 is the isolated r94b-based seed-75 test proposed in `results/review_r94b_r114_training_20261105/REVIEW.md`. Store behavior marginal likelihood for each active head, clip each head separately with its existing credit, log first and all replay visits by scenario group and role. The coherent sampler's marginal likelihood is approximate. One trainer gate is chained to fresh 50.135M-step fp64 3090 training; no scoring. Protocol and gate/runtime provenance: `results/study_r115_headwise_ratio/`.

The first gate stopped before training: its new smoke completed, but the log assertion misread the diagnostic column heading as a data row. The saved smoke log shows all requested group/role rows and zero nonfinite ratios. The heading and parser were corrected; the original gate output is retained in `gate_failed_20261105_0108/`.

The corrected gate ended `ALL CHECKS PASSED`; r115 trained to 50,135,040 steps on GPU 0 in 7,906 s (6,341 steps/s), with ten checkpoints and sidecars. The training panel at 20/30/40/45/50M was 0.08131/0.06455/0.05695/0.05328/0.05203; r94b s75 was 0.15871/0.06880/0.05394/0.06510/0.04613. r115 learned faster and held a smoother late level but did not beat the final control or the requested target. No scoring was run. Detailed recorded-data analysis: `results/review_r115_r94b_training_20261105/`.

### 2026-11-05 — r116 nonlinear zone actor plus headwise PPO approved

The user approved the single-factor candidate in `results/review_r115_r94b_training_20261105/REVIEW.md`. Fresh r115-based seed 75 with `policy.zone_mlp=1`, four fixed windows, 50.135M fp64 steps on GPU 0, no scoring. An existing trainer gate must pass before automatically launching. Passive telemetry is defined in `results/study_r116_zone_mlp_headwise/protocol.json`.

The trainer gate ended `ALL CHECKS PASSED` and r116 completed 50.135M steps on GPU 0 in 8,368 s (5,991 steps/s); all checkpoints and sidecars landed. The four-window training panel at 40/45/50M was 0.03677/0.03433/0.03308 versus r94b s75 0.05394/0.06510/0.04613 and r115 0.05695/0.05328/0.05203. The 45M and 50M saves met the registered intermediate training gates. Fairbanks remained 0.07649 at final. These are changing-policy training episodes, not frozen-policy or leaderboard scores. No scoring was run. Full learning curves, component/time/action/value slices and an unapproved next hypothesis: `results/review_r116_20261105/`.

### 2026-11-05 — r117 local value residual approved

User approved P51 in `results/review_r116_20261105/REVIEW.md`: fresh seed 75, r116 recipe with one zero-output-initialized shared local zone-value residual, 50.135M fp64 steps on GPU 0, no scoring. One trainer gate chains directly to training. Protocol and status: `results/study_r117_local_value_residual/`. The legacy r116 `.mb` sidecars carry Muon momentum but do not contain the separate role-Adam moments; r117 adds an explicitly versioned, shape-checked `.hadam` sidecar for that optimizer state.

The gate passed and r117 completed in 8,708.6 s (5,757 steps/s); ten weights and both optimizer sidecars landed. Training-only four-window panel at 40/45/50M was 0.04796/0.04837/0.04415 versus r116 0.03677/0.03433/0.03308. It learned faster through 25M but held a worse late level in every window, failing its registered gates. Fairbanks 40–50M live productivity/energy/carbon −0.07887/0.07347/0.16752 versus r116 −0.04596/0.07335/0.17191. The zone residual trained but did not deliver the intended outcome. Full evidence and unapproved next candidate: `results/review_r117_20261105/REVIEW.md`.


### 2026-11-05 — research provenance and r118 approval

For future experiments, record the research paper, what was actually adapted, differences from the paper, and whether the adaptation helped; mark results pending until measured. User approved the `results/review_r117_20261105/REVIEW.md` candidate: fresh r116-based r118, zone GAE/target/normalization refreshed together per pass, 50.135M fp64 on GPU 0, no scoring. Reference: Andrychowicz et al., https://arxiv.org/html/2006.05990, section 3.5 and Appendix B.1; our adaptation keeps fixed block ordering and the existing headwise correlated-Beta method. Benefit is not yet known. `results/study_r118_zone_pass_refresh/protocol.json` records the gate, criteria, and telemetry.


### 2026-11-05 — r118 P52 result

The required trainer gate ended `ALL CHECKS PASSED`; r118 completed fresh fp64 seed-75 training to 50.135M on GPU 0 in 8,680.1 s (5,775.8 SPS), with ten weight, Muon and role-Adam saves. No scoring. Its 40/45/50M training panel was 0.04261/0.04170/0.03681 vs r116 0.03677/0.03433/0.03308. Both registered late panel and Fairbanks gates failed, despite faster 15–25M learning, better Denver, and passed Fairbanks live comfort/energy guards. Phoenix and Atlanta were worse and fuel increased. Per-pass GAE drift was measurable but not shown to be the cause. The Andrychowicz et al. paper's recompute-per-pass/shuffle recommendation was adapted only to fixed-order zone returns and normalization; its r118 adaptation did not improve the late panel. Review and unapproved next single-factor shared nonlinear AHU actor hypothesis: `results/review_r118_20261105/REVIEW.md`.


### 2026-11-05 — r119 AHU actor approved

The user approved the single-factor candidate in `results/review_r118_20261105/REVIEW.md`: fresh seed-75 r116 recipe, opt-in shared nonlinear AHU actor branch, four groups, 50.135M fp64 steps on GPU 0 and no scoring. One trainer gate chains to the training run. Papers and exact adaptation, pending result, telemetry and registered criteria: `results/study_r119_ahu_mlp/protocol.json`.

### 2026-11-05 — Bernoulli-vs-Beta run-switch behavior compared across r94b/r95/r116 (`study_r95_binary_run/BERNOULLI_VS_BETA.md`)

Question: does the Bernoulli run head behave better than the Beta+threshold head it replaces? All recorded
deterministic months parsed identically — r94b_lr_s75 at 5 saves × 10 scenarios, r116 at 50.135M × 4, r95 at
57.6M × 15 (79 months).

- **The threshold pathology does not reproduce in the control lineage**: across 54 r94b + 4 r116 months, the
  run-head mean sits within ±0.05 of the decode threshold for 0.1–2.5 % of occupied head-steps, and **no month
  has any head lame > 50 % of occupied steps**. Both families commit per-AHU states (r94b fairbanks_1: AHUs
  0–3 at ~0.91, 4–8 at 0.00; r95 b7-fairbanks_7: 0–3 at 1.00). The r91/r93 "39/43 months below threshold"
  numbers were lineage-specific (r85/r91/r93, before the concentration cap and anneal).
- **The Bernoulli switches behave correctly**: on the unseen building they pick the same fan subset in all
  five weathers, run water in cold (0.70–1.00) and not in hot (0.00), track gas (water↔gas r = +0.71), and
  pre-heat (phoenix_6 water 0.75 empty vs 0.17 occupied).
- **The one score-relevant behavior difference**: at 10 % budget r95 under-fans its cold training month
  (fairbanks_1 fans on ~11–25 % vs r94b/r116 ~40 %) while burning 114 MWh of gas — the gas-without-delivery
  signature — scoring 0.2775 against 0.1035/0.0755. In the held-out cold month the same heads commit fans
  fully and score 0.041, so the capability is there; the commitment was not, at that budget.
- **r94b curve dynamics**: transitions/head/month 0.3 → 38 → ~30 across 10M→30M→50M; water pre-heat 10/10
  months at 10M → 0/10 from 30M; corr(water_on, gas) +0.27 → +0.99 — the switches become more structured and
  more fuel-aligned as the policy improves.
- **Verdict: the Bernoulli head works mechanically and is unmotivated** — the defect it targets is absent from
  the control lineage's frozen months; as a score lever it is unproven and negative at its single data point.
  Not adopted; flag kept; the clean `r94b + binary_run` vs `r94b` arm remains the only adoption test, with low
  expected value on this evidence.


### 2026-11-05 — r116 selected baseline, second seed and r119 partial result

The user selected r116 as the finalized research baseline. Seed 76 completed 50.135M fp64 steps on GPU 0 in 8,291.9 s (6,046.3 SPS), using the same frozen r116 binary/config and same flags except seed/path. It produced ten weights and Muon sidecars, TensorBoard through the final step, no scoring; neither r116 seed's `.mb` contains its separate role-Adam moments. Seed-76 training panel at 40/45/50M was 0.04410/0.03832/0.03522 vs seed 75 0.03677/0.03433/0.03308 and r94b s76 0.06391/0.05194/0.05404. At final all four seed-76 scenarios beat r94b s76. Both r116 seeds have two adjacent saves within 10% of their final grid best. Four fixed training windows and two seeds do not establish held-out, frozen-policy, Kaggle or 0.005–0.01 success. Full P/E/C, fuels, time/action/value analysis: `results/review_r116_two_seeds_20261105/REVIEW.md`.

The r119 gate passed, then the user stopped it. Its last logged step was 11,649,024 (epoch 157), retaining two checkpoints; at 10.027M the latest-completed training panel was 1.88588 vs r116 s75 0.27344, with Fairbanks 6.82356 vs 0.31081. Value loss was high, but its cause and any recovery after the stop are not established. The Deep Sets/ResNet-inspired AHU branch did not improve the observed early trajectory; its full-budget result is unmeasured.

### 2026-11-05 — opt-in random episode start

`env.random_start=1` selects one of seven 288-step calendar offsets per environment, including its first reset, and disables desync's initial episode truncation. The episode counter remains independent of the weather/calendar index. Calendar draws use the separate `env.random_start_seed`; the original start time and a 30-day episode are retained for the shipped scenarios. This changes the training weather windows and is recorded as P54 in `PRIORS.md`; benefit is unmeasured. No paper was used and no experiment was launched.

Four extended r85 blobs were baked into `artifacts/sets/r85_four_random_start/`; their original 30-day weather, calendar and auxiliary tables match the old blobs. The required trainer gate ended **ALL CHECKS PASSED** in `results/verify_20261105_210656.log`: its initial checks ran on the 3060, and the remaining checks continued on the 3090 at the user's request. `results/verify_random_start_20261105_204042/` includes seven 16-step shifted oracle traces, end-of-window calendar checks, and a two-building CUDA-graph smoke with 64 environments and 8-step episodes. The first attempted gate exposed a test error: it compared occupancy after advance with the preceding interval. The corrected gate compares the occupancy actually used by physics.

The user subsequently requested training. Fresh `r120_random_start_s75` started on the RTX 3090 at 2026-11-05 15:50 UTC with policy and calendar seeds 75, r116's 50.135M-step configuration and the four extended blobs. Its startup confirmed the random-start, desync-disabled, headwise PPO and zone-MLP paths; the first training updates and TensorBoard were enabled. The gated fp64 runtime, exact arguments and scenario checksums are retained under `results/runs/r120_random_start_s75/`. No evaluation was launched; results remain pending.

### 2026-11-06 — r121 complete-checkpoint interruption/resume mechanism test (PASS)

User-ordered mechanism test of `base.full_checkpoint=1` on the r116 control recipe, gate waived by
the user (the previous gate already includes `tools/check_full_resume.py`); RTX 3090, no scoring.
Fresh seed-75 run with r116's exact flags, 15,040,512 steps, `checkpoint_interval=34`, r116's own
50,135,040-step LR anneal. Leg 1 was SIGKILLed (rc 137) immediately after the committed epoch-136
snapshot (10,027,008 steps; 5,945 steps/s). The in-place resume with `--base.load_model_path=latest`
printed `FULL_CHECKPOINT restored … epoch 136 step 10027008`, no cold/truncated/ignored/mismatch
warnings, and ran to 15,040,512 (5,885 steps/s in the resumed leg) with six committed quartets
(`.bin/.mb/.hadam/.full`) and continuous TB events in one run directory. A replay twin, re-resumed
from the same snapshot and killed at the epoch-170 commit (12,533,760), produced sha256-identical
`.bin`, `.mb` and `.hadam` — the post-stop segment reproduced the uninterrupted continuation bit for
bit. This is a full-checkpoint resume, not a partial one: RNG, simulator phase, role-Adam, schedules
and log history all carried, so the two legs are one trajectory and are reported as one. Artifacts:
`results/study_r121_fullresume15m/` (driver, logs, `INTEGRITY.md`, `integrity.json`), run dirs
`results/runs/r121_resume15m_s75/` and `…_twin170/`. The r116 anchor comparison (this run kept
r116's anneal, so its checkpoints are comparable to r116 s75's saves) was not requested and was not
run.
# 2026-11-06 — portable r116-style checkpoint preparation

Added an explicit portable complete-checkpoint mode with device-memory readback
verification, preserving the strict same-device default. The local trainer gate
`results/verify_20261106_125046.log` passed, including uninterrupted versus
same-3090 continuation and restoration from the 3090 to the 3060. The latter
produced different subsequent weights, so cross-GPU bitwise equivalence is not
claimed. Prepared the fresh 30-scenario r122 Modal run and dataset separately;
the L40S run remains pending user approval. Details and constraints are in
`results/study_r122_train30_portable_s75/README.md`.

# 2026-11-06 — Kaggle submission from the r122 checkpoint (step 238,878,720)

`submission/238878720/` holds a self-contained numpy policy for the r116/r122 actor
(80-value pool → 256-wide encoder → two MinGRU layers → zone / AHU / plant readouts with
the shared 32-unit zone branch) plus the 1,814,944-byte checkpoint pulled read-only from
the Modal volume. Beta decode is `2·alpha/(alpha+beta) − 1` with `alpha, beta = softplus+1`
capped at concentration 12; a run command is on above −0.5, the canonical `decode_res_k`
rule.

The flat weight file was pinned from the trainer's registration order and closes exactly:
encoder 256×80, value 91×264, log-std 1×288 (288 = 256 continuous heads + 16 discrete ×2,
the decoder's `output_dim`), zone 4×324, AHU 8×288, plant 6×272, zone MLP 32×324 then 4×32,
two MinGRU 768×256 = 453,736 floats = the file size. Canary: the log-std row is unused under
`policy.beta` and is still exactly −1.0 in both r116 and r122 checkpoints.

Verification: the numpy actor reproduces the GPU's recorded deterministic action on r116's
`building_train_1/fairbanks_1` month to max 1.07e-6 over 8,704 steps; the canonical
observation and the 90×11 zone-memory block built from the official environment's reset
observation match the port exactly (0/983 and 0/990 differing values); a full 8,640-step
month of r116's weights through the official Python simulator scores 0.075465 against the
port's recorded 0.075480; and all six shipped buildings fit the 9×10 canonical slots with
every native action mapped to a distinct slot (`results/study_r122_submission/`).

The r122 checkpoint at 238.9M of 1,500.2M steps scores 0.095508 on that same deterministic
month (productivity +0.0602, energy 0.0901, carbon 0.2068). The official 30-scenario
`pid_water` pass finished: mean score 0.05496 over 30 full months
(`results/study_pid30/`). No kernel was pushed and no leaderboard submission was made.

# 2026-11-07 — five r122 checkpoints on the leaderboard

All five loaded their weights and returned public scores: 89,579,520 **0.01818** (best),
59,719,680 0.01842, 119,439,360 0.02104, 209,018,880 0.02262, 238,878,720 0.02392. Refs 58077977,
58077975, 58077978, 58077981, 58061991. Measured against the training curves (30 scenarios,
windows of 108 epochs): the curve ordered the three early saves above the two late ones, could not
separate the top three (0.0182-0.0210), understated every checkpoint by 0.003-0.007, and its
worst-scenario column ordered the five exactly (lowest worst case 0.08497 -> best score). The
checkpoint choice at this level is the tail of the cold scenarios, which is where the remaining
0.01818 -> 0.00598 gap also sits.

# 2026-11-08 — held-out 90-scenario set: five policies scored and verified

Built the measurement set the five-policy comparison is read on: 5 buildings × 18 held-out weather
windows = 90 scenarios. Buildings span 30–85 zones; two are new from the official floor generator,
`building_train_3` (seed 1018, 70 zones, 7 AHUs, exterior CV 0.520) and `building_train_6` (seed
1012, 43 zones, 6 AHUs, CV 0.260). Windows: `atlanta_4/7`, `denver_1/4`, `fairbanks_3/6`,
`phoenix_2/7`, `singapore_2/5`, `minneapolis_1..8`. 90 blobs in `artifacts/sets/eval5_heldout18/`,
90 configs under `artifacts/gen/heldout18/`. The gate ran on the 3090 itself: 67 PASS, 0 FAIL
(`results/verify_20261107_221257.log`).

r122 s75 @119,439,360 on all 90: **0.027229** (productivity −0.02717, energy 0.02751, carbon
0.02713). Per building: `train_8` 0.014583, `train_1` 0.016841, `train_6` 0.020538, `train_2`
0.030384, `train_3` 0.053797. The new 70-zone building owns the three worst scenarios
(`minneapolis_1` 0.10950, `minneapolis_2` 0.08676, `denver_1` 0.08079), and its gap is mostly
comfort (−0.0604 productivity against `train_1`'s −0.0101), not fuel. The 43-zone new building sits
between the shipped mid-size ones, so "new building" is not one behaviour.

Logging: `SB_RECORD` runs one part per building, 18 weathers merged into one env group (the
recorder refuses more than one group). `obs.bin` 637,009,920 B, `loc.bin`/`conc.bin` 179,159,040 B,
`parts.bin` 9,953,280 B, `rew.bin`/`done.bin` 622,080 B per building = 18 agents × 8640 steps.
Checks on the merged row: 90 unique scenarios; reward rebuilt from `parts.bin` agrees with
`rew.bin` to 3e-8; per-scenario means to 3e-7; α, β > 0; observations finite. One month re-scored
through the official Python simulator: `train_1/fairbanks_6` 0.004812 against the port's 0.00481093.

r116 s75 @50,135,040 **0.030123**, r116 s75 @45,121,536 **0.030705**, r122 s75 @89,579,520
**0.025315** — the best of the five and the save Kaggle also ranked best (0.01818). Per building
(`train_8` / `train_1` / `train_6` / `train_2` / `train_3`): r122 @89,579,520 0.01323 / 0.01603 /
0.01621 / 0.03141 / 0.04970; r116 @50,135,040 0.029150 / 0.022381 / 0.032187 / 0.020539 / 0.046356;
r116 @45,121,536 0.02884 / 0.02212 / 0.03363 / 0.02027 / 0.04866. r122 @89,579,520 wins `train_8`,
`train_1`, `train_6` and `train_3`, and loses `train_2` to both r116 saves; for the two r122 saves
the 90-set order matches the leaderboard order.

`pid_water` finished 90/90: **0.04609** (productivity −0.05429, energy 0.03908, carbon 0.02851),
per building `train_8` 0.03918, `train_1` 0.03538, `train_6` 0.04908, `train_2` 0.03792, `train_3`
0.06889. The learned policy is ahead on every building, and the gap is comfort and electricity
rather than a fuel swap (productivity −0.0543 → −0.0229, energy 0.0391 → 0.0297, carbon 0.0285 →
0.0283).

The new 70-zone `train_3` owns every policy's worst scenarios (best learned 0.0497, `pid_water`
0.0689), while the 43-zone `train_6` is easy for r122 (0.0162): what makes the tail is control
coupling on the larger plant, not the fact that a building is new.

Verification of the finished set: all four merged JSONs report 90 scenarios whose names equal the
set directory and a score equal to the mean of their five 18-scenario parts; every checkpoint's
sha256 was re-hashed from the source file and matches the launch record, and each eval log prints
the zone-MLP + Beta + concentration-cap banner; 20 recordings (4 × 5) are complete at 18 × 8640
steps and `reward` rebuilt from `parts.bin` matches `rew.bin` to 4.6e-7; all 90 `pid_water` months
have 8640 steps and an 8640-row CSV whose recomputed score matches the summary to <1e-6. One month
of r122 @119,439,360 re-scored through the official Python simulator: 0.004812 against the port's
0.00481093. The first chain leg initially pointed at the live run dir, which has no `args.txt`;
the reconstructed flags in `results/study_r122_train30_portable_s75/args.txt` are the ones the
scorer used.

# 2026-11-09 — r139 on the leaderboard: the mid-anneal save wins

r139 (r116 + `adam=1, role_adam=0, beta_cn_tau=0, beta_cn_tau_ahu=0, value_delay=0,
beta_floor=0.5`, 30 scenarios, 960 agents, 199.9M steps annealed to zero, seed 75) was trained on
one Modal L40S and six checkpoints pulled (39.8–199.9M). Two were submitted:

| checkpoint | training panel (mean / worst) | held-out 90 | public score |
| --- | --- | --- | --- |
| 159,252,480 | 0.0129 / 0.0702 | 0.018530 | **0.01750** |
| 199,065,600 | 0.0109 / 0.0545 | 0.016514 | 0.02944 |
| r122 @89,579,520 | 0.0210 / 0.135 | 0.025315 | 0.01818 |
| r122 @119,439,360 | — | 0.027229 | 0.02104 |

`pid_water` on the same 54 months scores **0.043733** (train_4 0.03650, train_5 0.04783,
train_7 0.04687; 11.0 core-hours, 11 workers), against 0.046092 on the 90, so 0.045207 over the 144.
The learned policy is 69% below it. The same baseline was submitted to the leaderboard as a
reference (ref 58184591, kernel `sidharth1743/submission-pid-water`, a verbatim copy of the
starter-kit's `pid.py` + `pid_plus.py` that reproduces the local months).

The table holds each run's best save on both measures. r122 has three further Kaggle rows
(59.7M 0.01842, 209.0M 0.02262, 238.9M 0.02392) that were never scored on the 90-set, so its best
held-out figure is the best of the two evaluated saves; 89.6M is its best on Kaggle and, of the two,
on the 90.

**Both local selectors inverted on the hidden 48.** The 199.1M save is better on the panel and on
the held-out 90 (0.016514 vs 0.018530, better on 88/90 scenarios and all five buildings), yet the
159.3M mid-anneal save is 0.0119 better on the leaderboard and is now the best public entry, ahead
of r122 @89,579,520 (0.01818). For r122 the training-panel worst column ordered the five saves
exactly; that did not transfer across r139's saves, and neither did the 90-set. One hidden scenario
at ≈0.57 on a 48-scenario mean explains the gap: the hidden set has a condition absent from all
144 local months.

On the held-out 90 r139 @199.1M scores 0.016514 against r122's 0.025315 (−34.8%; terms 42/31/27,
all buildings better, both fuels down). A 54-scenario extension to the three shipped buildings
missing from that set (train_4/5/7 × the same 18 windows) is r139-favoured too: 0.009793 vs
0.019217, 54/54. The submission itself was verified on those buildings: the CPU policy reproduces
the 3090 recording on `train_4/minneapolis_1` (0.02557 vs 0.02552) and `train_7/minneapolis_1`
(0.05681 vs 0.05710), and the commit's `atlanta_2` matches the local month (−0.0041 vs −0.00406).

The useful lesson is about selection: within a recipe the training-panel worst tracked the
leaderboard (r122), across recipes it did not (r139), and the mid-anneal save was the one that
generalised.

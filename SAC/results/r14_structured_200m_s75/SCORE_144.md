# r14_structured_200m_s75 — 144-scenario score, recordings, and Kaggle submissions

Updated 2026-11-12. All four pre-registered saves are scored on 90 + 54 = 144 held-out
scenarios, each recorded in the same pass.

Full analysis (per building, per weather, top carriers, time-of-day, fuel split):
[`ANALYSIS_144.md`](ANALYSIS_144.md).

## 144-scenario scores

| step | 90-set | 54-set | **144** | productivity | energy | carbon |
|---|---|---|---|---|---|---|
| **49,972,800** | 0.048599 | 0.032912 | **0.042717** | −0.0535 | 0.0267 | 0.0263 |
| 89,951,040 | 0.056733 | 0.033612 | 0.048063 | −0.0614 | 0.0276 | 0.0284 |
| 119,934,720 | 0.076204 | 0.053307 | 0.067618 | −0.0935 | 0.0289 | 0.0286 |
| 159,912,960 | 0.068675 | 0.053788 | 0.063092 | −0.0888 | 0.0266 | 0.0225 |

Per building (144, mean over its 18 weathers): train_3 and train_6 carry the run
(0.084→0.145 and 0.062→0.078); the other six stay within 0.027–0.064 at every save.

Same 90 scenarios, published references: r11 (50 M, same family) **0.04570**,
r116 @50.1 M 0.03010, r139 @199 M 0.01651. So this family's 50 M point sits at r11's level,
and the 200 M budget made it worse, not better.

## Protocol

One group per building, `graph=0, desync=0, rew_norm=0`, deterministic `tanh(mean)`, driver
`bin/sb_env` (sm_86, fp64 default solve, no `EEv5LuDevPf` symbol → a valid score binary).
Building 1 scored **0.035276** both before and after the recording opcode was added, so the
recording is neutral.

## Recording

`SB_RECORD_DIR=1 tools/score_par.sh <run> 90|54 <step>` writes, per building,
`results/<run>/record{90,54}/<step>/building_train_<b>/`:

| file | dtype | shape |
|---|---|---|
| obs.bin | f32 | steps × agents × 983 |
| loc.bin | f32 | steps × agents × 219 (executed mean, no sampling) |
| rew.bin | f32 | steps × agents |
| term.bin | f32 | steps × agents |
| mask.bin | f32 | agents × mask_width |
| parts.bin | f64 | steps × agents × 8 |
| record.json | — | shapes, part names, dtypes |

`parts` columns: productivity, energy_cost, carbon, comfort_dev_K, worst_dev_K,
out_of_band_frac, elec_kwh, gas_kwh. Verified against the eval JSON per scenario on all 18
scenarios of building 1 (productivity and energy match to 1e-4).

Mechanism: `env/sb_env.cu` opcode `P` copies `sbgpu::d_rparts` (allocated per env) to the host;
`envdriver.parts()` reads it; `score_set.py --record` writes the files. The core's
`sb_rec_parts()` helper is unusable here because it refuses more than one group and every eval
scenario is its own group. No physics line changed.

## Kaggle submissions

| ref | checkpoint | kernel | commit-log reward on `building_train_1/atlanta_2` | public score |
|---|---|---|---|---|
| 58257916 | r14 89,951,040 | `submission-sac-r14-90m` v1 | −0.0221 | PENDING |
| 58259442 | r14 159,912,960 | `submission-sac-r14-159m` v1 | −0.0218 | PENDING |
| 58259449 | Modal 199,895,040 | `submission-sac-modal-200m` v1 | −0.0250 | PENDING |
| (kernel running) | r14 49,972,800 | `submission-sac-r14-50m` v1 | — | not submitted yet |

Every kernel log carries `loaded weights /kaggle/input/datasets/sidharth1743/…-model/weights.bin
floats 36518`, and the rewards are far from the neutral action (≈ −0.30), so the policy acted.

Local oracle month of the 89.95 M weights through the official Python simulator,
`building_train_1/atlanta_2`, 8640 steps: mean reward **−0.023993** (productivity +0.0308,
energy 0.0137, carbon 0.0140), `|a| > 0.95` on 33.7 % of continuous commands, discrete ON 32.6 %.
The Kaggle example run of the same name reads −0.0221; the 0.002 gap is the
`evaluation_scenarios/` instance versus the `training_scenarios/` config.

The policy is the trained actor exactly: `tanh(mu)` of the structured readouts, 219 heads masked
by the contract, no controller, no sampling. Weights are a flat float32 dump of the 12 actor
tensors (`SAC/kaggle/export_weights.py`).

## Caveats

- One seed. The Kaggle public score is one number over hidden scenarios; it is not the 144 score.
- Checkpoint choice was made on the training signal at the user's instruction (a prior under
  AGENTS.md). The 144 numbers show 49.97 M is the better point, and 89.95 M was not.
- `parts` worst_dev_K is the instantaneous worst zone, not the trainer's episode-mean
  `env/worst_dev_K`.

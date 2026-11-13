# SAC: standard (CleanRL) Soft Actor-Critic on the competition contract

This directory is self-contained. **Nothing outside `SAC/` is edited or
regenerated**; the simulator sources under `complete_port/gpu` are only *read*
(the driver `#include`s them).

The arm is `r01_cleanrl_sac_s75`: CleanRL `sac_continuous_action.py` with its
defaults, on the same building batch, seed and step budget as the finalized PPO
baseline `r139_bestof_s75` (r85_four, 256 agents, seed 75, 50,135,040 steps).

Protocol, decision criteria and the declared adaptations: `protocol.json`.
Hyperparameters: `config/sac50m.json`.

## Why there is a driver

The GPU simulator has no Python binding, and PufferLib 5.0 here is PPO-only. So:

```
SAC/env/sb_env.cu      # includes complete_port/gpu/smart_buildings.cu, adds main()
SAC/bin/sb_env         # built binary (puffertank:5.0, compute_86)
SAC/train/envdriver.py # shm + control socket client, numpy views
SAC/train/train_sac.py # the learner (torch, GPU 0)
```

The env writes the canonical policy observation into a host-pinned row every
step and reads the action rows from one, so the two processes share data through
a POSIX shm file: the driver copies actions in and obs/reward/terminal out, and
the control socket only carries 1-byte opcodes.

Contract: 983 observation values (in a 1024-wide row), 219 canonical continuous
heads (stride 272), 288-wide mask, episode 8640 steps, `desync=1`,
`reward_scale=1`, `rew_norm=1`, `canonical=1`, no residual / comfort shaping /
zone command / Beta.

## Build and gates

```bash
SAC/env/build.sh                                          # ~17 s, needs the container
SAC/.venv/bin/python SAC/tools/smoke_env.py               # GATE1
SAC/.venv/bin/python SAC/tools/check_sac_math.py          # GATE2
SAC/.venv/bin/python SAC/tools/check_bridge.py            # GATE3 (recorded rollout replay)
```

GATE1 checks the env contract, the mask (97/135/193/209 active heads), the
present-zone structure per building (39/85/54/79), reset reproducibility, and
that a step replayed through a captured CUDA graph is **bit-identical** to the
direct step.

## Run

```bash
SAC/tools/run_50m.sh r01_cleanrl_sac_s75                  # 195,840 vector steps, ~2h20
SAC/.venv/bin/python SAC/tools/score_months.py --ckpt results/.../ckpt/000010027008.pt \
    --out results/.../scored/10027008.json --gpu 0
```

Checkpoints every 2,506,752 steps (20 saves). The pre-registered scored saves are
10,027,008 / 20,054,016 / 30,081,024 / 40,108,032 / 50,135,040 on the 10-month
fp64 grid built by `score_months.py` from `complete_port/artifacts/sets`.

Metrics in TensorBoard (`results/<run>/tb`): `losses/*` (CleanRL's), `env/*` and
`live/*` straight from the env, `scenario/*` per month during training.

## Rules

- One method change versus r139; the contract, reward, observation and bounds are untouched.
- No parity run, no resume, no checkpoint selection on the training scenario.
- Every reported month carries productivity, energy, carbon and both fuels.

## Results (2026-11-12)

The research arm is the r11 recipe at a 200 M budget: structured per-zone actor (r139 readout),
CleanRL SAC defaults, batch 2048 x utd 2 (4.27 passes per transition), 1 M replay, async, compiled
features, running reward normaliser, linear lr anneal to 0, seed 75, 30 scenarios / 960 agents,
`zone_memory=0`. Four declared adaptations only: masked 219 heads, per-sample entropy target
`-n_active`, `env.rew_norm=1`, terminal at truncation.

| run | where | steps | panel (30 training scenarios) | note |
|---|---|---|---|---|
| r11 | 3090 | 50 M | **0.0259** held over five saves | tanh-Gaussian, the reference point of the family |
| **r15** | 3090 | 50 M | see TB | **the same arm with the Beta head** (floor 0.5, cap 12) |
| r14 | 3090 | 200 M (stopped at 168 M on request) | 0.0281 best, 0.0373 at 168 M | 16 saves kept |
| Modal | L40S | 200 M complete | 0.0291 best, 0.0437 at 200 M | 21 saves, 5.03 h at 11,058 steps/s |

Held-out 144 scenarios (90 `eval5_heldout18` + 54 `eval3_heldout18`), lower is better:

| checkpoint | 90 | 54 | 144 |
|---|---|---|---|
| **r15 Beta 50,135,040** | **0.033818** | **0.022917** | **0.029730** |
| r14 49,972,800 (Gaussian) | 0.048599 | 0.032912 | 0.042717 |
| r14 89,951,040 | 0.056733 | 0.033612 | 0.048063 |
| r14 159,912,960 | 0.068675 | 0.053788 | 0.063092 |
| r14 119,934,720 | 0.076204 | 0.053307 | 0.067618 |

Same 90 scenarios: r11 @50 M 0.04570, r116 @50.1 M 0.03010, r139 @199 M 0.01651.

Conclusions carried by these numbers:

- **the action head was the binding constraint, not the off-policy learner.** At 50 M, same
  seed, data, batch and budget, the Beta head scores 0.033818 on the 90-set and 0.029730 on the
  144, against 0.04570 / 0.042717 for the tanh-Gaussian — 26-30 % better, with carbon halved
  (0.0263 -> 0.0139) and building 6 collapsing from 0.0624 to 0.0261. That is within 12 % of
  r116 (PPO) at the same budget, where the Gaussian arm was 52 % behind it. One factor moved,
  so this licenses "representation alone is not sufficient" and "the Gaussian head was a
  binding constraint"; it does not license "the setup decides";
- the Gaussian arm **peaks at 50 M and degrades after it**; four times the budget is a loss, and
  the loss is entirely `productivity` (comfort), with energy flat and carbon slightly better;
- two independent executions of the arm (3090 and L40S, same seed, same data, same torch) differ
  by ~0.003 on the training panel at matched steps, so panel differences under ~0.005 are not
  evidence, and the CPU/GPU split is not a source of it (the async queue is bounded at 2 vector
  steps);
- `worst_dev_K` at 33.5 K is unchanged across every save (one zone dominates), so it cannot rank
  checkpoints on the held-out set, unlike on the training panel.

Reports: `results/r15_structured_beta_b2048_s75/SCORE_144.md` (the Beta arm),
`results/r14_structured_200m_s75/SCORE_144.md` (scores, recordings, Kaggle),
`results/r14_structured_200m_s75/ANALYSIS_144.md` (per building, weather, time of day, fuel),
`results/CHECKPOINT_SELECTION_train_signal.md` (selection), `kaggle/README.md` (submission path).

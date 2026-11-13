# The SAC arms

The off-policy baseline of the paper. CleanRL's `sac_continuous_action.py` on the same contract,
blobs, 960 agents, 30 scenarios and seed 75 as the PPO recipe, so the only difference between the
reported arm and its ablation is the action head.

Code and how to run it: `../README.md`. One folder per arm, named after the run directory it was
trained in.

| folder | what changed | steps | training panel | 90 months | 54 months | public |
|---|---|---|---|---|---|---|
| `r04_utd4_async_960_1m_s75` | the base: 4 gradient passes, batch 256, 256 agents | 50.1M | 0.081 | | | |
| `r05_utd4_async_960_1m_frozen_s75` | frozen reward normalizer | 50.1M | 0.170 | | | |
| `r06_utd15_async_960_1m_s75` | 15 gradient passes | 50.1M | 0.071 | | | |
| `r07_utd15_ln_960_1m_s75` | 15 passes + LayerNorm | 50.1M | 0.094 | | | |
| `r11_structured_b2048_s75` | structured per-zone actor, batch 2048, **Gaussian head** | 50.1M | 0.026 | 0.0457 | | |
| `r13_structured_zm_s75` | r11 + the environment's zone memory | 50.1M | 0.034 | | | |
| `r14_structured_200m_s75` | 200M budget, annealed, Gaussian head | 199.9M | 0.0281 best | 0.0486 / 0.0567 / 0.0762 / 0.0687 | 0.0336 at 90M | 0.0630 / 0.0775 / 0.0908 |
| **`r15_structured_beta_b2048_s75`** | **r11 with the bounded Beta head**, floor 0.5, cap 12 | 50.1M | 0.0183 at 50M | **0.0338** | **0.0229** | **0.0252** |

The reported SAC arm is `r15`; `r11` is its Gaussian twin and the one-change comparison in the
paper; `r14` is the head and budget ablation. The four `r04`-`r07` runs and `r13` are the design
sequence of Section I.2, read on the training panel only.

## Files in each folder

| file | what |
|---|---|
| `args.json`, `manifest.json` | the run's settings and its start, as written by the trainer |
| `train.log` | the full training log |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `scored90/`, `scored54/` | the held-out scores behind Table 18 and Appendix F.4 (`r11`, `r14`, `r15` only) |
| `ckpt/` | the final save, shipped for `r11` and `r15` (12.4 MB each) |
| `ANALYSIS_144.md`, `SCORE_144.md` | the per-building, weather, time-of-day and fuel analysis of a scored run |

## Reproducing the 90-month score

The scorer needs the baked evaluation blobs, which are git-ignored and built with
`../complete_port/tools/bake_eval_set.sh`; the checkpoints alone are not enough.

```bash
SAC/env/build.sh                                             # the GPU env driver, ~17 s in the container
SAC/.venv/bin/python SAC/tools/score_set.py \
    --ckpt SAC/results/r15_structured_beta_b2048_s75/ckpt/000050135040.pt \
    --set  SAC/artifacts/eval90/building_train_1 --out /tmp/bt1.json --gpu 0
```

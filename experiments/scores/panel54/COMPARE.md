# Buildings 4 / 5 / 7 on the held-out windows — four checkpoints, two recipes

Generated 2026-11-09, completed 2026-11-10. Tests whether the Kaggle gap between the r139 saves is
explained by the three shipped buildings missing from the 90-scenario set. Same 18 held-out
windows as everywhere else, official scenario generator, 54 blobs, fp64 deterministic scoring on
the 3090 with recordings. 2026-11-10 adds the third and fourth checkpoints (r139 @159,252,480 and
r122 @119,439,360) and the `pid_water` baseline on the same 54 months.

## The 54-set ranks the four checkpoints exactly as the 90-set does

| checkpoint | train_4 | train_5 | train_7 | **54-set** | **90-set** | **144-set** | Kaggle |
|---|---:|---:|---:|---:|---:|---:|---:|
| r139 @199,065,600 | 0.00800 | 0.00860 | 0.01279 | **0.009793** | 0.016514 | **0.013994** | 0.02944 |
| r139 @159,252,480 | 0.00923 | 0.00980 | 0.01455 | 0.011194 | 0.018530 | 0.015779 | **0.01750** |
| r122 @89,579,520 | 0.01290 | 0.02520 | 0.01956 | 0.019217 | 0.025315 | 0.023029 | 0.01818 |
| r122 @119,439,360 | 0.01542 | 0.02488 | 0.02195 | 0.020749 | 0.027229 | 0.024799 | 0.02104 |
| `pid_water` baseline | 0.03650 | 0.04783 | 0.04687 | 0.043733 | 0.046092 | 0.045207 | pending |

Lower is better. `144-set` is the mean of the 90 and 54 per-scenario scores.

Both local sets order the four checkpoints identically — 199.1 < 159.3 < r122@89.6 < r122@119.4 —
and both put the two r139 saves ahead of both r122 saves on every one of the three buildings. The
leaderboard agrees on the r122 pair, inverts the r139 pair, and consequently ranks 159.3 first.
So the missing buildings do not explain the inversion either: on `train_4/5/7` the 199.1M save is
better than the 159.3M save by 12.5%, exactly the direction the leaderboard reverses.

## r139 vs r122 on the three missing buildings

| building | zones | r139 @199.1 | r122 @89.6 | vs r122 |
|---|---:|---:|---:|---:|
| `train_4` | 56 | 0.00800 | 0.01290 | **−38.0%** |
| `train_5` | 81 | 0.00860 | 0.02520 | **−65.9%** |
| `train_7` | 38 | 0.01279 | 0.01956 | **−34.6%** |
| **all 54** | | **0.009793** | **0.019217** | **−49.0%** |

r139 is better on **54/54 scenarios**, including the coldest windows (`minneapolis_1` 0.0371 vs
0.0586, `minneapolis_2` 0.0232 vs 0.0438, `denver_1` 0.0188 vs 0.0360, `fairbanks_3` 0.0198 vs
0.0329). The 159.3M save preserves the margin against r122 on the same 54 (0.011194 vs 0.019217,
−41.8%) even though it trails 199.1M within its own recipe.

## Combined picture on every shipped building

The 90-set covers 1/2/8 and the two generated buildings; this set covers 4/5/7. Per building, the
two headline saves:

| building | r139 @199.1 | r122 @89.6 |
|---|---:|---:|
| `train_1` | 0.00936 | 0.01603 |
| `train_2` | 0.00781 | 0.03141 |
| `train_4` | 0.00800 | 0.01290 |
| `train_5` | 0.00860 | 0.02520 |
| `train_7` | 0.01279 | 0.01956 |
| `train_8` | 0.00845 | 0.01323 |
| **six shipped, mean** | **0.00917** | **0.01972** |

So on all six shipped buildings × 18 held-out windows (108 scenarios), r139 is **54% better**. The
two generated buildings (3 and 6) also favour r139.

## The submission is not the explanation

The CPU Kaggle policy reproduces the fresh GPU recordings on the hidden-set buildings:

| scenario | GPU (fp64) | CPU policy | Δ |
|---|---:|---:|---:|
| `train_4/minneapolis_1` | 0.02552 | 0.02557 | +5×10⁻⁵ |
| `train_7/minneapolis_1` | 0.05710 | 0.05681 | −3×10⁻⁴ |

Together with the earlier checks (first canonical action 2.4×10⁻⁷; the kernel commit's
`atlanta_2` −0.0041 vs the local month −0.00406), the submitted controller is the trained one.

## `pid_water` on the same 54 months

Run on the host CPU with the starter-kit's own `PIDWaterResetPolicy` through
`complete_port/tools/score_cpu_pid.py`, 11 workers, one month per scenario, restart-safe. All 54
months are 8640 steps; 11.0 core-hours total.

| | train_4 | train_5 | train_7 | mean |
|---|---:|---:|---:|---:|
| `pid_water`, 54 | 0.036502 | 0.047828 | 0.046867 | **0.043733** |
| `pid_water`, 90 (for reference) | — | — | — | 0.046092 |
| **`pid_water`, 144** | | | | **0.045207** |

Terms on the 54: productivity −0.04864, energy 0.03683, carbon 0.03591. The 54 mean is 5% below the
90 mean, so the baseline is stable across the six shipped buildings; the gap to the learned policy
is not a property of the eval set. Against it:

| checkpoint | 54 | vs `pid_water` |
|---|---:|---:|
| r139 @199,065,600 | 0.009793 | **−77.6%** |
| r139 @159,252,480 | 0.011194 | −74.4% |
| r122 @89,579,520 | 0.019217 | −56.1% |
| r122 @119,439,360 | 0.020749 | −52.6% |

On the full 144-set r139 @199.1 is 0.013994 against `pid_water` 0.045207 (−69.0%); the 159.3M save
is 0.015779 (−65.1%).

The same baseline was also submitted to the leaderboard (kernel `sidharth1743/submission-pid-water`,
submission ref `58184591`) so the hidden 48 has a reference row. The kernel is a verbatim copy of
`pid.py` + `pid_plus.py`; it reproduces the local baseline month on `train_4/atlanta_4`
(0.026300) and the public `train_1/atlanta_2` (−0.0249 vs local 0.02491).

## What the evidence says now

Every local proxy — the training panel (30/30), the 90-set, this 54-set, and the one public
evaluation scenario (r139 0.00406 vs r122 0.00856) — puts 199.1M ahead of 159.3M, and r139 ahead
of r122. The hidden 48 reverses the first comparison by 0.0119 while agreeing with the second. On a
48-scenario mean, 0.0119 is one scenario at ≈0.57, so the inversion is most likely a single hidden
window where the 199.1M save sits in the cold basin and the mid-anneal save does not. Neither local
set contains that window: 144 deterministic months, all of them, favour 199.1M.

Consequence for selection: within a recipe the local sets tracked the leaderboard for r122, and
they did not for r139. A checkpoint from a new recipe has to be picked on the leaderboard, not on
any local panel.

Files: `r139/`, `r139_159m/`, `r122/`, `r122_119m/` (each with `score.json` and per-scenario
recordings), `pid_water/`, and the chain logs `chain.log` (2026-11-09) and `chain54b.log`
(2026-11-10).

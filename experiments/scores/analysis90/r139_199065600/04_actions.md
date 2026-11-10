# 04 — Actions: r139 @199,065,600

The same recording used for the score and time breakdown carries the executed mean action of every canonical head (`loc.bin`) and its Beta concentration (`conc.bin`). This file reads both, per building, and compares with r122 on the identical scenarios. `pid_water` has no action recording, so the comparison is checkpoint-to-checkpoint.

| building | zones | AHUs | native actions | static heads (σ<0.05) | pinned (|a|>0.95 half the time) | median per-head σ | vs 2 setpoints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 3 | 75 | 0% | 0% | 0.22 | 38× |
| building_train_1 | 39 | 4 | 97 | 0% | 0% | 0.21 | 48× |
| building_train_6 | 43 | 6 | 113 | 0% | 0% | 0.23 | 56× |
| building_train_3 | 70 | 7 | 171 | 0% | 0% | 0.27 | 86× |
| building_train_2 | 85 | 9 | 209 | 0% | 0% | 0.22 | 104× |

Coverage is complete: the mapped slots equal `n_action` for every building, the 69 padding slots are exactly zero, and no head is pinned at a rail. **No head is static** (0% in every building): unlike r122, which held the AHU static-pressure setpoint at -0.88 for 97-100% of the month, this checkpoint modulates it, so every mapped head moves.

## What each role does

| building | role | mean | σ | p5 / p50 / p95 | mean abs Δ/step | α+β median | static heads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | damper | -0.11 | 0.24 | -0.58 / -0.04 / 0.18 | 0.008 | 2.8 | 0% |
| building_train_8 | reheat | -0.04 | 0.24 | -0.63 / 0.01 / 0.22 | 0.009 | 2.5 | 0% |
| building_train_8 | SAT | 0.05 | 0.38 | -0.62 / 0.14 / 0.53 | 0.007 | 5.7 | 0% |
| building_train_8 | recirc | 0.31 | 0.28 | -0.12 / 0.28 / 0.76 | 0.004 | 3.7 | 0% |
| building_train_8 | water | -0.80 | 0.37 | -0.96 / -0.91 / 0.36 | 0.005 | 12.0 | 11% |
| building_train_1 | damper | -0.06 | 0.23 | -0.48 / -0.02 / 0.26 | 0.007 | 2.7 | 0% |
| building_train_1 | reheat | -0.05 | 0.23 | -0.56 / -0.00 / 0.18 | 0.006 | 2.4 | 0% |
| building_train_1 | SAT | 0.05 | 0.39 | -0.66 / 0.14 / 0.53 | 0.006 | 5.4 | 0% |
| building_train_1 | recirc | 0.30 | 0.28 | -0.12 / 0.26 / 0.75 | 0.004 | 3.6 | 0% |
| building_train_1 | water | -0.77 | 0.40 | -0.95 / -0.89 / 0.54 | 0.005 | 12.0 | 0% |
| building_train_6 | damper | 0.05 | 0.25 | -0.30 / 0.01 / 0.62 | 0.007 | 2.6 | 0% |
| building_train_6 | reheat | -0.10 | 0.29 | -0.78 / -0.02 / 0.19 | 0.006 | 2.5 | 0% |
| building_train_6 | SAT | -0.04 | 0.41 | -0.75 / 0.04 / 0.49 | 0.007 | 5.0 | 0% |
| building_train_6 | recirc | 0.30 | 0.29 | -0.13 / 0.27 / 0.77 | 0.004 | 3.7 | 0% |
| building_train_6 | water | -0.76 | 0.40 | -0.95 / -0.89 / 0.57 | 0.005 | 12.0 | 6% |
| building_train_3 | damper | 0.16 | 0.31 | -0.17 / 0.06 / 0.83 | 0.007 | 2.9 | 0% |
| building_train_3 | reheat | -0.18 | 0.33 | -0.82 / -0.09 / 0.20 | 0.006 | 2.8 | 0% |
| building_train_3 | SAT | -0.13 | 0.41 | -0.83 / -0.06 / 0.44 | 0.007 | 5.2 | 0% |
| building_train_3 | recirc | 0.30 | 0.30 | -0.14 / 0.26 / 0.81 | 0.004 | 4.0 | 0% |
| building_train_3 | water | -0.76 | 0.38 | -0.95 / -0.89 / 0.47 | 0.005 | 12.0 | 0% |
| building_train_2 | damper | -0.07 | 0.24 | -0.53 / -0.02 / 0.24 | 0.007 | 2.6 | 0% |
| building_train_2 | reheat | -0.12 | 0.26 | -0.73 / -0.03 / 0.14 | 0.006 | 2.4 | 0% |
| building_train_2 | SAT | 0.03 | 0.37 | -0.62 / 0.12 / 0.50 | 0.007 | 4.7 | 0% |
| building_train_2 | recirc | 0.25 | 0.27 | -0.14 / 0.20 / 0.72 | 0.004 | 3.5 | 0% |
| building_train_2 | water | -0.72 | 0.41 | -0.95 / -0.87 / 0.60 | 0.006 | 12.0 | 0% |

This checkpoint's action profile is visibly different from r122's. The zone heads are **hedged**, not decisive: the median α+β is 2.4-2.9 (a broad, U-shaped Beta) against 10.5 for r122; only the hot-water head sits at the cap (12). The actions are also much smoother (mean |Δ| 0.006-0.009 per step against 0.03), the supply-air setpoint sits near neutral (0.05) rather than high, recirculation is moderate (0.25-0.31) and the hot-water command is held **low** (mean -0.72 to -0.80, against +0.44 for r122). That is a different plant strategy, not a small perturbation.

## Run bits and the 18:00 shutdown

| building | AHU run on | AHU flips/day | AHU on-run steps | plant on | plant flips/day | plant on-run |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 39% | 4.9 | 137 | 17% | 0.8 | 116 |
| building_train_1 | 39% | 6.3 | 141 | 16% | 0.7 | 123 |
| building_train_6 | 40% | 9.3 | 146 | 16% | 0.7 | 135 |
| building_train_3 | 41% | 11.2 | 145 | 16% | 0.7 | 142 |
| building_train_2 | 39% | 13.8 | 146 | 17% | 0.8 | 125 |

At 18:00 the AHU run fraction stays at 0.95 (from 0.96 at 17:00), the damper mean changes +0.01 and SAT +0.03. Unlike r122, which shut the air handlers at 18:00 (0.99 -> 0.17) and produced the 18:00 comfort step, this checkpoint keeps them running: file 03's hour table has no step at 18:00. The run bits do cycle during the day (4.9-13.8 flips/day) in 137-146-step blocks.

## The AHUs are one, the zones are not

| building | AHUs | SAT spread | recirc spread | damper spread across zones | reheat spread across zones |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | 3 | 0.004 | 0.001 | 0.27 | 0.25 |
| building_train_1 | 4 | 0.013 | 0.002 | 0.28 | 0.31 |
| building_train_6 | 6 | 0.053 | 0.011 | 0.41 | 0.43 |
| building_train_3 | 7 | 0.087 | 0.010 | 0.39 | 0.49 |
| building_train_2 | 9 | 0.043 | 0.007 | 0.33 | 0.39 |

The AHU heads are nearly identical across air handlers; the zone heads span a wide range, so the zoning is real but it is a modulation on a common mode.

## Effective dimensionality

| building | heads | PC1 | PC1–3 | participation | zone-damper participation |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | 75 | 48% | 79% | 3.5 | 1.9 |
| building_train_1 | 97 | 44% | 75% | 3.9 | 2.4 |
| building_train_6 | 113 | 51% | 81% | 3.1 | 3.0 |
| building_train_3 | 171 | 57% | 85% | 2.7 | 2.3 |
| building_train_2 | 209 | 46% | 82% | 3.3 | 2.0 |

One component explains 44–57% of the action variance and the participation ratio is 2.7–3.9: the policy uses the full action list but behaves as a common mode plus a few zone corrections.

## Behavioural distance from r122 (same scenarios)

| building | all heads | damper | reheat | static | recirc | water |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 0.518 | 0.535 | 0.557 | 0.185 | 0.490 | 1.389 |
| building_train_1 | 0.486 | 0.499 | 0.484 | 0.203 | 0.503 | 1.267 |
| building_train_6 | 0.495 | 0.436 | 0.527 | 0.221 | 0.488 | 1.342 |
| building_train_3 | 0.496 | 0.415 | 0.583 | 0.236 | 0.471 | 1.284 |
| building_train_2 | 0.503 | 0.554 | 0.523 | 0.280 | 0.543 | 1.006 |

r139 is a different controller on the same scenarios: the mean |Δa| is 0.49-0.52 across the role means, the largest gap is the hot-water command (1.0-1.4), and even the static-pressure head is 0.19-0.28 away. That is consistent with a different recipe (Adam, no temporal coherence, floor 0.5), not a small perturbation of r122.

## What this file adds

- Full action coverage proof for this checkpoint, per role, with utilisation, concentration and chattering.
- The 18:00 shutdown in actions, matching the comfort step in file 03.
- Effective dimensionality and the behavioural distance from r122 on the same 90 months.


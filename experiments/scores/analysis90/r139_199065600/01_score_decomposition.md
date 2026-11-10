# 01 — Score decomposition: r139 @199,065,600

The checkpoint 199,065,600 of the r139 recipe, on the same 90 held-out months as the parent analysis. `score = −0.6·productivity + 0.2·energy + 0.2·carbon`; lower is better. Terms are the normalised competition quantities, and the identity was re-checked on all 90 rows (max error 1.0e-07).

| policy | score | productivity | energy | carbon | comfort pts | energy pts | carbon pts | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r139 @199,065,600 | 0.01651 | -0.01165 | 0.02540 | 0.02222 | 0.00699 | 0.00508 | 0.00444 | 42% / 31% / 27% |

Against the two references on the same set: **0.01651 vs r122 0.02532 (-34.8%) and pid_water 0.04609 (-64.2%)**. The paired per-scenario delta is -0.00880 against r122 (median -0.00561; better on 77/90) and -0.02958 against pid.

## Per building

| building | zones | score | productivity | energy | carbon | r122 | vs r122 | pid_water |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 0.00845 | -0.0008 | 0.0186 | 0.0213 | 0.01323 | -36.1% | 0.03918 |
| building_train_1 | 39 | 0.00936 | -0.0009 | 0.0217 | 0.0225 | 0.01603 | -41.6% | 0.03538 |
| building_train_6 | 43 | 0.01265 | -0.0071 | 0.0230 | 0.0189 | 0.01621 | -22.0% | 0.04908 |
| building_train_3 | 70 | 0.04430 | -0.0472 | 0.0473 | 0.0325 | 0.04970 | -10.8% | 0.06889 |
| building_train_2 | 85 | 0.00781 | -0.0022 | 0.0165 | 0.0159 | 0.03141 | -75.1% | 0.03792 |
| **all 90** |  | 0.01651 | -0.0116 | 0.0254 | 0.0222 | 0.02532 | -34.8% | 0.04609 |

Every building beats both references. The hardest remains `train_3` (0.0443) and the largest gain against r122 is `train_2` (0.0078 vs 0.0314).

## Per station (weather)

| station | n | score | productivity | energy | carbon | pid | vs pid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atlanta | 10 | 0.01303 | -0.0093 | 0.0288 | 0.0084 | 0.03161 | -58.8% |
| denver | 10 | 0.01805 | -0.0140 | 0.0210 | 0.0272 | 0.04896 | -63.1% |
| fairbanks | 10 | 0.01843 | -0.0155 | 0.0165 | 0.0290 | 0.04995 | -63.1% |
| minneapolis | 40 | 0.01804 | -0.0125 | 0.0232 | 0.0295 | 0.05195 | -65.3% |
| phoenix | 10 | 0.01121 | -0.0078 | 0.0259 | 0.0068 | 0.02783 | -59.7% |
| singapore | 10 | 0.01577 | -0.0082 | 0.0436 | 0.0105 | 0.04869 | -67.6% |
| **station-equal** |  | 0.01575 |  |  |  | 0.04316 | -63.5% |

The window-weighted mean is 0.01651 and the station-equal mean 0.01575: the same ordering, so the Minneapolis-heavy selection (8 of 18 windows) does not drive the result.

## The tail

| policy | worst-5 | worst-10 | worst-20 | <0.01 | <0.02 | <0.05 | <0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r139 | 20.5% | 35.2% | 58.0% | 47 | 67 | 86 | 90 |
| r122 | 18.7% | 29.6% | 46.5% | 25 | 43 | 83 | 89 |
| pid_water | 15.0% | 25.8% | 42.8% | 0 | 7 | 61 | 85 |

Worst five: building_train_3__minneapolis_1 0.0855, building_train_3__minneapolis_2 0.0596, building_train_3__denver_1 0.0583, building_train_3__fairbanks_3 0.0546, building_train_3__minneapolis_8 0.0468.

## What drives the variance

| building | weather window | interaction |
| --- | --- | --- |
| 74.2% | 22.1% | 3.8% |

The building explains 74% of the score variance and the weather window 22%; the interaction is the remainder (no replication, so it cannot be separated from noise). This checkpoint is more building-dominated than r122 (49/38) and much more than pid_water (19/72): the policy has removed most of the month-to-month sensitivity.

## What this file adds

- The checkpoint scored on the same 90 months, decomposed by building, station, window, tail and term contribution, with the paired deltas against r122 and the baseline.
- Term shares: comfort is the largest contributor, as in every policy on this set.


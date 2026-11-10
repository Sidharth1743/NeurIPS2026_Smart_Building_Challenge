# 05 — Zones: r139 @199,065,600

The comfort term is zero when a zone is unoccupied, so the scored regret is the deficit on occupied steps. The per-zone band deficit is decoded from the canonical observation (`max(Hs−T,0)+max(T−Cs,0)`) and validated against the recorded comfort parts. Free float (empty building) is reported separately; it is not scored.

| building | zones | scored deficit K | free float K | in band (occupied) | in band (all) | top-5 share |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 0.012 | 1.302 | 98.4% | 78.9% | 65% |
| building_train_1 | 39 | 0.011 | 0.514 | 98.3% | 85.7% | 65% |
| building_train_6 | 43 | 0.106 | 0.566 | 84.4% | 76.7% | 57% |
| building_train_3 | 70 | 0.631 | 1.205 | 47.1% | 48.6% | 25% |
| building_train_2 | 85 | 0.027 | 1.045 | 97.4% | 79.9% | 41% |

The scored residual is 0.01-0.11 K on four of the five buildings and 0.63 K on `train_3`; the occupied in-band share is 84-98% on those four and 47% on `train_3`. The free float is an order of magnitude larger (0.5-1.3 K) and is not scored.

## Where it sits

| building | worst zone | median zone | best zone | worst five (AHU/position) |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.08 | 0.00 | 0.00 | AHU2/1, AHU3/7, AHU3/6, AHU3/0, AHU2/4 |
| building_train_1 | 0.06 | 0.00 | 0.00 | AHU4/9, AHU4/5, AHU2/1, AHU4/8, AHU1/0 |
| building_train_6 | 1.11 | 0.04 | 0.00 | AHU2/1, AHU2/2, AHU3/1, AHU6/2, AHU6/1 |
| building_train_3 | 3.78 | 0.51 | 0.03 | AHU4/2, AHU4/5, AHU2/3, AHU4/3, AHU5/6 |
| building_train_2 | 0.32 | 0.01 | 0.01 | AHU9/8, AHU4/0, AHU6/3, AHU3/6, AHU1/3 |

The concentration differs by building: a few zones on the small ones, a broad floor with one or two outliers on `train_3`.

## Per AHU

| building | AHUs | best AHU | worst AHU | worst AHU # | worst/best |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | 3 | 0.01 | 0.02 | 3 | 3.2× |
| building_train_1 | 4 | 0.01 | 0.02 | 4 | 3.3× |
| building_train_6 | 6 | 0.04 | 0.40 | 2 | 10.9× |
| building_train_3 | 7 | 0.37 | 1.25 | 4 | 3.4× |
| building_train_2 | 9 | 0.02 | 0.05 | 9 | 3.1× |

Two named air handlers carry the residual: **`train_6` AHU2 at 0.40 K against 0.04 for the best (10.9×)** and **`train_3` AHU4 at 1.25 K against 0.37 (3.4×)**. The other three buildings are uniform (3.1-3.3× on a residual of 0.01-0.05 K). That is the actionable zone-level finding.

## Does the policy feed the zones that need it?

| building | deficit vs damper | deficit vs reheat | cold share |
| --- | --- | --- | --- |
| building_train_8 | +0.16 | +0.50 | 73% |
| building_train_1 | +0.17 | +0.49 | 70% |
| building_train_6 | +0.70 | -0.35 | 45% |
| building_train_3 | +0.77 | -0.51 | 33% |
| building_train_2 | +0.49 | +0.03 | 63% |

Unlike r122, the deficit now correlates **positively with the damper** (0.16-0.77): the policy gives the weak zones more air, and reheat is mixed (0.50 to -0.51). The cold share is 33-73% depending on the window. So this checkpoint does react to the bad zones on the air side; the residual is not a feeding failure but a distribution/capacity limit on the large plants.

## Same zones as r122?

| building | rank correlation | top-5 overlap | worst r139 | worst r122 |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.86 | 3/5 | 0.078 | 0.380 |
| building_train_1 | 0.74 | 5/5 | 0.064 | 0.510 |
| building_train_6 | 0.80 | 2/5 | 1.114 | 1.118 |
| building_train_3 | 0.94 | 5/5 | 3.777 | 3.876 |
| building_train_2 | 0.85 | 5/5 | 0.318 | 1.414 |

The two checkpoints rank the zone deficits similarly (Spearman 0.6-0.9) but r139 is uniformly lower; the weak rooms are a property of the building, not of the checkpoint.

## Day-level persistence

| building | median day | best day | worst day | worst/median |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.00 | 0.00 | 0.02 | 7.7× |
| building_train_1 | 0.00 | 0.00 | 0.09 | 66.2× |
| building_train_6 | 0.05 | 0.00 | 0.24 | 5.3× |
| building_train_3 | 0.46 | 0.03 | 0.78 | 1.7× |
| building_train_2 | 0.01 | 0.00 | 2.90 | 205.4× |

The deficit is a persistent operating condition, not a single bad day or a start transient.

## What this file adds

- Per-zone and per-AHU decomposition of the scored comfort regret for this checkpoint, with the free-float separation and the cross-checkpoint zone stability.
- A named air handler where the residual is not uniform, and the weak action-deficit link that rules out a simple feeding error.


> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# 07 — Appendix: the r116 recipe on the same 90 scenarios

The results section reports r122. This appendix documents the other recipe in the project, r116, on the identical 90 deterministic months, so the comparison is paired scenario by scenario. Both were scored with the same scorer (fp64, deterministic, `SB_RECORD`) on the same 90 blobs; the only difference is the checkpoint. r116 is not a weaker baseline: it was the selected research baseline, and this is the only place in the project where the same policy recipe was trained at two different scales.

## What actually differs between the recipes

|  | r116 | r122 |
| --- | --- | --- |
| training scenarios | 4 (`r85_four`: train_1/2/4/5) | 30 (`tr23_shipped30`) |
| agents | 256 | 960 |
| steps | 50,135,040 (anneal to 50.1M) | 1,500,180,480 (anneal to 1.5B) |
| checkpoints scored | 45,121,536 and 50,135,040 | 89,579,520 and 119,439,360 |
| policy / optimiser flags | identical | identical |

The diff of the two `args.txt` files is exactly the scale block: `env.groups` 4 → 30, `vec.total_agents` 256 → 960, `train.total_timesteps` and `train.anneal_steps` 50.1M → 1.5B, plus the portable/full-checkpoint flags. Hidden size, layers, Beta parameters, zone MLP, headwise ratio, learning rate, batch and clip are the same. So the appendix is a **data-and-scale comparison at fixed hyperparameters**, not a recipe comparison.

## Headline

| policy | score | productivity | energy | carbon | kg CO₂ (5 buildings) |
| --- | --- | --- | --- | --- | --- |
| pid_water | 0.04609 | -0.0543 | 0.0391 | 0.0285 | 24563 |
| r122@119,439,360 | 0.02723 | -0.0272 | 0.0275 | 0.0271 | 24311 |
| r122@89,579,520 | 0.02532 | -0.0228 | 0.0297 | 0.0283 | 26483 |
| r116@50,135,040 | 0.03012 | -0.0312 | 0.0321 | 0.0249 | 22037 |
| r116@45,121,536 | 0.03071 | -0.0321 | 0.0326 | 0.0247 | 21941 |

r122 `@89.6M` is 0.02532; the r116 pair is 0.03012 and 0.03071. Both recipes beat `pid_water` (0.04609).

## Paired comparison, r116 minus r122@89.6M

| r116 checkpoint | mean Δ | median Δ | σ | scenarios won | sign-test p | paired 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| r116@50,135,040 | +0.00481 | +0.00494 | 0.01427 | 24/90 | 1.1e-05 | [+0.00183, +0.00769] |
| r116@45,121,536 | +0.00539 | +0.00568 | 0.01407 | 22/90 | 1.2e-06 | [+0.00246, +0.00824] |

Positive Δ = r116 is worse. Both r116 saves lose on average, and the sign test is decisive (p ≈ 10⁻⁵–10⁻⁶): r116 wins only 22–24 of the 90 scenarios. That is not noise, but the win is not uniform — see the next table.

## Where r116 wins

| building | r116@50.1M | r122@89.6M | Δ (50.1M) | Δ (45.1M) |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.02915 | 0.01323 | +0.01592 | +0.01561 |
| building_train_1 | 0.02238 | 0.01603 | +0.00635 | +0.00610 |
| building_train_6 | 0.03219 | 0.01621 | +0.01597 | +0.01742 |
| building_train_3 | 0.04636 | 0.04970 | -0.00334 | -0.00104 |
| building_train_2 | 0.02054 | 0.03141 | -0.01087 | -0.01114 |

**r116 is better on the two largest plants** (`train_2` by 0.0109, `train_3` by 0.0033) and worse on the three smaller ones (`train_8` +0.0159, `train_6` +0.0160, `train_1` +0.0064). By station r116 is worse everywhere (Atlanta +0.0062 … Phoenix +0.0097), so the building effect dominates the climate effect: the r116 advantage is a large-plant effect, not a weather effect. Within the 18 windows the best for r116 are the two coldest Minneapolis windows (−0.0036 and −0.0025) and the worst are the hot ones (Phoenix_2 +0.0106, Singapore_5 +0.0090).

## Which term moves

| term | mean Δ (points) | min | max |
| --- | --- | --- | --- |
| comfort | +0.00501 | -0.02864 | +0.03080 |
| energy | +0.00048 | -0.00356 | +0.00592 |
| carbon | -0.00068 | -0.00918 | +0.00148 |

The gap is **comfort** (+0.0050) and a little energy (+0.0005); r116 is actually **better on carbon** (−0.0007). The appendix therefore cannot say r116 is worse across the board: it trades comfort and electricity for carbon, the opposite trade to r122.

## Where in the day

| day part | r122 | r116 | Δ |
| --- | --- | --- | --- |
| weekday night | 0.0049 | 0.0054 | +0.0005 |
| weekday morning | 0.0633 | 0.0835 | +0.0201 |
| weekday afternoon | 0.0585 | 0.0645 | +0.0060 |
| weekday evening | 0.0616 | 0.0695 | +0.0079 |
| weekend day | 0.0012 | 0.0003 | -0.0009 |
| weekend night | 0.0017 | 0.0006 | -0.0010 |

The whole r116 disadvantage is the **weekday morning** (+0.0201); the rest of the weekday adds +0.0144, while r116 is slightly better on both weekend bins (−0.0009, −0.0010). That is the same bin where r122 beats `pid_water` by the largest margin, so the morning is where the two recipes separate.

## The behaviour behind the difference

| role | r122 mean | r122 mean abs Δ/step | r116 mean | r116 mean abs Δ/step |
| --- | --- | --- | --- | --- |
| damper | +0.16 | 0.032 | -0.32 | 0.010 |
| reheat | -0.07 | 0.034 | +0.29 | 0.017 |
| SAT | +0.42 | 0.012 | +0.42 | 0.005 |
| static | -0.88 | 0.002 | -0.81 | 0.002 |
| recirc | +0.68 | 0.006 | +0.38 | 0.006 |
| water | +0.44 | 0.012 | +0.16 | 0.009 |

r116 runs a **closed-damper, reheat-heavy plant**: dampers average −0.32 against +0.16, reheat +0.29 against −0.07, recirculation 0.38 against 0.68, hot water 0.16 against 0.44, and it moves the actuators about three times more slowly (mean |Δ| 0.010 vs 0.032 on the dampers). It also holds the static-pressure head less tightly (−0.81 ± 0.13 vs −0.88 ± 0.03). The two policies are not the same controller at different quality; they are different operating strategies.

- **Action distance**: mean |Δa| between r116 and r122 is 0.61–0.72 on the mapped heads (the two r122 saves differ by only 0.33–0.48).
- **Effective dimensionality**: r116 is even lower-dimensional than r122 (participation ratio 1.8–2.8 vs 3.6–6.5), i.e. it drives the dampers more as one block.
- **Zones**: the two recipes rank the zone deficits similarly (Spearman 0.65–0.98), so they fail on the same rooms — but at different levels.

## The same zones, different levels

| building | r122 scored deficit K | r116 scored deficit K | Δ |
| --- | --- | --- | --- |
| building_train_8 | 0.055 | 0.385 | +0.330 |
| building_train_1 | 0.110 | 0.248 | +0.138 |
| building_train_6 | 0.153 | 0.531 | +0.378 |
| building_train_3 | 0.702 | 0.667 | -0.035 |
| building_train_2 | 0.484 | 0.310 | -0.174 |

The scored zone deficit tells the same story as the score: r116 is better on `train_2` (0.310 vs 0.484) and `train_3` (0.667 vs 0.702), worse on the three smaller buildings. The large-plant advantage is a real comfort improvement, not a scoring artifact.

## Fuel: the opposite trade

| building | elec r122 | elec r116 | gas r122 | gas r116 | kg r122 | kg r116 |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 3.4 | 4.8 | 13.7 | 14.6 | 2830 | 3108 |
| building_train_1 | 5.2 | 6.3 | 17.2 | 16.4 | 3635 | 3604 |
| building_train_6 | 9.0 | 10.6 | 18.9 | 14.3 | 4310 | 3653 |
| building_train_3 | 23.4 | 27.2 | 33.9 | 20.7 | 8484 | 6472 |
| building_train_2 | 9.9 | 9.6 | 34.6 | 23.4 | 7224 | 5200 |

r116 uses **more electricity on four of five buildings and less gas on four of five** (the exception is `train_8`, where both are slightly higher), with gas 14.3–23.4 MWh against 13.7–34.6. Its carbon is lower on four of five (higher on `train_8`). That is the mirror image of r122, which cuts electricity and raises gas on the large plants. Neither is "the" answer: r122 wins the score, r116 wins the physical carbon on the big plants.

## Stability over the month

| policy | mean daily slope | min | max |
| --- | --- | --- | --- |
| r122@89,579,520 | -1.55e-05 | -2.79e-04 | +1.46e-04 |
| r116@50,135,040 | +7.48e-05 | -1.13e-04 | +3.01e-04 |

Both are flat to a few 10⁻⁵ per day; r116 drifts slightly upward on average (+7.5×10⁻⁵/day), r122 slightly downward (−1.6×10⁻⁵). Neither has an end-of-month collapse.

## What this appendix does and does not claim

- **Does**: r116 (the 4-building, 50M-step recipe) is 0.0048 worse than r122 on the same 90 months (paired, sign test p≈10⁻⁵), and the difference is not uniform — r116 is better on the two largest plants by 0.0033–0.0109, worse on the three smaller ones, and better on carbon.
- **Does**: the two recipes differ in operating strategy (closed dampers + reheat + more electricity vs open dampers + low static + more gas), and both win comfortably over `pid_water`.
- **Does not**: make a leaderboard claim. These are 90 deterministic local months, one episode per scenario, and the scenario population is the five buildings and eighteen windows chosen here.
- **Does not**: separate "more data" from "longer training". r122 changes both (4→30 scenarios, 50M→1.5B steps) at fixed hyperparameters; this design cannot attribute the gain to either alone.


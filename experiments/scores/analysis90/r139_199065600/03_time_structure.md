> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# 03 — Time structure: r139 @199,065,600

Same clock for every episode: 8,640 five-minute steps starting Sunday 06:00. Bins: weekday night 19–06, weekday morning 06–12, afternoon 12–16, evening 16–19, weekend day, weekend night. Day 0 is an 18-hour Sunday.

| day part | steps | score/step | pid/step | vs pid |
| --- | --- | --- | --- | --- |
| weekday night | 2844 | 0.0045 | 0.0037 | +20% |
| weekday morning | 1512 | 0.0458 | 0.1280 | -64% |
| weekday afternoon | 1008 | 0.0353 | 0.1043 | -66% |
| weekday evening | 756 | 0.0326 | 0.0974 | -67% |
| weekend day | 1404 | 0.0001 | 0.0061 | -99% |
| weekend night | 1116 | 0.0003 | 0.0061 | -96% |
| **month** | 8640 | 0.01651 | 0.04609 | -64% |

Per-step score by bin. The weekday morning is the largest bill for both policies; the gap to the baseline is carried by the daytime bins, as it was for r122.

## Hour of day (weekdays, mean over buildings)

| hour | score/step |
| --- | --- |
| 00 | 0.0021 |
| 01 | 0.0022 |
| 02 | 0.0032 |
| 03 | 0.0046 |
| 04 | 0.0081 |
| 05 | 0.0227 |
| 06 | 0.0579 |
| 07 | 0.0493 |
| 08 | 0.0462 |
| 09 | 0.0431 |
| 10 | 0.0403 |
| 11 | 0.0383 |
| 12 | 0.0369 |
| 13 | 0.0357 |
| 14 | 0.0347 |
| 15 | 0.0339 |
| 16 | 0.0338 |
| 17 | 0.0334 |
| 18 | 0.0305 |
| 19 | 0.0028 |
| 20 | 0.0009 |
| 21 | 0.0008 |
| 22 | 0.0008 |
| 23 | 0.0008 |

The familiar shape: a pre-occupancy ramp before 06:00, the working day, and a drop after 19:00.

## 30-day trajectory

| building | mean | best day | worst day | slope/day |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.0084 | 0.0001 | 0.0174 | +3.50e-06 |
| building_train_1 | 0.0094 | 0.0001 | 0.0226 | +5.74e-05 |
| building_train_6 | 0.0126 | 0.0001 | 0.0323 | +1.13e-04 |
| building_train_3 | 0.0443 | 0.0001 | 0.0925 | +3.78e-04 |
| building_train_2 | 0.0078 | 0.0000 | 0.0224 | +9.07e-05 |

The mean slope is +1.29e-04 per day, but it is not uniform. `train_3` drifts up +3.78e-04/day (about +0.011 over the month), `train_6` +1.13e-04 and `train_2` +9.07e-05; `train_8` and `train_1` are flat. That is an early-warning signal on the large plants, not a collapse.

## The first 24 hours

| building | day-0 score | gas first 24 h kWh | of which last 6 h | days 1–29 |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.00144 | 59 | 37 | 0.00869 |
| building_train_1 | 0.00073 | 44 | 25 | 0.00966 |
| building_train_6 | 0.00097 | 62 | 43 | 0.01305 |
| building_train_3 | 0.00190 | 213 | 191 | 0.04577 |
| building_train_2 | 0.00196 | 294 | 273 | 0.00801 |

Day 0 is an empty Sunday; the policy heats through it (the pre-occupancy investment) and the comfort bill starts Monday. That is the same delayed-credit structure measured for r122.

## Weekends and peaks

| building | weekend kWh share | elec p99 | elec max | mean abs Δ elec |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.6% | 2.2 | 8.3 | 0.010 |
| building_train_1 | 0.6% | 4.0 | 9.8 | 0.013 |
| building_train_6 | 0.6% | 6.4 | 13.9 | 0.020 |
| building_train_3 | 0.4% | 15.1 | 25.1 | 0.045 |
| building_train_2 | 0.6% | 6.4 | 28.1 | 0.032 |

Weekends are 29% of the steps and carry 0.6% of the electricity, a real setback.

## What this file adds

- The checkpoint's time profile (month, day, day-part, hour) with the baseline beside it; the delayed-credit first-day structure and the weekend setback are visible in the same recording.


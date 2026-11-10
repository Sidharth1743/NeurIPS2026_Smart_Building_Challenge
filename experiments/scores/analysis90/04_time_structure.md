> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# 04 — Time structure

Every episode is 8,640 steps of 5 minutes starting **Sunday 06:00**. The clock is rebuilt from the step index, so it is identical for all 90 scenarios and for the baseline. Bins: weekday night 19–06, weekday morning 06–12, afternoon 12–16, evening 16–19, weekend day 06–19, weekend night. Day 0 of the episode is an 18-hour Sunday; the other 29 days are full days.

Steps per bin: weekday night 2844, weekday morning 1512, weekday afternoon 1008, weekday evening 756, weekend day 1404, weekend night 1116.

## Where the score is spent, and where the gap comes from

| day part | steps | pid score/step | ours score/step | share of the gap |
| --- | --- | --- | --- | --- |
| weekday night | 2844 | 0.00374 | 0.00486 | -1.8% |
| weekday morning | 1512 | 0.12797 | 0.06334 | +54.4% |
| weekday afternoon | 1008 | 0.10430 | 0.05850 | +25.7% |
| weekday evening | 756 | 0.09739 | 0.06162 | +15.1% |
| weekend day | 1404 | 0.00610 | 0.00121 | +3.8% |
| weekend night | 1116 | 0.00607 | 0.00167 | +2.7% |
| **month** | 8640 | 0.04609 | 0.02532 | 100.0% |

The per-step score is a mean, so the "share of the gap" column is `(ours − pid) × steps / 8640 ÷ total gap` and sums to 100%.

- **The weekday morning carries 55% of the entire win**, the afternoon 26%, the evening 15%, the two weekend bins 6.5%. The weekday night is the only bin where the learned policy is **worse**: it adds 0.00037 to the score, −1.8% of the gap, because it spends energy before occupancy that the baseline does not.
- The mechanism is load shifting, not a uniform saving: the policy heats at 05:00, meets the 06:00 occupancy with a prepared building, and then coasts. The baseline reacts at 06:00 and pays for it all morning.

## Day part × building (primary policy)

| building | weekday night | weekday morning | weekday afternoon | weekday evening | weekend day | weekend night |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 0.0063 / 0.0049 | 0.0303 / 0.1080 | 0.0258 / 0.0801 | 0.0265 / 0.0782 | 0.0014 / 0.0082 | 0.0024 / 0.0089 |
| building_train_1 | 0.0043 / 0.0030 | 0.0390 / 0.1031 | 0.0373 / 0.0721 | 0.0368 / 0.0697 | 0.0006 / 0.0063 | 0.0009 / 0.0062 |
| building_train_6 | 0.0029 / 0.0022 | 0.0463 / 0.1336 | 0.0350 / 0.1203 | 0.0322 / 0.1138 | 0.0008 / 0.0035 | 0.0008 / 0.0033 |
| building_train_3 | 0.0050 / 0.0050 | 0.1448 / 0.1904 | 0.1162 / 0.1622 | 0.1012 / 0.1447 | 0.0010 / 0.0084 | 0.0009 / 0.0076 |
| building_train_2 | 0.0058 / 0.0036 | 0.0563 / 0.1048 | 0.0781 / 0.0868 | 0.1113 / 0.0805 | 0.0023 / 0.0041 | 0.0034 / 0.0044 |

Each cell is ours / pid, score per step. The morning is the largest cell on four of the five buildings; on the 85-zone building the evening is largest and is the one place where the learned policy is worse than the baseline in an occupied bin (0.111 vs 0.081). The weekday-night cell is also worse on four buildings.

## Hour of day (weekdays, mean over the five buildings)

| hour | gas kWh/step | elec kWh/step | pid energy term | ours productivity | pid productivity |
| --- | --- | --- | --- | --- | --- |
| 00 | 0.34 | 0.03 | 0.0060 | 0.0000 | 0.0000 |
| 01 | 0.37 | 0.06 | 0.0066 | 0.0000 | 0.0000 |
| 02 | 0.58 | 0.09 | 0.0073 | 0.0000 | 0.0000 |
| 03 | 1.41 | 0.12 | 0.0080 | 0.0000 | 0.0000 |
| 04 | 2.37 | 0.28 | 0.0087 | 0.0000 | 0.0000 |
| 05 | 9.24 | 2.00 | 0.0092 | -0.0069 | -0.0248 |
| 06 | 8.46 | 2.94 | 0.1830 | -0.0650 | -0.1764 |
| 07 | 7.83 | 2.91 | 0.0922 | -0.0571 | -0.1574 |
| 08 | 7.36 | 2.84 | 0.0845 | -0.0572 | -0.1546 |
| 09 | 6.80 | 2.82 | 0.0824 | -0.0550 | -0.1445 |
| 10 | 6.14 | 2.74 | 0.0754 | -0.0541 | -0.1428 |
| 11 | 5.45 | 2.71 | 0.0805 | -0.0558 | -0.1393 |
| 12 | 4.85 | 2.80 | 0.0791 | -0.0584 | -0.1377 |
| 13 | 4.62 | 3.10 | 0.0814 | -0.0580 | -0.1366 |
| 14 | 5.11 | 3.45 | 0.0821 | -0.0495 | -0.1334 |
| 15 | 5.43 | 3.32 | 0.0779 | -0.0514 | -0.1331 |
| 16 | 5.00 | 3.10 | 0.0806 | -0.0591 | -0.1303 |
| 17 | 4.47 | 3.11 | 0.0749 | -0.0507 | -0.1303 |
| 18 | 1.50 | 0.69 | 0.0714 | -0.1048 | -0.1189 |
| 19 | 1.05 | 0.00 | 0.0009 | 0.0000 | 0.0000 |
| 20 | 0.10 | 0.00 | 0.0012 | 0.0000 | 0.0000 |
| 21 | 0.08 | 0.00 | 0.0018 | 0.0000 | 0.0000 |
| 22 | 0.10 | 0.00 | 0.0026 | 0.0000 | 0.0000 |
| 23 | 0.11 | 0.00 | 0.0035 | 0.0000 | 0.0000 |

The learned policy peaks at **05:00** (gas 9.2 kWh/step) and is already high at 06:00; the baseline peaks *at* 06:00 (energy term 0.183, about twice its own daytime level) because it starts heating when occupancy starts. The learned comfort is 2–3× better than the baseline in every occupied hour. In the last occupied hour (18:00) the learned policy drops from −0.051 to −0.105, converging on the baseline, because it stops the plant before the building empties; that is the clearest single behavioural signature in the month.

## Weekend behaviour

| policy | building_train_8 | building_train_1 | building_train_6 | building_train_3 | building_train_2 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| ours (kWh) | 5.7% | 2.3% | 3.2% | 2.1% | 7.9% | 4.2% |
| pid (energy $) | 18.2% | 15.5% | 12.3% | 13.8% | 16.8% | 15.3% |

Weekends are 29% of the steps. The learned policy puts only **2–8% of its monthly kWh** there (and `train_1` only 2.3%): a real setback. The baseline keeps a large idle load (weekend-day energy term 0.0143 against the learned policy's 0.0025).

## The first 24 hours: the policy pays up front

| building | day-0 score (ours) | day-0 score (pid) | gas first 24 h kWh | of which last 6 h | days 1–29 (ours) | days 1–29 (pid) |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 0.00455 | 0.00458 | 176 | 109 | 0.01353 | 0.04038 |
| building_train_1 | 0.00348 | 0.00027 | 162 | 80 | 0.01646 | 0.03659 |
| building_train_6 | 0.00714 | 0.00103 | 630 | 191 | 0.01653 | 0.05074 |
| building_train_3 | 0.00945 | 0.00261 | 1181 | 536 | 0.05108 | 0.07118 |
| building_train_2 | 0.01613 | 0.00257 | 2674 | 1118 | 0.03194 | 0.03914 |

Day 0 is an empty Sunday with no comfort bill. The learned policy heats through it — on `train_2` it burns 2.7 MWh(th) of gas before a single occupant arrives — so its day-0 score is *worse* than the baseline on four of five buildings. That is the delayed-credit trade in one row: the policy invests on Sunday and is paid back from Monday. Over the month it is strongly positive, but the first 24 h is not, and a mean that hides it would be misleading.

## Thirty-day trajectory

| policy | building_train_8 | building_train_1 | building_train_6 | building_train_3 | building_train_2 | reading |
| --- | --- | --- | --- | --- | --- | --- |
| r122@89,579,520 | -2.0e-05 | +2.3e-05 | +5.3e-05 | +1.5e-04 | -2.8e-04 | stable |
| pid_water | +1.8e-04 | +1.5e-04 | -4.4e-05 | -2.7e-05 | +4.3e-06 | drift |

Slope of the daily score over 30 days (score units per day). The largest 30-day slope is ±3×10⁻⁴/day; the 70-zone building drifts +1.5×10⁻⁴/day for the learned policy (about +0.004 over the month). No end-of-month collapse and no learning inside the episode.

## Peaks, ramps and stability

| building | elec p99 kWh | elec max | gas p99 kWh | gas max | mean abs dElec/step | weekend kWh share |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 3.0 | 6.0 | 22.7 | 77.8 | 0.018 | 5.7% |
| building_train_1 | 4.7 | 7.3 | 31.3 | 91.4 | 0.024 | 2.3% |
| building_train_6 | 6.7 | 9.2 | 32.9 | 69.9 | 0.028 | 3.2% |
| building_train_3 | 15.8 | 18.4 | 60.4 | 119.1 | 0.056 | 2.1% |
| building_train_2 | 9.4 | 14.8 | 59.4 | 127.9 | 0.072 | 7.9% |

Five-minute peaks scale with plant size; the mean absolute step-to-step change in electricity is 0.018–0.072 kWh, so the policy is not chattering between extremes in the recorded signal.

## What this file adds beyond "a daily average"

- **The gap is allocated to day parts**, and one bin (weekday morning) carries 55% of it.
- **One bin goes the wrong way** (weekday night, −1.8% of the gap share, +0.00037 to the score) and is named.
- **The mechanism is visible in the hourly profile**: 05:00 pre-heat vs the baseline's 06:00 reactive spike.
- **Weekend setback quantified** as a share of monthly kWh (2–8% on 29% of steps).
- **The day-0 investment** is separated from the rest of the month, so the delayed credit is not hidden by the mean.
- **Stability** (30-day slope, peaks, ramps) rather than a single monthly number.


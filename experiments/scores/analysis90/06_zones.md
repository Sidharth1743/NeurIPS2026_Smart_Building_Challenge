# 06 — Zones: where the scored comfort regret lives

Comfort is a building mean of a per-zone quantity, and it is **zero when a zone is unoccupied**. The canonical observation carries, per zone slot, the temperature and both setpoints (`T`, `Hs`, `Cs`); decoding them reproduces the recorded `comfort_dev_K` (0.414305 vs 0.414324) and the worst-zone maximum (20.916 K both). The band deficit is `D = max(Hs − T, 0) + max(T − Cs, 0)` in kelvin.

That forces two numbers per building, and conflating them would be misleading:

- **free float** — mean deficit while the building is empty. The reward does not bill it; it is the night/weekend setback by design.
- **scored** — mean deficit on occupied steps, which is what the comfort term actually charges.

Zone labels are canonical slots (AHU k, position j); the recording does not carry VAV names. The baseline has no zone recording, so all comparisons are among learned checkpoints.

## Scored deficit vs free float

| building | zones | scored deficit K | free-float deficit K | in band (occupied) | in band (all steps) | scored day | scored night |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 0.055 | 1.224 | 89.8% | 73.9% | 0.056 | 0.008 |
| building_train_1 | 39 | 0.110 | 0.589 | 81.7% | 76.7% | 0.110 | 0.014 |
| building_train_6 | 43 | 0.153 | 0.551 | 83.0% | 76.1% | 0.154 | 0.041 |
| building_train_3 | 70 | 0.702 | 1.096 | 57.5% | 55.4% | 0.704 | 0.356 |
| building_train_2 | 85 | 0.484 | 1.013 | 51.7% | 53.0% | 0.487 | 0.015 |

The free-float deficit is 1.6–22× the scored one, and that is correct behaviour: the policy lets the empty building drift and pays nothing for it. The scored residual runs from 0.06 K (`train_8`) to 0.70 K (`train_3`). Day and night are nearly equal on `train_3`, so its problem is not the setback (on the other four the scored deficit is almost all daytime, because that is when the building is occupied).

## Concentration inside the building (occupied steps)

| building | worst zone | median zone | best zone | top-5 share | top-10 share | worst five (AHU/position) |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 0.38 | 0.03 | 0.01 | 57% | 75% | AHU2/1, AHU3/6, AHU1/0, AHU3/0, AHU3/3 |
| building_train_1 | 0.51 | 0.05 | 0.01 | 49% | 69% | AHU4/9, AHU2/1, AHU4/5, AHU1/0, AHU4/8 |
| building_train_6 | 1.12 | 0.05 | 0.00 | 47% | 71% | AHU2/1, AHU2/2, AHU1/6, AHU4/2, AHU3/3 |
| building_train_3 | 3.88 | 0.62 | 0.02 | 24% | 39% | AHU4/2, AHU2/3, AHU4/5, AHU5/6, AHU4/3 |
| building_train_2 | 1.41 | 0.42 | 0.32 | 14% | 23% | AHU9/8, AHU4/0, AHU6/3, AHU3/6, AHU1/3 |

Two different shapes. `train_8`, `train_1` and `train_6` are **concentrated**: the top five zones carry 47–57% of the scored deficit, and the best zones are essentially perfect (0.00–0.01 K), so the ratio to the worst zone is large but not the useful number. `train_2` is **diffuse**: top-5 only 14% and the worst zone is 1.41 K against a 0.42 K median — there is no single room to fix. `train_3` is the extreme case: one zone at 3.9 K above a 0.62 K median, but the top-5 share is still 24%, so it is one very bad zone on a broadly mediocre floor.

## Per AHU (occupied steps)

| building | AHUs | best AHU | worst AHU | worst AHU # | worst/best |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | 3 | 0.03 | 0.07 | 2 | 1.9× |
| building_train_1 | 4 | 0.08 | 0.16 | 4 | 1.9× |
| building_train_6 | 6 | 0.07 | 0.38 | 2 | 5.3× |
| building_train_3 | 7 | 0.43 | 1.28 | 4 | 2.9× |
| building_train_2 | 9 | 0.41 | 0.56 | 4 | 1.4× |

`train_1` and `train_2` are uniform across their AHUs (1.9× and 1.4×): the residual is a few zones, not an air handler. `train_6` is not — **AHU2 at 0.38 K against 0.07 K for the best handler, a factor of 5.3**. `train_3` is the same story: **AHU4 at 1.28 K and 42% in-band against AHU5 at 0.43 K and 78%**, a factor of 2.9. These two AHU effects are the most actionable zone-level findings in the set.

## Does the policy feed the zones that need it?

| building | deficit vs damper | deficit vs reheat | cold share |
| --- | --- | --- | --- |
| building_train_8 | -0.33 | +0.37 | 38% |
| building_train_1 | +0.18 | -0.14 | 35% |
| building_train_6 | +0.06 | -0.02 | 50% |
| building_train_3 | +0.25 | -0.46 | 36% |
| building_train_2 | -0.12 | +0.25 | 30% |

On occupied steps the zone-level link is **weak and not consistent** (−0.33 to +0.25 for the damper, −0.46 to +0.37 for reheat). Only `train_8` shows the clean "losing a zone" signature (more reheat, less damper); on `train_1` and `train_6` the correlation is near zero, and on `train_3` reheat is *negatively* correlated with deficit (the cold zones get less reheat). That is a useful negative result: the residual on the scored steps is not explained by how much air or heat each zone receives, so it is not a simple per-zone setpoint or feeding error. The deficit is 30–50% cold depending on the window; both sides matter.

## Is it the same zones under every checkpoint?

| building | r122 saves: rank corr | top-5 overlap | vs r116: rank corr | top-5 overlap |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.94 | 4/5 | 0.74 | 2/5 |
| building_train_1 | 0.78 | 5/5 | 0.84 | 3/5 |
| building_train_6 | 0.97 | 5/5 | 0.90 | 2/5 |
| building_train_3 | 0.98 | 4/5 | 0.98 | 5/5 |
| building_train_2 | 0.56 | 4/5 | 0.65 | 3/5 |

The rank correlation between the two r122 saves is 0.56–0.98 (top-5 overlap 4–5 of 5), and r116 agrees on the same zones (0.65–0.98), though its top-5 set overlaps only 2–5 of 5. The weak zones are a stable property of the building, not a checkpoint accident — which is what makes them addressable.

## The worst month, day by day (occupied steps only)

| building | worst scenario | mean daily deficit | best day | worst day | worst / median day |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | building_train_8__minneapolis_1 | 0.05 | 0.00 | 0.09 | 1.8× |
| building_train_1 | building_train_1__minneapolis_1 | 0.10 | 0.00 | 0.20 | 2.0× |
| building_train_6 | building_train_6__minneapolis_1 | 0.16 | 0.00 | 0.33 | 2.4× |
| building_train_3 | building_train_3__minneapolis_1 | 0.69 | 0.10 | 1.06 | 1.6× |
| building_train_2 | building_train_2__denver_4 | 0.39 | 0.00 | 0.66 | 1.4× |

On occupied steps the daily spread is much smaller than on all steps: the worst day is 1.4–2.4× the median day. The deficit is a persistent operating condition, not one bad day or a start transient.

## What this file adds beyond the building mean

- **Separates free float from scored comfort**, which is the difference between a 0.4 K and a 0.11 K problem on `train_1`. Reporting the all-steps number alone would be wrong.
- **Per-zone concentration**: a few zones on `train_8`/`train_1`, diffuse on `train_2`.
- **The AHU4 effect on `train_3`** (1.28 K vs 0.43 K) as a specific target for the next batch.
- **A negative result on the action link**: on occupied steps the deficit does not track how much air or heat a zone receives, so it is not a simple per-zone setpoint or feeding error.
- **The same weak zones under every checkpoint**, and day-level persistence, so the tail is stable and addressable.


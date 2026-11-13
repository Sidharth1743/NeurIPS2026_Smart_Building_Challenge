# r15_structured_beta_b2048_s75 — 144-scenario analysis (from the recordings)

Score = mean over scenarios of the return; lower is better. Terms are the fleet means of the per-step parts over all 8640 steps x agents.

## Headline

| step | 144 score | productivity | energy | carbon | comfort dev K | worst zone K | elec kWh | gas kWh |
|---|---|---|---|---|---|---|---|---|
| 50,135,040 | **0.029730** | -0.0381 | 0.0209 | 0.0134 | 0.872 | 31.8 | 8349 | 9234 |

## Per building

| step | train_1 | train_2 | train_3 | train_4 | train_5 | train_6 | train_7 | train_8 |
|---|---|---|---|---|---|---|---|---|
| 50,135,040 | 0.0223 | 0.0208 | 0.0814 | 0.0174 | 0.0250 | 0.0261 | 0.0264 | 0.0185 |

## Per weather (mean over the buildings/weathers available)

| step | atlanta | denver | fairbanks | minneapolis | phoenix | singapore |
|---|---|---|---|---|---|---|
| 50,135,040 | 0.0189 | 0.0326 | 0.0345 | 0.0352 | 0.0164 | 0.0245 |

## Scenarios that carry the sum (top 10 at the final step)

| scenario | mean reward | productivity | energy | carbon |
|---|---|---|---|---|
| building_train_3__minneapolis_1 | 0.1695 | -0.2587 | 0.0234 | 0.0479 |
| building_train_3__fairbanks_3 | 0.1614 | -0.2487 | 0.0230 | 0.0381 |
| building_train_3__minneapolis_2 | 0.1510 | -0.2341 | 0.0181 | 0.0349 |
| building_train_3__denver_1 | 0.1479 | -0.2328 | 0.0157 | 0.0252 |
| building_train_3__minneapolis_8 | 0.1404 | -0.2092 | 0.0293 | 0.0449 |
| building_train_3__minneapolis_3 | 0.1104 | -0.1661 | 0.0296 | 0.0242 |
| building_train_7__minneapolis_1 | 0.0882 | -0.1101 | 0.0347 | 0.0761 |
| building_train_3__minneapolis_7 | 0.0772 | -0.1119 | 0.0349 | 0.0157 |
| building_train_5__minneapolis_1 | 0.0710 | -0.0973 | 0.0217 | 0.0413 |
| building_train_3__fairbanks_6 | 0.0699 | -0.1013 | 0.0319 | 0.0135 |

## Time of day (fleet means of the per-step parts)

| bin | 50,135,040 |
|---|---|
| night 19-06 | -0.0161 |
| morning 06-12 | +0.0012 |
| afternoon 12-16 | +0.0010 |
| evening 16-19 | - |
| weekend 06-19 | -0.0197 |

(cell value = 0.6*productivity + 0.2*energy + 0.2*carbon for that bin, so it is the score's arithmetic on the fleet means, not a per-scenario mean.)


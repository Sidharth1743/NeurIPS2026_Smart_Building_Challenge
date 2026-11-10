> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# 02 — Score decomposition

`score = −0.6·productivity + 0.2·energy + 0.2·carbon`, with productivity stored signed (negative = comfort regret) and both energy and carbon already normalised. Lower is better; the ideal is 0. The identity was re-checked on all 450 policy×scenario rows (max error 2.9e-07).

**Term contributions.** Because the score is linear in the terms, each term contributes `−0.6·productivity`, `+0.2·energy`, `+0.2·carbon` points to the score. The three add to the score exactly; the last column is the share of the score each term carries.

| policy | score | productivity | energy | carbon | comfort pts | energy pts | carbon pts | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pid_water | 0.0461 | -0.05429 | 0.03908 | 0.02851 | 0.0326 | 0.0078 | 0.0057 | 71% / 17% / 12% |
| r122@119,439,360 | 0.0272 | -0.02717 | 0.02751 | 0.02713 | 0.0163 | 0.0055 | 0.0054 | 60% / 20% / 20% |
| r122@89,579,520 | 0.0253 | -0.02285 | 0.02971 | 0.02833 | 0.0137 | 0.0059 | 0.0057 | 54% / 23% / 22% |
| r116@50,135,040 | 0.0301 | -0.03119 | 0.03210 | 0.02494 | 0.0187 | 0.0064 | 0.0050 | 62% / 21% / 17% |
| r116@45,121,536 | 0.0307 | -0.03207 | 0.03263 | 0.02468 | 0.0192 | 0.0065 | 0.0049 | 63% / 21% / 16% |

Comfort is the largest single contributor to every score, which is why a policy that only chased energy would not look like this.

## Overall comparison against the baseline

| policy | score | vs pid | absolute |
| --- | --- | --- | --- |
| r122@119,439,360 | 0.0272 | -40.9% | -0.01886 |
| r122@89,579,520 | 0.0253 | -45.1% | -0.02078 |
| r116@50,135,040 | 0.0301 | -34.6% | -0.01597 |
| r116@45,121,536 | 0.0307 | -33.4% | -0.01539 |

Gap to the baseline decomposed into the three terms (negative = improvement):

| policy | comfort pts | energy pts | carbon pts | total |
| --- | --- | --- | --- | --- |
| r122@119,439,360 | -0.01627 | -0.00231 | -0.00028 | -0.01886 |
| r122@89,579,520 | -0.01887 | -0.00188 | -0.00004 | -0.02078 |
| r116@50,135,040 | -0.01386 | -0.00140 | -0.00071 | -0.01597 |
| r116@45,121,536 | -0.01333 | -0.00129 | -0.00077 | -0.01539 |

For the primary row the gap is almost entirely comfort and electricity; the carbon term moves by −0.00004 (0.2% of the gap), so no fuel swap is doing the work.

## Per building

| building | zones | r122@89,579,520 | pid_water | vs pid | productivity | energy | carbon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 0.0132 | 0.0392 | -66.2% | -0.00445 | 0.02361 | 0.02920 |
| building_train_1 | 39 | 0.0160 | 0.0354 | -54.7% | -0.00898 | 0.02512 | 0.02806 |
| building_train_6 | 43 | 0.0162 | 0.0491 | -67.0% | -0.01115 | 0.02547 | 0.02217 |
| building_train_3 | 70 | 0.0497 | 0.0689 | -27.9% | -0.05276 | 0.05282 | 0.03738 |
| building_train_2 | 85 | 0.0314 | 0.0379 | -17.2% | -0.03689 | 0.02151 | 0.02485 |

The margin is not uniform: it is largest on the two smallest plants (66%, 67%) and smallest on the two largest (17% on the 85-zone building, 28% on the 70-zone one). That is the honest limitation, and it lines up with the score level: the two largest plants are also the two highest-scoring buildings for every policy.

## Per station (weather)

| station | n | r122@89,579,520 | pid_water | vs pid | productivity | energy | carbon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atlanta | 10 | 0.0188 | 0.0316 | -40.7% | -0.01676 | 0.03296 | 0.01055 |
| denver | 10 | 0.0279 | 0.0490 | -43.1% | -0.02659 | 0.02504 | 0.03456 |
| fairbanks | 10 | 0.0252 | 0.0499 | -49.6% | -0.02408 | 0.01928 | 0.03445 |
| minneapolis | 40 | 0.0287 | 0.0519 | -44.8% | -0.02559 | 0.02797 | 0.03864 |
| phoenix | 10 | 0.0181 | 0.0278 | -34.9% | -0.01744 | 0.02972 | 0.00849 |
| singapore | 10 | 0.0232 | 0.0487 | -52.3% | -0.01839 | 0.04845 | 0.01241 |

Phoenix is the weakest relative win (−34.9%) and the only station where the r116 saves land at the baseline. The largest absolute gaps are Singapore (0.0255), Fairbanks (0.0247) and Minneapolis (0.0232). The station-equal mean is 0.02364 vs pid 0.04317 (−45.2%), the same as the window-weighted −45.1%.

## Per window (mean over the five buildings)

| window | HDD18 | CDD21 | r122@89,579,520 | pid_water | vs pid |
| --- | --- | --- | --- | --- | --- |
| atlanta_4 | 0 | 158 | 0.0207 | 0.0347 | -40.3% |
| atlanta_7 | 80 | 40 | 0.0168 | 0.0285 | -41.1% |
| denver_1 | 567 | 0 | 0.0381 | 0.0718 | -46.9% |
| denver_4 | 75 | 53 | 0.0176 | 0.0261 | -32.6% |
| fairbanks_3 | 591 | 0 | 0.0366 | 0.0723 | -49.4% |
| fairbanks_6 | 117 | 9 | 0.0138 | 0.0275 | -50.0% |
| minneapolis_1 | 798 | 0 | 0.0606 | 0.1109 | -45.4% |
| minneapolis_2 | 634 | 0 | 0.0455 | 0.0840 | -45.8% |
| minneapolis_3 | 321 | 11 | 0.0239 | 0.0488 | -51.0% |
| minneapolis_4 | 55 | 64 | 0.0179 | 0.0292 | -38.7% |
| minneapolis_5 | 7 | 89 | 0.0187 | 0.0280 | -33.3% |
| minneapolis_6 | 8 | 59 | 0.0174 | 0.0264 | -34.2% |
| minneapolis_7 | 154 | 22 | 0.0164 | 0.0317 | -48.4% |
| minneapolis_8 | 467 | 0 | 0.0290 | 0.0566 | -48.7% |
| phoenix_2 | 34 | 63 | 0.0173 | 0.0269 | -35.8% |
| phoenix_7 | 14 | 105 | 0.0190 | 0.0288 | -34.1% |
| singapore_2 | 0 | 219 | 0.0230 | 0.0462 | -50.2% |
| singapore_5 | 0 | 221 | 0.0234 | 0.0512 | -54.3% |

Every window favours the learned policy. The largest relative gains are Singapore and the shoulder seasons (−50% to −54%); the smallest are the mild mixed windows (denver_4 −33%, minneapolis_5 −33%, phoenix_7 −34%), where the baseline is already cheap.

## The tail

| policy | worst-5 share | worst-10 | worst-20 | <0.01 | <0.02 | <0.05 | <0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pid_water | 15.0% | 25.8% | 42.8% | 0 | 7 | 61 | 85 |
| r122@119,439,360 | 17.0% | 29.1% | 47.9% | 22 | 43 | 76 | 89 |
| r122@89,579,520 | 18.7% | 29.6% | 46.5% | 25 | 43 | 83 | 89 |
| r116@50,135,040 | 13.1% | 22.1% | 37.5% | 6 | 22 | 84 | 90 |
| r116@45,121,536 | 13.5% | 22.6% | 38.1% | 5 | 22 | 83 | 90 |

The learned policy is slightly more tail-concentrated than the baseline (worst-10 = 29.6% vs 25.8%), and still has fewer scenarios above 0.05. The tail is named in file 01: it is the 70-zone new building at Minneapolis, not a diffuse noise floor.

## What drives the variance

| policy | building | weather window | interaction |
| --- | --- | --- | --- |
| pid_water | 19.2% | 71.6% | 9.2% |
| r122@119,439,360 | 46.7% | 49.4% | 3.9% |
| r122@89,579,520 | 49.1% | 37.5% | 13.4% |
| r116@50,135,040 | 38.0% | 37.1% | 24.9% |
| r116@45,121,536 | 41.8% | 36.1% | 22.1% |

With the full factorial (5×18, one cell each) the score variance splits into a building share, a weather share and a residual interaction; no replication, so the interaction cannot be separated from noise. Reading: the baseline is weather-dominated (72% weather, 19% building); every learned policy cuts the weather share to 36–49% and raises the building share to 38–49%. For the primary save the building is now the larger term (49% vs 38%). The policy removes most of the month-to-month sensitivity.

## Checkpoint-to-checkpoint agreement (results section) and r116 (appendix)

- The two r122 saves on the same 90 scenarios: mean |Δ| 0.00497, max |Δ| 0.01970, Pearson r 0.955, Spearman 0.930. 54/90 scenarios favour the later save.
- r116@50,135,040 vs r122@89,579,520: mean Δ +0.00481, Pearson r 0.690, wins on 24/90 scenarios (appendix).
- r116@45,121,536 vs r122@89,579,520: mean Δ +0.00539, Pearson r 0.704, wins on 22/90 scenarios (appendix).

## Fuel, for the learned checkpoints

| building | electricity MWh/month | gas MWh(th)/month | gas share |
| --- | --- | --- | --- |
| building_train_8 | 3.4 | 13.7 | 80% |
| building_train_1 | 5.2 | 17.2 | 77% |
| building_train_6 | 9.0 | 18.9 | 68% |
| building_train_3 | 23.4 | 33.9 | 59% |
| building_train_2 | 9.9 | 34.6 | 78% |

Monthly totals per scenario (mean over the 18 windows). The split is gas-heavy on every building (59–80% of the two carriers), and the carbon term is flat, so the policy is not moving load from gas onto electricity. The baseline CSV stores normalised cost rather than kWh, so a like-for-like fuel split for `pid_water` is not in this file; its energy and carbon terms are in the tables above.

## What this file adds beyond "score by building"

- **Term contributions in score points**, so a 0.005 change is attributable to comfort or fuel.
- **Gap decomposition**: how much of the −45% comes from each term — 91% of the gap is comfort, 9% electricity, 0.2% carbon.
- **Station-equal mean** beside the window-weighted headline, which removes the Minneapolis tilt without hiding it.
- **Variance split** (building vs weather vs interaction), which answers "is this a weather result?" directly.
- **Tail shares and threshold counts**, which the mean alone hides.
- **Scenario-level agreement** between saves, which is the reproducibility evidence a single-episode evaluation otherwise lacks.
- **Fuel in kWh**, not only the normalised term.


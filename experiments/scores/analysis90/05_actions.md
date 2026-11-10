# 05 — Actions: what the policy does with 75–209 actuators

The paper's RL demo restricted the action space to **2 setpoints** (supply-air and hot-water temperature, Appendix J). This policy controls every native actuator the gateway exposes, and the recording shows what it does with them. `loc.bin` is the executed mean action per canonical head per step (2 per zone, 4 per AHU, 3 plant); `conc.bin` is the Beta concentration (α+β, capped at 12) of the same head. The baseline has no action recording, so every comparison here is among the four learned checkpoints.

## Scale and coverage

| building | zones | AHUs | zone heads (damper+reheat) | AHU heads | plant heads | native actions | vs 2 setpoints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | 3 | 60 | 12 | 3 | 75 | 38× |
| building_train_1 | 39 | 4 | 78 | 16 | 3 | 97 | 48× |
| building_train_6 | 43 | 6 | 86 | 24 | 3 | 113 | 56× |
| building_train_3 | 70 | 7 | 140 | 28 | 3 | 171 | 86× |
| building_train_2 | 85 | 9 | 170 | 36 | 3 | 209 | 104× |

Every native action maps to a distinct canonical slot (verified: the number of mapped slots equals `n_action` from the blob for all 20 recordings), so 100% of the gateway action space is policy-controlled — there is no residual, no guard and no fallback. The recorder's 288 columns include 69 padding slots; they are exactly zero.

## Every mapped head moves; one is pinned

| building | policy | heads | static heads (σ<0.05) | pinned (|a|>0.95 half the time) | median per-head σ |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | r122@89,579,520 | 75 | 5% | 0% | 0.51 |
| building_train_8 | r116@50,135,040 | 75 | 2% | 0% | 0.76 |
| building_train_1 | r122@89,579,520 | 97 | 6% | 0% | 0.54 |
| building_train_1 | r116@50,135,040 | 97 | 3% | 0% | 0.69 |
| building_train_6 | r122@89,579,520 | 113 | 6% | 0% | 0.51 |
| building_train_6 | r116@50,135,040 | 113 | 3% | 0% | 0.75 |
| building_train_3 | r122@89,579,520 | 171 | 4% | 0% | 0.48 |
| building_train_3 | r116@50,135,040 | 171 | 2% | 0% | 0.70 |
| building_train_2 | r122@89,579,520 | 209 | 5% | 0% | 0.54 |
| building_train_2 | r116@50,135,040 | 209 | 2% | 0% | 0.55 |

The dampers, reheat valves and AHU run bits all move. The **AHU static-pressure setpoint** is the one head that is static in essentially every AHU and every building (97–100%, held at -0.89 to -0.88 ± 0.03 on the [−1,1] scale, i.e. the low end of the 0–2000 Pa range), while the r116 saves move it (σ 0.10–0.17) and leave 22–33% of those heads static. Supply-air temperature and recirculation are also static on 0–22% of the AHU heads, which is why the table's "static heads" column is 4–6% rather than exactly one head per AHU.

## What each role does (primary policy)

| building | role | mean | σ | p5 / p50 / p95 | mean abs Δ/step | α+β median | static heads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | damper | 0.03 | 0.57 | -0.84 / 0.04 / 0.83 | 0.047 | 10.8 | 0% |
| building_train_8 | reheat | -0.17 | 0.57 | -0.85 / -0.26 / 0.80 | 0.043 | 10.5 | 0% |
| building_train_8 | SAT | 0.42 | 0.61 | -0.80 / 0.84 / 0.90 | 0.011 | 12.0 | 6% |
| building_train_8 | recirc | 0.69 | 0.39 | -0.77 / 0.82 / 0.86 | 0.006 | 12.0 | 0% |
| building_train_8 | water | 0.56 | 0.66 | -0.90 / 0.89 / 0.92 | 0.009 | 12.0 | 6% |
| building_train_1 | damper | 0.00 | 0.56 | -0.84 / 0.04 / 0.82 | 0.036 | 10.5 | 0% |
| building_train_1 | reheat | -0.05 | 0.56 | -0.84 / -0.14 / 0.84 | 0.035 | 10.5 | 0% |
| building_train_1 | SAT | 0.47 | 0.58 | -0.68 / 0.85 / 0.90 | 0.011 | 12.0 | 22% |
| building_train_1 | recirc | 0.69 | 0.39 | -0.76 / 0.81 / 0.85 | 0.006 | 12.0 | 21% |
| building_train_1 | water | 0.47 | 0.71 | -0.90 / 0.87 / 0.92 | 0.011 | 12.0 | 0% |
| building_train_6 | damper | 0.21 | 0.54 | -0.83 / 0.29 / 0.86 | 0.032 | 11.2 | 0% |
| building_train_6 | reheat | -0.04 | 0.57 | -0.84 / -0.09 / 0.83 | 0.033 | 10.5 | 0% |
| building_train_6 | SAT | 0.36 | 0.66 | -0.79 / 0.82 / 0.89 | 0.011 | 12.0 | 0% |
| building_train_6 | recirc | 0.69 | 0.37 | -0.72 / 0.81 / 0.86 | 0.006 | 12.0 | 1% |
| building_train_6 | water | 0.54 | 0.65 | -0.89 / 0.88 / 0.92 | 0.009 | 12.0 | 0% |
| building_train_3 | damper | 0.40 | 0.49 | -0.76 / 0.58 / 0.87 | 0.020 | 12.0 | 0% |
| building_train_3 | reheat | 0.00 | 0.57 | -0.82 / -0.04 / 0.84 | 0.026 | 10.6 | 0% |
| building_train_3 | SAT | 0.30 | 0.72 | -0.87 / 0.82 / 0.89 | 0.013 | 12.0 | 0% |
| building_train_3 | recirc | 0.62 | 0.45 | -0.84 / 0.79 / 0.85 | 0.006 | 12.0 | 0% |
| building_train_3 | water | 0.46 | 0.67 | -0.90 / 0.86 / 0.91 | 0.010 | 12.0 | 0% |
| building_train_2 | damper | 0.15 | 0.57 | -0.84 / 0.23 / 0.84 | 0.026 | 9.9 | 0% |
| building_train_2 | reheat | -0.10 | 0.60 | -0.81 / -0.24 / 0.86 | 0.034 | 9.6 | 0% |
| building_train_2 | SAT | 0.54 | 0.52 | -0.65 / 0.85 / 0.90 | 0.013 | 12.0 | 11% |
| building_train_2 | recirc | 0.73 | 0.32 | 0.50 / 0.81 / 0.87 | 0.004 | 12.0 | 2% |
| building_train_2 | water | 0.18 | 0.77 | -0.90 / 0.67 / 0.91 | 0.019 | 12.0 | 0% |

Read as: the damper is centred but swings almost the full range (σ≈0.5); reheat is mostly closed at the small buildings and near neutral on the large ones; the supply-air setpoint sits high (0.3–0.5 in normalised units, i.e. warm supply) with occasional low excursions; recirculation is held high (0.6–0.7) on the r122 saves. The Beta concentration is at or near the cap (12) almost everywhere: the policy is decisive, not hedging. The only heads that hedge are the r116 recirculation heads (median α+β 6.1–6.9, i.e. a broad Beta that still covers most of the range).

## Run bits: the plant runs in blocks, not chatter

| building | policy | AHU run on | AHU flips/day | AHU mean on-run (steps) | plant on | plant flips/day | plant mean on-run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | r122@89,579,520 | 39% | 4.4 | 149 | 17% | 0.8 | 113 |
| building_train_8 | r116@50,135,040 | 76% | 1.6 | 748 | 20% | 1.1 | 103 |
| building_train_1 | r122@89,579,520 | 39% | 5.7 | 150 | 16% | 1.0 | 88 |
| building_train_1 | r116@50,135,040 | 64% | 2.8 | 490 | 16% | 0.9 | 93 |
| building_train_6 | r122@89,579,520 | 41% | 8.4 | 162 | 14% | 0.6 | 127 |
| building_train_6 | r116@50,135,040 | 65% | 3.7 | 560 | 13% | 0.9 | 85 |
| building_train_3 | r122@89,579,520 | 41% | 9.9 | 158 | 13% | 0.5 | 129 |
| building_train_3 | r116@50,135,040 | 55% | 5.0 | 416 | 12% | 0.7 | 96 |
| building_train_2 | r122@89,579,520 | 51% | 12.7 | 199 | 18% | 1.0 | 98 |
| building_train_2 | r116@50,135,040 | 45% | 9.8 | 227 | 12% | 0.8 | 81 |

A 5-minute step: 150 steps is 12.5 h, 490 steps is 41 h. The r122 policy runs the AHUs in 149–199-step blocks (12–17 h) and cycles them 4.4–12.7 times a day; the r116 saves run them 227–748 steps (19–62 h) and flip 1.6–9.8 times a day. Neither chatters at the step level (mean |Δ| ≈0.01), but they are different operating strategies: r122 stops and starts the air handlers, r116 leaves them on.

## The 18:00 cliff is the air handlers switching off

| hour | damper | reheat | SAT | recirc | AHU run on | water |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.418 | 0.008 | 0.143 | 0.833 | 0.99 | 0.322 |
| 17 | 0.398 | -0.071 | 0.131 | 0.825 | 0.99 | 0.283 |
| 18 | 0.577 | -0.242 | 0.357 | 0.779 | 0.17 | 0.343 |
| 19 | 0.195 | -0.291 | 0.692 | 0.654 | 0.00 | 0.673 |

At 18:00 the AHU run fraction drops from 0.99 to **0.17** — the policy shuts the air handlers one hour before occupancy ends — while the damper mean jumps (+0.18) and SAT rises (+0.23). The comfort regret in that hour goes from −0.051 to −0.105 (file 04): the plant is off, the last occupied hour pays for it, and the month still wins. This is a measurable trade the policy made on its own, not a constraint we imposed.

## The AHUs are commanded as one, the zones are not

| building | AHUs | SAT spread across AHUs | recirc spread | damper spread across zones | reheat spread across zones |
| --- | --- | --- | --- | --- | --- |
| building_train_8 | 3 | 0.004 | 0.001 | 0.43 | 0.34 |
| building_train_1 | 4 | 0.011 | 0.007 | 0.41 | 0.39 |
| building_train_6 | 6 | 0.029 | 0.021 | 0.71 | 0.58 |
| building_train_3 | 7 | 0.05 | 0.04 | 0.68 | 0.71 |
| building_train_2 | 9 | 0.02 | 0.026 | 0.5 | 0.45 |

The AHU heads are nearly identical across AHUs (SAT spread 0.004–0.050 on a [−1,1] scale): the policy commands the building as one air system. The zone heads do differ (per-zone damper means span 0.41–0.71), so the zoning is real, but it is a modulation on a common mode.

## Effective dimensionality: 75–209 heads, ~2–7 degrees of freedom

| building | policy | heads | PC1 | PC1–3 | PC1–10 | participation ratio |
| --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | r122@119,439,360 | 75 | 44% | 72% | 86% | 4.1 |
| building_train_8 | r122@89,579,520 | 75 | 37% | 71% | 87% | 4.5 |
| building_train_8 | r116@50,135,040 | 75 | 74% | 90% | 97% | 1.8 |
| building_train_8 | r116@45,121,536 | 75 | 75% | 91% | 97% | 1.7 |
| building_train_1 | r122@119,439,360 | 97 | 43% | 70% | 84% | 4.3 |
| building_train_1 | r122@89,579,520 | 97 | 41% | 73% | 89% | 4.2 |
| building_train_1 | r116@50,135,040 | 97 | 65% | 87% | 95% | 2.2 |
| building_train_1 | r116@45,121,536 | 97 | 64% | 87% | 96% | 2.3 |
| building_train_6 | r122@119,439,360 | 113 | 39% | 75% | 90% | 4.4 |
| building_train_6 | r122@89,579,520 | 113 | 28% | 63% | 83% | 6.5 |
| building_train_6 | r116@50,135,040 | 113 | 67% | 85% | 95% | 2.2 |
| building_train_6 | r116@45,121,536 | 113 | 62% | 84% | 94% | 2.4 |
| building_train_3 | r122@119,439,360 | 171 | 45% | 73% | 89% | 3.9 |
| building_train_3 | r122@89,579,520 | 171 | 39% | 72% | 89% | 4.6 |
| building_train_3 | r116@50,135,040 | 171 | 56% | 83% | 94% | 2.8 |
| building_train_3 | r116@45,121,536 | 171 | 53% | 83% | 94% | 3.0 |
| building_train_2 | r122@119,439,360 | 209 | 44% | 75% | 90% | 4.0 |
| building_train_2 | r122@89,579,520 | 209 | 45% | 79% | 92% | 3.6 |
| building_train_2 | r116@50,135,040 | 209 | 61% | 84% | 94% | 2.5 |
| building_train_2 | r116@45,121,536 | 209 | 61% | 86% | 95% | 2.4 |

PCA over every mapped head and every step: one component explains 28–75% of the action variance, three explain 63–91%, and the participation ratio (effective number of dimensions) is **1.7–6.5**. The 30–85 zone dampers alone have a participation ratio of 1.1–3.4. So the policy uses the full action list — every head moves and 100% of the native actions are controlled — but the realised behaviour is a low-dimensional common mode plus a few zone corrections. That is the honest version of the scale claim, and it also explains why a closed-damper r116 strategy and an open-damper r122 strategy can reach similar scores.

## Cross-checkpoint behaviour distance

| building | vs r122@119.4M | vs r116@50.1M | vs r116@45.1M |
| --- | --- | --- | --- |
| building_train_8 | 0.434 | 0.631 | 0.639 |
| building_train_1 | 0.433 | 0.615 | 0.625 |
| building_train_6 | 0.358 | 0.656 | 0.655 |
| building_train_3 | 0.335 | 0.679 | 0.688 |
| building_train_2 | 0.483 | 0.721 | 0.704 |

Mean |Δa| on the same mapped heads, [0,2] range. The two r122 saves of the same recipe differ by 0.33–0.48 per head — a fifth to a quarter of the full range — while their 90-month scores differ by only 0.002. The action path is not unique; the score is the invariant. r116 is 0.61–0.72 away, and the role table shows why: r116 keeps the dampers more closed (mean −0.39 vs 0.00 on `train_1`) and leans on reheat (+0.34 vs −0.05), the opposite balance from r122.

## What this file adds beyond the action count

- **Coverage proof**: mapped heads = native actions for every recording, padding exactly zero.
- **Utilisation**: which heads move, which are static (the static-pressure setpoint), which are pinned.
- **The 18:00 shutdown** named in actions, not inferred from the score.
- **AHU homogeneity** and per-zone spread, so "171 actuators" is not mistaken for 171 independent decisions.
- **Effective dimensionality** (2–5) — the honest size of the behaviour behind the action list.
- **Behavioural distance between checkpoints**, which shows the two r122 saves are not the same controller and that r116 solved the plant differently.


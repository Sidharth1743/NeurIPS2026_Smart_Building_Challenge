# r14_structured_200m_s75 — 144-scenario analysis (from the recordings)

Score = mean over scenarios of the return; lower is better. Terms are the fleet means of the per-step parts over all 8640 steps x agents.

## Headline

| step | 144 score | productivity | energy | carbon | comfort dev K | worst zone K | elec kWh | gas kWh |
|---|---|---|---|---|---|---|---|---|
| 49,972,800 | **0.042717** | -0.0535 | 0.0267 | 0.0263 | 0.894 | 33.5 | 9084 | 21602 |
| 89,951,040 | **0.048063** | -0.0614 | 0.0276 | 0.0284 | 0.986 | 33.5 | 9174 | 24455 |
| 119,934,720 | **0.067618** | -0.0935 | 0.0289 | 0.0286 | 1.296 | 33.5 | 9909 | 24542 |
| 159,912,960 | **0.063092** | -0.0888 | 0.0266 | 0.0225 | 1.304 | 33.4 | 9530 | 17296 |

## Per building

| step | train_1 | train_2 | train_3 | train_4 | train_5 | train_6 | train_7 | train_8 |
|---|---|---|---|---|---|---|---|---|
| 49,972,800 | 0.0308 | 0.0334 | 0.0840 | 0.0272 | 0.0329 | 0.0624 | 0.0387 | 0.0324 |
| 89,951,040 | 0.0353 | 0.0319 | 0.0971 | 0.0280 | 0.0338 | 0.0841 | 0.0390 | 0.0353 |
| 119,934,720 | 0.0463 | 0.0510 | 0.1378 | 0.0392 | 0.0644 | 0.1102 | 0.0563 | 0.0357 |
| 159,912,960 | 0.0401 | 0.0463 | 0.1450 | 0.0451 | 0.0525 | 0.0784 | 0.0638 | 0.0336 |

## Per weather (mean over the buildings/weathers available)

| step | atlanta | denver | fairbanks | minneapolis | phoenix | singapore |
|---|---|---|---|---|---|---|
| 49,972,800 | 0.0244 | 0.0487 | 0.0487 | 0.0519 | 0.0226 | 0.0326 |
| 89,951,040 | 0.0292 | 0.0520 | 0.0561 | 0.0570 | 0.0269 | 0.0402 |
| 119,934,720 | 0.0454 | 0.0724 | 0.0824 | 0.0777 | 0.0457 | 0.0518 |
| 159,912,960 | 0.0389 | 0.0690 | 0.0688 | 0.0748 | 0.0377 | 0.0543 |

## Scenarios that carry the sum (top 10 at the final step)

| scenario | mean reward | productivity | energy | carbon |
|---|---|---|---|---|
| building_train_3__minneapolis_1 | 0.3189 | -0.4901 | 0.0392 | 0.0850 |
| building_train_7__minneapolis_1 | 0.2965 | -0.4384 | 0.0535 | 0.1134 |
| building_train_3__minneapolis_2 | 0.2166 | -0.3350 | 0.0259 | 0.0521 |
| building_train_3__denver_1 | 0.1950 | -0.3053 | 0.0218 | 0.0372 |
| building_train_5__minneapolis_1 | 0.1916 | -0.2712 | 0.0473 | 0.0970 |
| building_train_3__fairbanks_3 | 0.1868 | -0.2945 | 0.0181 | 0.0323 |
| building_train_6__minneapolis_1 | 0.1790 | -0.2664 | 0.0323 | 0.0634 |
| building_train_4__minneapolis_1 | 0.1669 | -0.2433 | 0.0352 | 0.0695 |
| building_train_7__minneapolis_2 | 0.1631 | -0.2278 | 0.0409 | 0.0910 |
| building_train_3__minneapolis_8 | 0.1390 | -0.2205 | 0.0142 | 0.0194 |

## Time of day (fleet means of the per-step parts)

| bin | 49,972,800 | 89,951,040 | 119,934,720 | 159,912,960 |
|---|---|---|---|---|
| night 19-06 | -0.0217 | -0.0258 | -0.0449 | -0.0438 |
| morning 06-12 | +0.0001 | +0.0002 | +0.0002 | +0.0013 |
| afternoon 12-16 | +0.0001 | +0.0002 | +0.0003 | +0.0013 |
| evening 16-19 | - | - | - | - |
| weekend 06-19 | -0.0278 | -0.0332 | -0.0638 | -0.0622 |

(cell value = 0.6*productivity + 0.2*energy + 0.2*carbon for that bin, so it is the score's arithmetic on the fleet means, not a per-scenario mean.)


## Reading

**The best point of this run is 50 M, not the end.** 144 scenarios: 49.97 M **0.042717**,
89.95 M 0.048063, 159.91 M 0.063092, 119.93 M 0.067618. Every extra 35 M steps costs
0.005–0.02, and the curve is not monotone in between (119.9 M is worse than 159.9 M), so this
is drift, not a slow climb.

**All of the loss is productivity.** −0.0535 at 50 M → −0.0935 at 119.9 M, while comfort dev
goes 0.894 → 1.296 K. Energy does not improve (0.0267 → 0.0289 → 0.0266) and carbon only
improves at the last save (0.0263 at 50 M → 0.0225 at 159.9 M). Paying 0.4 K of comfort and
0.040 of productivity for 0.004 of carbon is a losing trade under 0.6/0.2/0.2.

**One fuel trade is visible.** From 50 M to 159.9 M gas falls 21,602 → 17,296 kWh while
electricity rises 9,084 → 9,530 kWh and carbon falls. That is the direction the paper describes
(electric carbon is the expensive one), but it arrives with the comfort bill, so it does not pay.

**Two buildings carry the whole sum**: train_3 (0.0840 → 0.1450) and train_6 (0.0624 → 0.0784).
train_1/2/4/5/7/8 stay inside 0.027–0.064 across all four saves. On the weather axis,
minneapolis and fairbanks are the expensive ones and atlanta/phoenix the cheap ones, at every
save.

**The worst zone does not move**: 33.5 K at all four checkpoints, min and max. A single zone is
responsible, and it is insensitive to the policy, so `worst_dev_K` cannot rank these saves —
unlike the training panel, where it did discriminate.

**Time of day** (with the convention that an episode starts Sunday 06:00): the arithmetic cost
sits entirely in the bins labelled night and weekend daytime (−0.022 to −0.064), while
06:00–16:00 is ≈ 0 (+0.0001 to +0.0013). The phase is an assumption — `obs.bin` carries the
occupancy features and would settle it — so treat the bin edges as provisional, not the totals.

## Caveats

- One run, one seed; the checkpoints are selected from the training signal at the user's
  instruction. The 144 numbers here are the held-out evidence that the selection at 89.95 M was
  not the best point of the run.
- `parts` worst_dev_K is an instantaneous worst *zone*; the trainer's `env/worst_dev_K` is the
  episode mean of the per-step *mean* deviation. They are different quantities and are not
  comparable.
- These are the 8 buildings of `eval5_heldout18` + `eval3_heldout18`, not the hidden Kaggle set.

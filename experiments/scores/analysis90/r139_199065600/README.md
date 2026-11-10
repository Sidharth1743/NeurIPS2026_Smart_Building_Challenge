> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# r139 @199,065,600 — checkpoint breakdown (parts 1–3)

The r139 recipe trained on the 30-scenario batch, 199.9M steps annealed to zero, seed 75. Checkpoint
**199,065,600** (epoch 720) was the best on both criteria of the training panel: lowest mean and lowest
worst scenario. This directory applies the parent suite's first three analyses to that checkpoint on the
same **90 held-out months** (5 buildings × 18 windows, 8,640 steps each, deterministic).

Headline on the 90: **0.016514**, against r122 @89,579,520 **0.025315** (−34.8%) and `pid_water`
**0.046092** (−64.2%). All three terms improve at once, and both fuels fall overall.

## Files

| file | question it answers |
|---|---|
| `01_score_decomposition.md` | Score, term contributions, per building/station/window, the tail, the variance split, and the paired deltas against r122 and the baseline. |
| `02_terms_and_fuel.md` | Comfort in kelvin and in-band share, then electricity/gas MWh and CO₂ kg per building, against the baseline converted from its own cost. |
| `03_time_structure.md` | Month → day → day-part → hour; the first-24-hour investment, the weekend setback, peaks and the 30-day drift. |
| `04_actions.md` | What the checkpoint does with 75–209 actuators: utilisation, concentration, chattering, the 18:00 behaviour, effective dimensionality, and the behavioural distance from r122. |
| `05_zones.md` | Where the scored comfort regret lives: per zone, per AHU, free float vs scored, the action link, and the same zones under r122. |

## Findings, one line each

- **01 score** — 0.016514, −34.8% vs r122 and −64.2% vs `pid_water`; better on **77/90** scenarios and on
  every building (−10.8% on `train_3` to −75.1% on `train_2`). Terms 42% comfort / 31% energy / 27% carbon.
  Variance is **74% building / 22% weather**; the five worst months are all `train_3`.
- **02 terms and fuel** — occupied in-band 84–98% on four buildings, **47% on `train_3`** (0.997 K mean
  deviation). Electricity falls on every building (45.2 vs 73.3 MWh), gas on 3/5 (89.1 vs 95.6), CO₂ on 4/5
  (**20.6 vs 24.6 t**). Not the electricity→gas trade r122 made.
- **03 time** — the weekday morning is the largest bill (0.0458 vs pid 0.1280); the only bin worse than pid
  is the **weekday night (+20%)**. **No 18:00 step** (unlike r122). Weekends carry 0.4–0.6% of the kWh.
  `train_3` drifts **+3.8×10⁻⁴/day** (≈ +0.011 over the month); the small plants are flat.
- **04 actions** — zone heads are **hedged** (α+β 2.4–2.9 vs r122's 10.5), motion is smooth (0.006–0.009
  |Δa|/step vs 0.03), **hot water is held low** (−0.8 vs +0.44), **no head is static** (r122 froze the
  static-pressure head), the air handlers stay on at 18:00, and effective dimensionality is 2.7–3.9.
- **05 zones** — scored deficit 0.011–0.106 K on four buildings and **0.631 K on `train_3`**; free float is
  an order larger and unscored. `train_6` AHU2 (0.40 vs 0.04 K, 10.9×) and `train_3` AHU4 (1.25 vs 0.37)
  carry the residual; the deficit now correlates **positively with the damper**, so it is a distribution
  limit, not a feeding error. Same weak zones as r122 (rank 0.74–0.94), at lower levels.

## Reproduce

```bash
K=starter-kit/.venv/bin/python
$K results/eval90_analysis/r139_199065600/make_breakdown.py        # 01-03
$K results/eval90_analysis/r139_199065600/make_actions_zones.py    # 04-05
```

Reads the checkpoint's own recording at
`complete_port/results/study_r139_eval90/gpu/r139/0000000199065600/<building>/record/`
(`parts.bin`, `obs.bin`, `mask.bin`), the `pid_water` per-step CSVs, the r122 recording for the paired
delta, and the set tables in `../data/`. Every number in the three markdown files is generated; nothing
is typed by hand.

`data/` holds `scenarios_r139.csv` (90 rows: score, terms and characteristics), `scores_by_building.csv`,
`scores_by_station.csv`, `scores_by_window.csv`, `terms_by_building.csv`, `fuel_vs_pid.csv`,
`daypart.csv`, `daily.csv`, `first24h.csv`, `action_roles.csv`, `action_diurnal.csv`, `action_pca.csv`,
`action_utilization.csv`, `action_distance.csv`, `zone_stats.csv`, `zone_ahu.csv`, `zone_cross.csv`,
`zone_days.csv`.

## Validity

- **One checkpoint, one deterministic episode per scenario.** No seed variance; every distribution is
  across scenarios or time, not across runs.
- **Same 90 scenarios and the same fp64 deterministic evaluator** as the parent analysis, so the numbers
  are directly comparable with files 02–04 there.
- **Zone-level in-band** is decoded from the canonical observation and validated against the recorded
  comfort parts; the all-steps column includes unoccupied free float and is **not** scored — use the
  occupied column.
- **The baseline's MWh and kg** are reconstructed from its recorded cost and the competition's price and
  emission schedules; the reconstruction reproduces its `carbon_emitted` exactly.
- The Kaggle leaderboard row for this checkpoint is the external check; nothing here is a hidden-test
  claim.

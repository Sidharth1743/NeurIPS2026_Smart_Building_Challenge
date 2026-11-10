# 09 — Behaviour of the reported checkpoint (199,065,600)

## Heads by role, all 90 months

| role | heads | mean | median sd | static <0.05 | pinned | median |da| |
|---|---:|---:|---:|---:|---:|---:|
| zone damper | 90 | +0.01 | 0.23 | 0% | 0% | 0.153 |
| zone reheat | 90 | -0.11 | 0.24 | 0% | 0% | 0.150 |
| air supply temp | 9 | -0.02 | 0.39 | 0% | 0% | 0.300 |
| static pressure | 9 | -0.65 | 0.15 | 0% | 0% | 0.060 |
| recirculation | 9 | +0.29 | 0.26 | 0% | 0% | 0.190 |
| air-handler run | 9 | -0.42 | 0.47 | 9% | 0% | 0.101 |
| water temperature | 1 | -0.76 | 0.37 | 1% | 0% | 0.205 |
| pump dP | 1 | -0.29 | 0.36 | 0% | 0% | 0.266 |
| plant run | 1 | -0.68 | 0.43 | 10% | 0% | 0.259 |

## Zones, occupied steps

| building | zones | scored K | free float K | in band | top-5 share | top-10 share |
|---|---:|---:|---:|---:|---:|---:|
| building_train_1 | 39 | 0.011 | 0.514 | 98.3% | 65% | 80% |
| building_train_2 | 85 | 0.027 | 1.045 | 97.4% | 41% | 53% |
| building_train_3 | 70 | 0.631 | 1.205 | 47.1% | 25% | 40% |
| building_train_6 | 43 | 0.106 | 0.566 | 84.4% | 57% | 74% |
| building_train_8 | 30 | 0.012 | 1.302 | 98.4% | 65% | 79% |

| building | worst zone | median zone | worst AHU | corr damper | corr reheat | cold share |
|---|---:|---:|---:|---:|---:|---:|
| building_train_1 | 0.06 | 0.00 | 0.02 | +0.17 | +0.49 | 87% |
| building_train_2 | 0.32 | 0.01 | 0.05 | +0.49 | +0.03 | 68% |
| building_train_3 | 3.78 | 0.51 | 1.25 | +0.77 | -0.51 | 20% |
| building_train_6 | 1.11 | 0.04 | 0.40 | +0.70 | -0.35 | 22% |
| building_train_8 | 0.08 | 0.00 | 0.02 | +0.16 | +0.50 | 80% |

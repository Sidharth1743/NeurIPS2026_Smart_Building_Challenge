# Scores

The deterministic evaluation the paper reports. Every number here is a fixed checkpoint scored over
whole months, one fresh episode per scenario, `env.desync=0`, fp64 solve. The same checkpoint always
reproduces the same value.

## Layout

```
panel90_r139/<run>/<step>/score.json      the 90-month panel for one checkpoint
panel54/<run>/<step>/score.json           the 54-month panel
panel90_r122/<run>/<step>/score.json
panel90_pid_water/metrics.json            the organiser baseline, 90 months
panel54_pid_water/metrics.json            the organiser baseline, 54 months
analysis90/                               the analysis built from the 90-month recording
```

`score.json` holds the leaderboard score of the merged panel plus, under `scenarios`, one entry per
scenario with `mean_reward`, `productivity`, `energy_cost` and `carbon`. The score is
`-0.6 * productivity + 0.2 * energy_cost + 0.2 * carbon`, averaged over the scenarios.

`checkpoint.json` beside it records the checkpoint's step, byte count and sha256, so a score can be
tied to one file. The per-building `score.json` files under each checkpoint are the parts the merged
file was assembled from.

## The numbers in the paper

| policy | run | step | 90 months | 54 months |
|---|---|---|---|---|
| Final | `r139` | 199,065,600 | 0.0165 | 0.0098 |
| Mid | `r139` | 159,252,480 | 0.0185 | 0.0112 |
| Long | `r122` | 89,579,520 | 0.0253 | 0.0192 |
| PID | `pid_water` | | 0.0461 | 0.0437 |

```python
import json
d = json.load(open("panel90_r139/r139/0000000199065600/score.json"))
print(d["score"], d["n_scenarios"], d["complete"])
```

## analysis90

`analysis90/` holds the tables the Results section is built from: the score decomposition, the two
fuels, the time-of-day and month structure, the per-zone deficits, and the action statistics. It is
produced by the scripts in the same folder from the per-step recordings, which are 7.6 GB per
checkpoint and are not shipped; the scores they summarise are.

## Rebuilding one panel

```bash
./run.sh score 19_final_r139_30_scenario_200m 199065600 eval5_heldout18
```

writes to `complete_port/results/` in the same layout, with the per-step recording beside it.

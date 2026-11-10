# Beta shape floor, 1.0 against 0.5

**Change:** policy.beta_floor 1.0 against 0.5, two arms on seed 77 with every other flag identical.

**What the ledger records:** At r81 with the seed held fixed, floor 1.0 reaches 0.129 against 0.222 for floor 0.5, so the floor of 1 looked necessary (30 to 60M 0.403 to 0.199 across seeds, floor 0.5 better on only 1 of 3). At r116 the ordering inverts: floor 0.5 is 6% better and wins 5 of 6 saves (ablation 10).

**Verdict:** the floor is not load-bearing; the reported recipe keeps 0.5.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r81_floor05_s77` | 60,162,048 | 60,162,048 | 0.2223 |
| `r81_floor1_s77` | 60,162,048 | 60,162,048 | 0.1291 |

The panel is `env/episode_score_scen_mean` at the last recorded epoch: a changing-policy
training read, useful for comparing runs inside one family, not a score. It is a different
measure from the number quoted above, which is a frozen month or a reported checkpoint. Frozen
scores are in `../scores/`.

## Files

| file | what |
|---|---|
| `args.txt` | the complete argument list, diff it against another arm to see one change |
| `command.txt` | the launch command, development paths marked |
| `train.log.head`, `train.log.tail` | startup banners and the last epochs |
| `tb/scalars.csv.gz` | every epoch scalar, `tag,step,value` |
| `ckpt/` | `master_weights` dumps (1.8 MB each) with manifests |


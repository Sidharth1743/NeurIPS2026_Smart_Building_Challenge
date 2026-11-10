# Per-head PPO ratio

**Change:** train.headwise_ratio=1 (train.ratio_per_dim was already on), with the training budget cut from 80.2M to 50.135M so every later arm is comparable.

**What the ledger records:** Removing it is 361% worse and never recovers (ablation 02). The budget cut is what makes the 50M ablation grid affordable.

**Verdict:** essential.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r115_headwise_ratio_s75` | 50,135,040 | 50,135,040 | 0.0520 |

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


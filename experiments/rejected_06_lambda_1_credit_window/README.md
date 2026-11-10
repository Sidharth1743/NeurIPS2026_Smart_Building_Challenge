# Credit window at lambda 1

**Change:** train.gae_lambda=1.

**What the ledger records:** 0.237 not held (neighbours 0.515 and 7.618).

**Verdict:** rejected, does not hold.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r36_fairbanks_rl` | 39,813,120 | 28,753,920 | 9.6576 |
| `r59_credit_window_lambda1` | 80,216,064 | 10,027,008 | 0.5115 |

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

Weights are the final save only: the evidence for a rejected lever is its curve, and the
checkpoint is kept so the claim can be re-scored without retraining.

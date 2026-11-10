# Adam on the whole network, old binary

**Change:** replace Muon everywhere.

**What the ledger records:** 0.797 to 9.825 against Muon 1.525 to 0.561. Inverted at r116: plain Adam is the best arm (ablation 12, -12%).

**Verdict:** rejected then, adopted later.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r47_fairbanks_adam` | 30,081,024 | 30,081,024 | 5.4202 |

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

# Role-split second moments

**Change:** train.role_adam=1: separate Adam second moments per parameter role on top of the Muon trunk.

**What the ledger records:** Removing it later costs 48% (ablation 05: 0.049 against 0.033). Superseded by plain Adam everywhere at r139.

**Verdict:** essential until the optimizer swap.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r93_tau_adam_s75` | 80,216,064 | 40,108,032 | 0.2462 |

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


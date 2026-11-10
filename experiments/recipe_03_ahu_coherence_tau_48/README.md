# Longer hold for the air-handler heads, tau 48

**Change:** policy.beta_cn_tau_ahu=48: the air-handler heads hold their Beta try for 48 steps while zone and plant heads keep 12.

**What the ledger records:** Later found inert at r116 (ablation 09: tau 12 is -6%, 6/6 saves), so the split costs nothing but buys nothing at this budget.

**Verdict:** inert at r116.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r91_ahu_tau_s75` | 80,216,064 | 30,081,024 | 0.1409 |

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


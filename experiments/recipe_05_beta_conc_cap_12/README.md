# Concentration cap at 12

**Change:** policy.beta_conc_max=12: alpha+beta is rescaled to 12 when it would exceed it, preserving the mean.

**What the ledger records:** Removed the hot-month lock; panel 0.256 to 0.177 over 8 cells. Confirmed at r116: without the cap the run is 64% worse (ablation 03).

**Verdict:** essential.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r94_conc_cap_s75` | 80,216,064 | 50,135,040 | 0.1888 |

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


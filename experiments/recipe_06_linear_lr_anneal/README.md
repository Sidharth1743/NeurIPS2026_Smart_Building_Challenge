# Linear learning-rate anneal

**Change:** anneal_lr=1, linear_lr_anneal=1, anneal_steps=50135040.

**What the ledger records:** 5-save 0.1648 to 0.1071, final 0.338 to 0.040. Confirmed at r116: constant rate is 15% worse (ablation 07). This is the run named r94b, the starting point of the improvement chain in the paper.

**Verdict:** essential, and the measured control for every arm.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r94b_lr_s75` | 80,216,064 | 50,135,040 | 0.0458 |

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


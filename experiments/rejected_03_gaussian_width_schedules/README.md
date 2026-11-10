# Gaussian width schedules and pins

**Change:** explicit sigma schedules and a frozen-sigma twin.

**What the ledger records:** Walked the mean off the heat; the frozen-sigma twin diverged in 10M.

**Verdict:** rejected.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r13` | 20,000,000 | 6,553,600 | not recorded |
| `r16` | 40,000,000 | 39,976,960 | not recorded |
| `r21` | 300,023,808 | 281,804,800 | not recorded |

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

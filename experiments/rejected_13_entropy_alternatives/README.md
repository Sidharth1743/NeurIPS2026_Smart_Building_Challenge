# Entropy schedules and role entropy

**Change:** entropy targets, annealed coefficients and per-role entropy.

**What the ledger records:** Dose-insensitive collapse: no entropy target shortened the cold basin.

**Verdict:** rejected.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r63_ent_anneal_s73` | 65,175,552 | 65,175,552 | 7.6741 |
| `r82_ent80_s75` | 60,162,048 | 60,162,048 | 0.1738 |
| `r83_entrole_s75` | 60,162,048 | 60,162,048 | 0.2138 |

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

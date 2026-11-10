# Potential-based comfort shaping

**Change:** a potential on the comfort term added to the reward.

**What the ledger records:** 1.058 against 0.327. The reward is the only teacher; a shaping prior was rejected.

**Verdict:** rejected.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r9d_s00` | 8,000,000 | 7,987,200 | not recorded |

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

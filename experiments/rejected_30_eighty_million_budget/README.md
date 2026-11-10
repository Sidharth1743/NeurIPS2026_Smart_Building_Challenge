# 80M budget with a refitted anneal

**Change:** the budget raised to 80M with the anneal stretched.

**What the ledger records:** Telemetry only, no frozen months.

**Verdict:** no evidence, not adopted.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r104_fp64_80m_s75` | 80,216,064 | 80,216,064 | 0.0715 |

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

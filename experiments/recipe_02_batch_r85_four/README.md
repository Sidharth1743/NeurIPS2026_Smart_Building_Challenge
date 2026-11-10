# Batch: one blob to four scenarios

**Change:** the batch goes from one scenario to the four-scenario r85_four set, one group per scenario.

**What the ledger records:** The batch every later 50M arm is measured on. 265 rooms across four buildings in the heavy set; the light set used from r96 has 216.

**Verdict:** the batch of record.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r85_four_s75` | 80,216,064 | 80,216,064 | 0.0620 |

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


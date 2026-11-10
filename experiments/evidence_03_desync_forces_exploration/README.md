# Episode desync on the Fairbanks month

**Change:** env.desync=1: each environment's first episode is cut at a phase hashed from its id, so the batch never sees one calendar state.

**What the ledger records:** 5.07 to 0.330 on one seed and one month at r37. Confirmed at r116 with the largest effect in the campaign: 0.033 to 2.213 (ablation 01).

**Verdict:** essential, and the single largest effect measured.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r37_fairbanks_rl` | 50,135,040 | 50,135,040 | 0.5763 |

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


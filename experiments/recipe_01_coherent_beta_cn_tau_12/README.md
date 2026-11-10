# Coherent Beta draws, tau 12

**Change:** policy.beta_cn_tau=12: the Beta sampler draws AR(1)-correlated normals with a 12-step time constant instead of independent draws per step.

**What the ledger records:** The lever that ended the cold basin: 30 to 60M panel 0.199 to 0.113 on 3 of 3 seeds, worst save 0.512 to 0.141. Later found inert at the r116 budget (ablation 08: -6%, 6/6 saves).

**Verdict:** load-bearing in its era, inert at r116.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r84_cn12_s75` | 60,162,048 | 60,162,048 | 0.1406 |

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


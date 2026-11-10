# Beta head against the squashed Gaussian

**Change:** policy.beta=1: every continuous command is a Beta on (0,1) instead of a tanh-squashed Gaussian.

**What the ledger records:** 0.1784 to 0.1221 on one seed with a five-replay control. The Gaussian never learned to come back from the cold basin; the Beta did, because its support covers the whole action range and it can widen again. Only the Beta arm's artifacts survive in this archive; the control is recorded in ABLATIONS.md.

**Verdict:** adopted; locked at r116 because it cannot be removed without losing the headwise ratio.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r79_beta_s75` | 80,216,064 | 80,216,064 | 0.1283 |

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


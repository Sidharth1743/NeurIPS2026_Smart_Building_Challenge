# Air-handler actor MLP

**Change:** a shared nonlinear branch on the AHU readout.

**What the ledger records:** User-stopped at 11.6M, panel 1.886 against 0.273.

**Verdict:** rejected, stopped.

| run | budget (steps) | last step | final recorded panel |
|---|---|---|---|
| `r119_ahu_mlp_s75` | 50,135,040 | 10,027,008 | 1.4812 |

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

# Baseline r116: the recipe every arm is measured against

**Configuration:** zone_mlp=1 on top of r94b, headwise ratio, Beta head, coherent Beta draws, cap 12, role Adam

**Runs:** `r116_zone_mlp_headwise_s75`, `r116_zone_mlp_headwise_s76`

| seed | 10M | 20M | 30M | 40M | 45M | 50.1M |
|---|---|---|---|---|---|---|

| s75 | 0.273 | 0.126 | 0.052 | 0.036 | 0.034 | 0.033 |
| s76 | 0.336 | 0.146 | 0.072 | 0.044 | 0.039 | 0.035 |

Baseline. Seed spread 6% between s75 and s76, which is the resolution at which any arm's delta is meaningful.


## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

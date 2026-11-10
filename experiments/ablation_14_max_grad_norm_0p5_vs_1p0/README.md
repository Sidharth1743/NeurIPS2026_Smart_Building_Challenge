# Gradient clip 0.5 vs 1.0

**Change against the baseline:** `train.max_grad_norm 0.5 -> 1.0`

**Run:** `r136_maxgrad10_s75`  **Steps:** 50,135,040  **Final panel:** 0.0340  **Delta vs r116 (0.0330):** +3%  **Saves better than baseline:** 3/6

**Verdict:** insensitive.

First-ever test: 0.5 in all 425 launches before it. Doubling the clip is within seed noise, so the trunk is not gradient-bound.

| save | panel |
|---|---|
| 10M | n/a |
| 20M | n/a |
| 30M | n/a |
| 40M | n/a |
| 45M | n/a |
| 50.1M | 0.034 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

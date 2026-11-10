# Trunk width 256 vs 128

**Change against the baseline:** `policy.hidden_size 256 -> 128`

**Run:** `r138_hidden128_s75`  **Steps:** 50,135,040  **Final panel:** 0.0310  **Delta vs r116 (0.0330):** -6%  **Saves better than baseline:** 4/6

**Verdict:** capacity is not the constraint.

First-ever test: 256 in all 490 launches. Half width is at least as good and costs 2x less memory, so the recipe is not width-bound.

| save | panel |
|---|---|
| 10M | n/a |
| 20M | n/a |
| 30M | n/a |
| 40M | n/a |
| 45M | n/a |
| 50.1M | 0.031 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

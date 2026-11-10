# PPO clip coefficient 0.2 vs 0.1

**Change against the baseline:** `train.clip_coef 0.2 -> 0.1`

**Run:** `r135_clip01_s75`  **Steps:** 50,135,040  **Final panel:** 0.0510  **Delta vs r116 (0.0330):** +55%  **Saves better than baseline:** 1/6

**Verdict:** the trust region was right.

First-ever test of the clip coefficient at this recipe: 0.2 in all 424 launches before it. Tightening costs 55%.

| save | panel |
|---|---|
| 10M | n/a |
| 20M | n/a |
| 30M | n/a |
| 40M | n/a |
| 45M | n/a |
| 50.1M | 0.051 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

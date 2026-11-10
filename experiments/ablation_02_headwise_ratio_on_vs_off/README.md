# Per-head PPO ratio on vs off

**Change against the baseline:** `train.ratio_per_dim 1 -> 0 (joint ratio over all heads)`

**Run:** `r125_no_headwise_s75`  **Steps:** 50,135,040  **Final panel:** 0.1520  **Delta vs r116 (0.0330):** +361%  **Saves better than baseline:** 1/6

**Verdict:** essential.

A joint ratio lets large steps on a few heads hide behind near-zero shifts on the others; the run degrades and stalls after 10M.

| save | panel |
|---|---|
| 10M | 0.725 |
| 20M | 0.096 |
| 30M | 0.176 |
| 40M | 0.160 |
| 45M | 0.165 |
| 50.1M | 0.152 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

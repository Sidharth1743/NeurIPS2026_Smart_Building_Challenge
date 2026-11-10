# AHU coherence tau 48 vs 12

**Change against the baseline:** `policy.beta_cn_tau_ahu 48 -> 12`

**Run:** `r126_tau_ahu12_s75`  **Steps:** 50,135,040  **Final panel:** 0.0310  **Delta vs r116 (0.0330):** -6%  **Saves better than baseline:** 6/6

**Verdict:** inert at this budget.

The longer AHU hold is not load-bearing.

| save | panel |
|---|---|
| 10M | 0.223 |
| 20M | 0.066 |
| 30M | 0.038 |
| 40M | 0.030 |
| 45M | 0.031 |
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

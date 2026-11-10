# Beta shape floor 1.0 vs 0.5

**Change against the baseline:** `policy.beta_floor 1.0 -> 0.5`

**Run:** `r133_floor05_s75`  **Steps:** 50,135,040  **Final panel:** 0.0310  **Delta vs r116 (0.0330):** -6%  **Saves better than baseline:** 5/6

**Verdict:** inert, floor claim inverts.

The floor exists so the density can be bimodal. Removing it does not hurt, so the claim that the floor is load-bearing does not hold.

| save | panel |
|---|---|
| 10M | 0.376 |
| 20M | 0.077 |
| 30M | 0.049 |
| 40M | 0.033 |
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

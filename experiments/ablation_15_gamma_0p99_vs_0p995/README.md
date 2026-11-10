# Discount 0.99 vs 0.995

**Change against the baseline:** `train.gamma 0.99 -> 0.995`

**Run:** `r137_gamma995_s75`  **Steps:** 50,135,040  **Final panel:** 0.0380  **Delta vs r116 (0.0330):** +15%  **Saves better than baseline:** 3/6

**Verdict:** 0.99 was right.

The longer horizon learns faster to 30M (0.158 against 0.273 at 10M) and converges worse after 40M.

| save | panel |
|---|---|
| 10M | 0.158 |
| 20M | n/a |
| 30M | n/a |
| 40M | n/a |
| 45M | n/a |
| 50.1M | 0.038 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

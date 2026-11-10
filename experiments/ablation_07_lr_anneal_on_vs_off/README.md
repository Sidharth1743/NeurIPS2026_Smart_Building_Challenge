# Linear LR anneal on vs off

**Change against the baseline:** `anneal_lr 1 -> 0 (constant 1e-4)`

**Run:** `r131_no_anneal_s75`  **Steps:** 50,135,040  **Final panel:** 0.0380  **Delta vs r116 (0.0330):** +15%  **Saves better than baseline:** 1/6

**Verdict:** helps late.

Constant rate reaches a similar level by 45M but never settles as low, and holds 0.038 rather than the annealed 0.033.

| save | panel |
|---|---|
| 10M | 0.355 |
| 20M | 0.074 |
| 30M | 0.070 |
| 40M | 0.052 |
| 45M | 0.038 |
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

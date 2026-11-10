# Role-split Adam on vs off

**Change against the baseline:** `train.role_adam 1 -> 0`

**Run:** `r128_no_role_adam_s75`  **Steps:** 50,135,040  **Final panel:** 0.0490  **Delta vs r116 (0.0330):** +48%  **Saves better than baseline:** 0/6

**Verdict:** essential.

Per-role second moments matter early. Removing them costs more than removing the whole Muon stack (see 12).

| save | panel |
|---|---|
| 10M | 0.803 |
| 20M | 0.137 |
| 30M | 0.079 |
| 40M | 0.055 |
| 45M | 0.049 |
| 50.1M | 0.049 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

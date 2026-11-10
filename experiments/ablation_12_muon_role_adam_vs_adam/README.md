# Muon with role-split Adam vs plain Adam everywhere

**Change against the baseline:** `train.adam 0 -> 1, train.role_adam 1 -> 0`

**Run:** `r130_adam_s75`  **Steps:** 50,135,040  **Final panel:** 0.0290  **Delta vs r116 (0.0330):** -12%  **Saves better than baseline:** 6/6

**Verdict:** the Muon stack is a liability.

Best single arm of the campaign. Plain Adam dominates the orthogonalised update on every save and removes a large amount of optimizer state.

| save | panel |
|---|---|
| 10M | 0.106 |
| 20M | 0.044 |
| 30M | 0.035 |
| 40M | 0.032 |
| 45M | 0.029 |
| 50.1M | 0.029 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

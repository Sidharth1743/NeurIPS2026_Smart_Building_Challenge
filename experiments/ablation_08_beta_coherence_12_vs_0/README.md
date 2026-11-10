# Coherent Beta draws (tau 12) vs independent draws

**Change against the baseline:** `policy.beta_cn_tau 12 -> 0 (and tau_ahu 48 -> 0, forced)`

**Run:** `r124_no_coherence_s75`  **Steps:** 50,135,040  **Final panel:** 0.0310  **Delta vs r116 (0.0330):** -6%  **Saves better than baseline:** 6/6

**Verdict:** inert at this budget.

Beats the baseline at every save. The earlier attribution of the cold basin to independent draws does not survive this arm at 50M.

| save | panel |
|---|---|
| 10M | 0.063 |
| 20M | 0.047 |
| 30M | 0.036 |
| 40M | 0.033 |
| 45M | 0.032 |
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

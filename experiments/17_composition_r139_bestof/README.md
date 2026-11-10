# Best-of composition: the five mechanisms that beat r116, combined

**Changes against the baseline:**

- train.adam 1, train.role_adam 0 (12, -12%)
- policy.beta_cn_tau 0, beta_cn_tau_ahu 0 (08, -6%)
- policy.value_delay 0 (11, -9%)
- policy.beta_floor 0.5 (10, -6%)

**Run:** `r139_bestof_s75`  **Steps:** 50,135,040

**Verdict:** sub-additive but adopted.

The additive prediction was 0.022; the measured 0.028 means the five mechanisms overlap. Held: last five readings 0.026, 0.027, 0.027, 0.028, 0.028. This is the architecture and recipe of the reported policy.

| save | panel |
|---|---|
| 10M | 0.048 |
| 20M | 0.042 |
| 30M | 0.036 |
| 40M | 0.030 |
| 45M | 0.028 |
| 50.1M | 0.028 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

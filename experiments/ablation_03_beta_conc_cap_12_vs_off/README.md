# Beta concentration cap 12 vs no cap

**Change against the baseline:** `policy.beta_conc_max 12 -> 0`

**Run:** `r127_no_conc_cap_s75`  **Steps:** 50,135,040  **Final panel:** 0.0540  **Delta vs r116 (0.0330):** +64%  **Saves better than baseline:** 0/6

**Verdict:** essential.

Without the cap the density sharpens without bound and exploration collapses in the first few million steps.

| save | panel |
|---|---|
| 10M | 2.180 |
| 20M | 0.168 |
| 30M | 0.150 |
| 40M | 0.070 |
| 45M | 0.060 |
| 50.1M | 0.054 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

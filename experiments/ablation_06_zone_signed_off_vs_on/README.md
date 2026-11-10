# Signed zone deviations off vs on

**Change against the baseline:** `policy.zone_signed 0 -> 1`

**Run:** `r134_zone_signed_s75`  **Steps:** 50,135,040  **Final panel:** 0.0440  **Delta vs r116 (0.0330):** +33%  **Saves better than baseline:** 0/6

**Verdict:** rectified magnitudes are right.

Signing the zone deviations slows every save. The rectified magnitudes carry the useful signal.

| save | panel |
|---|---|
| 10M | 0.718 |
| 20M | 0.136 |
| 30M | 0.083 |
| 40M | 0.047 |
| 45M | 0.043 |
| 50.1M | 0.044 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

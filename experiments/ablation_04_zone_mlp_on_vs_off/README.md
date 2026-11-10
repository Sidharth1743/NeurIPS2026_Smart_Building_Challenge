# Zone MLP branch on vs off

**Change against the baseline:** `policy.zone_mlp 1 -> 0`

**Run:** `r123_no_zone_mlp_s75`  **Steps:** 50,135,040  **Final panel:** 0.0520  **Delta vs r116 (0.0330):** +58%  **Saves better than baseline:** 1/6

**Verdict:** essential.

The shared 32-unit nonlinear branch on the zone readout is what buys the early learning; without it the first two saves are 2x to 4x worse.

| save | panel |
|---|---|
| 10M | 0.492 |
| 20M | 0.082 |
| 30M | 0.065 |
| 40M | 0.055 |
| 45M | 0.053 |
| 50.1M | 0.052 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

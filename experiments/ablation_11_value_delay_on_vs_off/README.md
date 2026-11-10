# Critic delay summary on vs off

**Change against the baseline:** `policy.value_delay 1 -> 0`

**Run:** `r132_no_value_delay_s75`  **Steps:** 50,135,040  **Final panel:** 0.0300  **Delta vs r116 (0.0330):** -9%  **Saves better than baseline:** 6/6

**Verdict:** inert at this budget.

Eight critic inputs are not needed at 50M on four scenarios. The delay features still matter for the actor's own observation, which is a different mechanism.

| save | panel |
|---|---|
| 10M | 0.272 |
| 20M | 0.074 |
| 30M | 0.047 |
| 40M | 0.034 |
| 45M | 0.032 |
| 50.1M | 0.030 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

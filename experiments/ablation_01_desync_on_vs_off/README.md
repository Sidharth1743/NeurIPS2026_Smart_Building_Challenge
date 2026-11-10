# Episode desync on vs off

**Change against the baseline:** `env.desync 1 -> 0`

**Run:** `r129_no_desync_s75`  **Steps:** 50,135,040  **Final panel:** 2.2130  **Delta vs r116 (0.0330):** +6606%  **Saves better than baseline:** 0/6

**Verdict:** essential.

Largest effect in the campaign. With desync off every env starts at the same Sunday 06:00 phase, so the advantage cannot separate seasonal from behavioural error, and the cut was also the only source of episode-length diversity. Never recovers.

| save | panel |
|---|---|
| 10M | 2.641 |
| 20M | 2.301 |
| 30M | 2.290 |
| 40M | 2.083 |
| 45M | 2.158 |
| 50.1M | 2.213 |

## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

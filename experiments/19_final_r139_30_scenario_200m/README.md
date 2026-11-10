# The reported policy: r139 recipe at 30 scenarios, 199.9M steps

**Changes against the baseline:**

- r139 recipe
- budget 199,895,040 steps, linear anneal to zero
- 960 agents, horizon 288, minibatch 2304

**Run:** `r139_train30_200m_s75`  **Steps:** 199,065,600

**Verdict:** Final and Mid.

Step 199,065,600 is Final and step 159,252,480 is Mid in the paper. Both are shipped, each named with the measure it wins: Final leads the 90-month and 54-month panels, Mid leads the public leaderboard entry.


## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

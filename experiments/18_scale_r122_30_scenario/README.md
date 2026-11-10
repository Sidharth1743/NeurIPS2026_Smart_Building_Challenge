# Scale-up to 30 scenarios and 960 agents (r122)

**Changes against the baseline:**

- batch 4 x 64 -> 30 scenarios x 32 (960 agents)
- budget 1.5B, stopped at 507.6M

**Run:** `r122_train30_portable_s75`  **Steps:** 89,579,520

**Verdict:** scale-up evidence.

The run that proves the recipe holds at 960 agents on six buildings. Best Kaggle row 0.01818 at step 89,579,520; the policy degrades after that, which is why the reported policy is not the longest run. Long in the paper.


## Files

| file | what |
|---|---|
| `args.txt` | the complete `./puffer train` argument list this run used |
| `command.txt` | the launch command, including the container |
| `train.log.head` | the startup banners: which mechanisms were on |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every TensorBoard scalar, `tag,step,value` |
| `ckpt/` | the run's checkpoints (`master_weights`, 1.8 MB each) |

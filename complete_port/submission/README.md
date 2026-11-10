# Exported policies

Two self-contained Kaggle policies, numpy only, weights beside the script.

| folder | policy | run | step | 90-month | 54-month | public |
|---|---|---|---|---|---|---|
| `final_r139_199065600/` | Final | r139 30-scenario | 199,065,600 | **0.0165** | **0.0098** | 0.0294 |
| `long_r122_268738560/` | Long | r122 30-scenario | 268,738,560 | 0.0253 | 0.0192 | 0.0182 |

The folder names carry the checkpoint step, and the weights are the trainer's flat
`master_weights` dump, so `model/weights.bin` here is byte-identical to the run's
checkpoint of the same step in `experiments/`.

`residual_policy.py` and `pidw_base.py` are the earlier residual interface, kept because
`tools/export_submission.sh` still writes it.

Run one locally with the gateway in `kaggle_evaluation/`:

```bash
cd final_r139_199065600 && python submission.py
```

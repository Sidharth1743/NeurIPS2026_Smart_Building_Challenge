# Kaggle submission path for the SAC policy

Numpy only, no torch in the kernel (`enable_gpu: false`, `enable_internet: false`). The policy
is the trained actor exactly: `tanh(mu)` of the structured readouts, 219 heads masked by the
contract, then the canonical 219-head vector decoded to the building's `continuous_action` /
`discrete_action`.

## Files

| file | what |
|---|---|
| `build_submission.py` | assembles `submission.py` from two proven sources (see below) |
| `submission.py` | the kernel body (generated — do not hand-edit) |
| `export_weights.py` | checkpoint `.pt` → flat float32 `weights.bin` (36,518 floats, 146,072 B) |
| `run_month.py` | one deterministic month of a `weights.bin` through the official Python simulator |
| `verify_actor.py` | torch actor vs numpy actor on the same driver observations (tolerance 1e-4) |
| `<name>/{model,dataset,kernel}/` | one directory per submitted checkpoint |

`build_submission.py` copies, verbatim:

- `results/r139_submission/submission.py` — the canonical 983-value observation built from
  `building_meta_data`, the metadata layout and normalisation, the 219-head → gateway action
  decode, the `/kaggle/input` weight finder. Proven on the leaderboard.
- `SAC/train/features.py` — the numpy reference for the pool / zone / AHU / plant feature
  blocks, already gated against the torch trainer at 2.2e-16 by `SAC/tools/check_features.py`.

Only the actor forward (7 linear layers) and its weight parser are new.

## Submitting a checkpoint

```bash
cd SAC
.venv/bin/python kaggle/export_weights.py results/<run>/ckpt/<step>.pt kaggle/<name>/model
mkdir -p kaggle/<name>/{dataset,kernel}
cp kaggle/<name>/model/weights.bin   kaggle/<name>/dataset/
cp kaggle/submission.py              kaggle/<name>/kernel/
# dataset-metadata.json + kernel-metadata.json: copy an existing pair, change id/title
cd ..
set -a; . ./.env; set +a
starter-kit/.venv/bin/kaggle datasets create -p SAC/kaggle/<name>/dataset
starter-kit/.venv/bin/kaggle datasets status sidharth1743/<name>-model      # wait for ready
starter-kit/.venv/bin/kaggle kernels push    -p SAC/kaggle/<name>/kernel
starter-kit/.venv/bin/kaggle kernels status  sidharth1743/<name>            # wait for COMPLETE
starter-kit/.venv/bin/kaggle competitions submit -c neurips-2026-smart-buildings \
    -f submission.parquet -k sidharth1743/<name> -v 1 -m "<what this checkpoint is>"
```

The head travels with the weights: `export_weights.py` writes `weights.json` beside
`weights.bin` (`head`, `beta_floor`, `beta_conc_max`), the kernel reads it via `_sidecar`, and
`_actor_forward` decodes `tanh(mean)` for `gauss` or `2a/(a+b)-1` for `beta`. A missing sidecar
means gauss. Both heads are verified against the torch actor on recorded observations
(`verify_actor.py`, tolerance 1e-4; measured 5.7e-6 for the Beta head).

`-f` is the **name** of the kernel's output file, not a path — passing a path returns
`400 Client Error … CreateCodeSubmission`.

Two gates before a submission counts as good:

1. the kernel log contains `loaded weights /kaggle/input/…/weights.bin floats 36518`;
2. the same log's `[gateway] … mean reward=` is nowhere near the neutral action (≈ −0.30).

## Submitted (2026-11-12)

| ref | checkpoint | kernel | commit-log reward |
|---|---|---|---|
| 58257916 | r14 89,951,040 | `submission-sac-r14-90m` | −0.0221 |
| 58259442 | r14 159,912,960 | `submission-sac-r14-159m` | −0.0218 |
| 58259449 | Modal 199,895,040 | `submission-sac-modal-200m` | −0.0250 |
| — | r14 49,972,800 | `submission-sac-r14-50m` | kernel was still running when this was written |
| **58276615** | **r15 Beta 50,135,040** | `submission-sac-r15-beta-50m` | **−0.0138** (`head beta floor 0.5 cap 12.0`) |

All leaderboard rows were `PENDING` when this was written.

## Local check of one checkpoint

```bash
SAC/../starter-kit/.venv/bin/python SAC/kaggle/run_month.py \
    SAC/kaggle/model/weights.bin building_train_1/atlanta_2 8640
```

prints the competition score, the three terms and the action statistics of the month. The
starter-kit venv is needed only for `pandas`/`numba` in the Python simulator; the kernel path
itself needs numpy alone.

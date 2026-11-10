# What this release changes

Nothing. Every file under `complete_port/`, `kaggle_evaluation/`, `data/`, `experiments/` and
`paper/` is the development tree as it stands, copied. The sources were fixed instead of the
copies patched: the tools take `SB_CP`, `SB_ROOT` and `SB_PL_ROOT` with a default derived from
`__file__`, so a checkout runs anywhere and no file differs from the tree the gates ran on.

The assembler refuses to finish if any shipped script still names a development path, which is
how the last stale default was caught.

## Files this release adds

| file | what |
|---|---|
| `run.sh` | the entry point: wraps the tools above, adds no logic of its own |
| `README.md`, `docs/`, `STARTER_KIT.md`, `CITATION.cff`, `LICENSE`, `NOTICE` | the documents |
| `data/` | scenario lists, blob manifest, per-blob metadata, the data card |
| `experiments/**/README.md` | one per run, generated from `args.txt`, the scalar series and the ledger |
| `experiments/_tools/tb_to_csv.py` | event file to CSV, standard library only |
| `experiments/scores/*/README.md`, `metrics.json` | the merged `pid_water` rows, which the workspace kept only per scenario |
| `paper/build.sh` | rebuilds both PDFs |
| `docs/MANIFEST.md`, `SHA256SUMS` | every file, its size and its digest |


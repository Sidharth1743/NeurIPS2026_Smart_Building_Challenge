# Release edits

Every file below differs from the development workspace only in the path handling it needs
to run outside `/data`. No algorithm, physics, reward or hyperparameter is touched, and the
edits are applied to the copies in this release, never to the originals.

| file | change |
|---|---|
| `tools/verify_all.sh` | SB_PL_ROOT replaces the /data/puffer-work literal, SB_CP replaces the workspace root |
| `tools/eval_policy.sh` | SB_PL_ROOT default for the PufferLib checkout |
| `tools/run_train.sh` | SB_PL_ROOT default for the PufferLib checkout |
| `tools/run_pure_chain.sh` | SB_PL_ROOT default for the PufferLib checkout |
| `tools/tb.sh` | SB_PL_ROOT default for the TensorBoard log directory |
| `tools/check_random_start.py` | SB_PL_ROOT replaces the /data/puffer-work literal |
| `tools/check_ahu_mlp_trainer.py` | SB_PL_ROOT replaces the /data/puffer-work literal |
| `tools/check_zone_residual_trainer.py` | SB_PL_ROOT replaces the /data/puffer-work literal |
| `tools/check_zone_pass_trainer.py` | SB_PL_ROOT replaces the /data/puffer-work literal |
| `tools/check_full_resume.py` | SB_PL_ROOT default for --checkout |
| `tools/score_eval90.py` | SB_CP replaces the workspace literal |
| `tools/score_cpu_pid.py` | SB_ROOT replaces the workspace literal |
| `paper-writing/results_fig_data.py` | SB_CP replaces the workspace literal (recordings are not shipped) |
| `complete_port/csim/gate.sh` | development workspace literal replaced by a depth-independent SB_ROOT |
| `complete_port/pysim/bake_all.sh` | development workspace literal replaced by a depth-independent SB_ROOT |
| `complete_port/pysim/diag_bridge.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `complete_port/pysim/diag_replay.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `complete_port/pysim/sweep_scores.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `complete_port/submission/final_r139_199065600/run_month.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `paper/figures/make_appendix_fig_fg.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `paper/figures/make_appendix_fig_h.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `paper/figures/make_appendix_fig_recipe.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `paper/figures/make_fig_ablation_grid.py` | development workspace literal replaced by a depth-independent SB_ROOT |
| `paper/figures/make_fig_results_scatter.py` | development workspace literal replaced by a depth-independent SB_ROOT |

Overrides, all optional:

- `SB_CP` (Python) and `SB_CP` (shell) point at `complete_port/`.
- `SB_PL_ROOT` points at the directory that holds the PufferLib checkouts (default `/data/puffer-work`, created by `./run.sh setup`).
- `PUFFERLIB` points at one patched checkout, as `tools/train.sh` already required.
- `SB_ROOT` is the workspace root, used by `tools/score_cpu_pid.py`.

## Files this release adds

Nothing under `complete_port/` is new code: every file there is the development tree.
The release adds only the entry point, the documents and the archives:

| file | what |
|---|---|
| `run.sh` | the six commands; wraps the tools above, adds no logic of its own |
| `README.md`, `docs/`, `STARTER_KIT.md`, `CITATION.cff`, `LICENSE` | the documents |
| `data/` | scenario lists, blob manifest, per-blob metadata, the data card |
| `experiments/**/README.md` | one per run, generated from `args.txt`, the scalar series and the ledger |
| `experiments/_tools/tb_to_csv.py` | event file to CSV, standard library only |
| `experiments/scores/*/README.md`, `metrics.json` | the merged `pid_water` rows, which the workspace kept only per scenario |
| `paper/build.sh` | rebuilds both PDFs |
| `MANIFEST.md`, `SHA256SUMS` | every file, its size and its digest |

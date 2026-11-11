# pysim

The Python side of the port: it packs a scenario into the binary blob the GPU reads, and it dumps the
oracle traces the gates compare against. Nothing here simulates for training; the C core does that.

All of it runs from the **starter kit directory**, with the starter kit's interpreter, because the
scenario configs are relative to it:

```bash
cd starter-kit && .venv/bin/python ../complete_port/pysim/bake_scenario.py --help
```

`tools/bake_training_set.sh` and `tools/bake_eval_set.sh` wrap the two common cases.

| tool | usage | notes |
|---|---|---|
| `bake_scenario.py` | `--out <dir> --name <building>__<weather> <config.gin>` | packs geometry, solver matrices, convection tables, HVAC statics, the schedule and the exogenous traces into one `.sbb`. `--random-start-days 7` bakes the desync phase table instead of the fixed start. Deterministic: the same config gives a byte-identical blob |
| `bake_env.py` | imported | the HVAC, reward, schedule and exogenous half of the bake; called by `bake_scenario.py` |
| `sbb.py` | imported | the SBB1 container format: header, sections, checksums |
| `dump_step.py` | `<config.gin> <out.step> <steps>` | the actions-only oracle trace the gates replay |
| `dump_episode.py` | `--help` | a full 8,640-step episode driven by a real baseline policy |
| `dump_fdm.py` | `--help` | per-step inputs and outputs of the finite-difference step |
| `check_l0.py` | `--help` | the baked blob must equal the live scenario exactly |
| `sweep_scores.py` | `--pycheck <building> <weather> [steps] [policy]`, or `--policies bang_bang,pid --scenarios 3 --workers 3 [--steps 8640]` | scores the shipped baselines through the verified C core and compares with `reference_baselines.json`; the root is resolved from `__file__`, `SB_ROOT` overrides it |
| `diag_bridge.py`, `diag_replay.py` | `--help` | lockstep diagnostics of the oracle against the C bridge |
| `bake_all.sh` | `tools/bake_all.sh` | bakes every scenario in `training_set.yaml` into `csim/baked`; `SB_ROOT` is resolved from the script path |

# Verification record

Two rounds, both from a clean copy of this repository, never from the development workspace.

## Round 1: the gates, on a 12 GB card

| | |
|---|---|
| date | 2026-11-10, 15:49 to 16:23 UTC |
| host | Linux, one NVIDIA RTX 3060, 12 GB |
| container | `pufferai/puffertank:5.0` (CUDA 13.0.88) |
| copy | the tracked files of this repository, `rsync`ed to an empty directory, no build output |
| upstreams | starter kit cloned at `23a4560`, PufferLib cloned at `6ffa5b1` and patched, both by `./run.sh setup` |
| commands | `./run.sh setup`, `./run.sh data`, `SB_GPU=1 ./run.sh verify` |
| result | **67 checks passed, 0 failed, `ALL CHECKS PASSED`** in 34 minutes |

## Round 2: the whole pipeline, on a 24 GB card

| | |
|---|---|
| date | 2026-11-10 19:00 to 2026-11-11 03:30 UTC |
| host | one NVIDIA RTX 3090, 24 GB; 12 cores |
| commands | `./run.sh verify`, `./run.sh data eval5_heldout18`, `./run.sh score`, `./run.sh eval`, the exported policy through the organiser simulator |
| gates | **67 passed, 0 failed, `ALL CHECKS PASSED`** |
| the reported policy, 90 months | **0.016514** over 90/90 scenarios, complete. The paper reports 0.0165 |
| one scenario, `./run.sh eval` | 0.008134 on `building_train_1/atlanta_4` |
| the same month through the gateway and `submission.py` | 0.008189 |

The last two rows are two independent paths to the same month: the GPU environment with the canonical
observation rebuild, and the organiser's Python simulator driven by the exported policy. They agree to
0.7%, which is the check that matters for the leaderboard path.

The evaluation sets need 144 scenario blobs, 53 GB. They bake cleanly: 90/90 and 54/54, no failures.
Two of the five buildings in the panels, `building_train_3` and `building_train_6`, are generated ones
that the starter kit does not ship; their YAML configs are in `data/generated_buildings/` and
`bake_eval_set.sh` installs them before generating. A rebuilt `building_train_3__atlanta_4.sbb` is
byte-identical to the development blob, so the panel is exactly reproducible.

## Reproducing a score from a 50M run

As a last exercise the release trained the reported recipe from scratch for 50,135,040 steps and
scored it on both panels. It ran the whole chain without intervention:

| checkpoint | agents | steps | 90 months | 54 months |
|---|---|---|---|---|
| r140, fresh | 480 | 50,042,880 | 0.0222 | 0.0143 |
| the paper's Final | 960 | 199,065,600 | 0.0165 | 0.0098 |
| the paper's Mid | 960 | 159,252,480 | 0.0185 | 0.0112 |
| the paper's Long | 960 | 89,579,520 | 0.0253 | 0.0192 |

The 50M run beats Long and loses to Mid and Final. Two caveats, both structural: it differs from the
Final in batch as well as budget (see the memory note below), and it is one seed. Its own training
panel read 0.0086 at 50M, better than the Final's 0.0109, and the frozen panels say the opposite,
which is why a training panel is not a score.

## What the gates cover

One line per check, all passing: the oracle trace match on all six shipped shapes, multi-episode
against the CPU core with host reset and in-step restart, `sb_lu_solve` bit identical at 5 and 40
environments, convection bit identical at `p=1e-4`, `0.5` and `1.0`, the `SB_LU_EXACT` build, the fp32
training solve with bounded drift, the two generated buildings, the weather merge, the fused
six-building solve, the trainer and shared-policy graphs, and both viewer checks.

## Two measured limits

- The **shared canonical policy** that the reported recipe uses does not fit 960 agents in 24 GB: the
  Beta head doubles the output block and the allocation exceeds 23.6 GB of free memory. 480 agents
  (6 groups x 80) fits and runs at 7,045 steps/s, so 50.1M steps is about 1.8 hours. The reported run
  used 960 agents on a 46 GB L40S.
- A 12 GB card needs `--env.groups=3 --vec.total_agents=480` even for the plain policy, because the
  CUDA graph capture for the larger layout does not fit.

## What this does not prove

It does not retrain the reported policy: 199,895,040 steps on 960 agents needs a 46 GB card. It does
not touch a real building, and it does not see the hidden evaluation set. Everything above is inside
the organiser's simulator.

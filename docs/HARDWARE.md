# Hardware and software

Measured on the machine that produced the paper.

## What the paper's numbers were produced on

| | |
|---|---|
| training | one L40S, 46 GB, through Modal; 199,895,040 steps in 4.6 h for the reported run |
| evaluation | one RTX 3090, 24 GB; the 90-month panel is about 10 minutes |
| development and the ablation set | one RTX 3090 (GPU 0), a second RTX 3060 12 GB as a second experiment slot |
| container | `pufferai/puffertank:5.0` (CUDA 13.0.88 inside) |
| host | the host `nvcc` (12.0) is too old; every build goes through the container |

## Throughput, measured

Steps per second for the shape the paper trains on (30 scenarios, 960 agents, horizon 288,
minibatch 2304), from the trainer's own `Steps`/`Uptime` pair rather than the dashboard `SPS` field,
which counts issued steps against CPU-side phase times and is wrong in every regime.

| batch | solve | steps/s | notes |
|---|---|---|---|
| 4 scenarios x 64, light set | fp32 | 9,400 | training-only solve |
| 4 scenarios x 64, light set | fp64 | 7,818 | the reported recipe's batch |
| 30 scenarios x 32, 960 agents | fp64 | 17,400 | the paper's training shape on one 3090 |
| 30 scenarios x 32, 960 agents | fp32 | 23,400 | plus 35% |

The fp32 solve is for training only: it changes the last digits of the physics, which is fine for
gradients and not acceptable for a reported score. Every number in the paper comes from the fp64
build, and `tools/eval_policy.sh` refuses to score with a binary that contains the fp32 solve symbol.

## Memory

| job | VRAM |
|---|---|
| 960 agents, fp64, CUDA graphs | 16 to 18 GB |
| 960 agents, fp32 | 11 GB |
| 480 agents (the 12 GB card) | 6.3 to 6.6 GB |
| one evaluation group, 18 scenarios | 4 to 6 GB |

A 12 GB card can train, but not the 960-agent fp64 batch: graph capture for that layout does not fit.
Use `--env.groups=3 --vec.total_agents=480`.

## Reproducing the environment

```bash
./run.sh doctor    # what is present, what is missing
./run.sh setup     # pulls the container, the oracle env and the PufferLib checkout
```

The container image is public. The PufferLib base commit is in
`complete_port/patches/pufferlib_base_commit.txt` and the patch is
`complete_port/patches/pufferlib_core.diff`; the checkout is never edited directly.

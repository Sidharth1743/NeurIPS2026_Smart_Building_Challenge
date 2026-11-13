# One Shared Bounded-Beta PPO Policy for Zero-Shot Commercial HVAC Control

Commercial HVAC is a large share of building energy and carbon, and the same controller rarely works in the next building. Layouts, climates, occupancy, and plant sizes differ, so practice tunes a controller per site, and most learned controllers are fine-tuned on the building they will run.

This repository is a single Proximal Policy Optimization (PPO) policy for that problem. It is trained only on the [NeurIPS 2026 Smart Buildings Challenge](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/) reward, which scores comfort, energy cost, and carbon together. The observation carries zone and plant sensors. It does not carry a building id, a weather name, or a month. One set of weights is applied, unchanged, to buildings of 41 to 81 rooms and one to four air handlers.

Each actuator is a bounded Beta distribution over its full legal range, so a damper or valve can sit open, shut, or anywhere between, and the distribution can widen again after it narrows. A squashed Gaussian on the same task collapses to one end of the range and stays there. Trained on 30 scenarios (six layouts, five climates) and evaluated on held-out weather, the policy scores **0.0165** on 90 months and **0.0098** on 54 more. The shipped PID controller scores 0.0461 and 0.0437 on those same panels. Lower is better. Zero would be occupied zones inside the comfort band with no energy and no carbon. The paper is in [`paper/`](paper/).

## Dependencies

| Requirement | Notes |
|---|---|
| NVIDIA GPU | 24 GB, compute capability 8.6 or newer. Training used one L40S (46 GB). A 12 GB card can run the verification gates. |
| Driver | New enough for CUDA 13, plus `nvidia-container-toolkit`. |
| Docker | CUDA builds run inside `pufferai/puffertank:5.0`. `./run.sh setup` pulls it. The host `nvcc` is not used. |
| `uv` | Builds the Python environments. |
| Python | 3.12 for the organiser simulator (`numba` has no 3.13 wheels). 3.11 or newer for the rest. |
| `tectonic` | Only for `./run.sh paper`. |
| Disk | About 10 GB for the code and the 30 training scenarios. About 30 GB if the evaluation sets are baked as well. |
| Network | The first `setup` clones [PufferLib](https://github.com/PufferAI/PufferLib) and the [starter kit](https://github.com/smart-buildings-2026/starter-kit). After that, training and evaluation are local. |

`./run.sh doctor` prints which of these are present. Reading the paper, the score files, and the experiment notes does not need a GPU.

Optional environment variables, read by every command:

| Variable | Default | Meaning |
|---|---|---|
| `SB_IMAGE` | `pufferai/puffertank:5.0` | Container image. |
| `SB_PL_ROOT` | `~/.cache/sb/pufferlib` | Where the PufferLib checkout is stored. |
| `PUFFERLIB` | `$SB_PL_ROOT/plverify` | The patched checkout the trainer uses. |
| `SB_GPU` | all devices | Device index, for example `0`. |

## Run

All commands run from the repository root. `./run.sh help` prints the list. Extra arguments are forwarded to the tool underneath.

```bash
./run.sh setup      # container, Python 3.12 env, PufferLib at the pinned commit, our patch
./run.sh data       # bake the 30 training scenarios
./run.sh verify     # build the simulator and run the correctness gates
./run.sh train      # PPO on the 30 scenarios
./run.sh eval       # one deterministic month per scenario in a set
./run.sh score      # a held-out panel for one checkpoint
./run.sh reproduce  # data, verify, then the reported 90-month panel
./run.sh paper      # rebuild the PDFs
./run.sh doctor     # what is installed and what is missing
```

| Command | Arguments | Example |
|---|---|---|
| `setup` | `--image <image>` `--pl-root <dir>` | `./run.sh setup --pl-root ~/pufferlib` |
| `data` | `[scenario-set] [jobs]` | `./run.sh data eval5_heldout18 4` |
| `verify` | flags of `complete_port/tools/verify_all.sh` | `SB_GPU=0 ./run.sh verify` |
| `train` | `[scenario] [trainer flags...]` | `./run.sh train --env.max_steps=1000` |
| `eval` | `<set> <out.json> <checkpoint> [run] [trainer flags...]` | `./run.sh eval eval5_heldout18 out.json ckpt.bin 19_final_r139_30_scenario_200m` |
| `score` | `<run> <step> [set] [scorer flags...]` | `./run.sh score 19_final_r139_30_scenario_200m 199065600 eval3_heldout18` |
| `reproduce` | `[set] [jobs]` | `./run.sh reproduce eval3_heldout18` |
| `paper` | `[.tex files...]` | `./run.sh paper SBchallenge26_round1.tex` |

`data` with no arguments bakes the 30 shipped scenarios (about 4.6 GB, 5 to 10 minutes) and the oracle traces. A set name also bakes that evaluation list. `verify` takes 35 to 50 minutes and ends with `ALL CHECKS PASSED`. `train` with no arguments runs the paper recipe: 30 scenarios, 960 agents, about 200 million steps. On one L40S that is about 4.6 hours. `eval` and `score` are deterministic: one fresh episode per scenario, 8,640 steps, fp64 physics, `env.desync=0`.

The reported checkpoint is `experiments/19_final_r139_30_scenario_200m` at step `199065600`.

```bash
./run.sh score 19_final_r139_30_scenario_200m 199065600                  # 90 months, the paper's main panel
./run.sh score 19_final_r139_30_scenario_200m 199065600 eval3_heldout18  # 54 months
```

## Repository layout

```
run.sh                  entry point for setup, data, training, evaluation, and the paper
LICENSE                 CC BY 4.0 for the code and archive written here
NOTICE                  attribution for the organiser code and the two benchmark papers
CITATION.cff            machine-readable citation
SHA256SUMS              digests of the released files
STARTER_KIT.md          pinned commit of the organiser simulator

complete_port/          GPU simulator, trainer patch, and tools
  gpu/                  CUDA kernels: physics, reward, observation, action decode
  csim/                 CPU C core the GPU is compared against
  patches/              patch on PufferLib; base commit in patches/pufferlib_base_commit.txt
  pysim/                packs a scenario config and a weather file into a .sbb blob
  tools/                train, evaluate, verify, bake, score
  config/               smart_buildings.ini, the training recipe
  policy/               observation features shared with the submission policy
  submission/           exported Kaggle policy (NumPy only)
  viewer/               replay viewer
  ocean/                symlinks into the PufferLib environment package
  docs/                 port notes: HISTORY.md, PRIORS.md

starter-kit/            organiser simulator, cloned by ./run.sh setup, unmodified, Apache-2.0
kaggle_evaluation/      organiser scoring gateway, unmodified

data/                   what a scenario is, the scenario lists, blob sizes and hashes
  scenarios/sets/       one text file per evaluation panel
experiments/            one folder per run: arguments, logs, checkpoints, scores
SAC/                    off-policy baseline on the same simulator and scenarios
paper/                  LaTeX source, figures, and the compiled PDFs
docs/                   hardware limits, the verification record, the file manifest
build/                  script that assembles this release
```

Scenario blobs are a cache, not source. `./run.sh data` rebuilds them. Their names and sizes are in `data/blobs_manifest.tsv`. Nothing from the hidden leaderboard evaluation is in this repository.

## License

The code, experiment archive, and paper written for this entry are [CC BY 4.0](LICENSE). Reuse them, cite this work, and say what you changed.

`starter-kit/` and `kaggle_evaluation/` are the organisers' work under Apache-2.0 and keep those terms. `starter-kit/` is not vendored; `./run.sh setup` clones it. `NOTICE` lists each part.

## Citation

```bibtex
@inproceedings{sidharthan2026smartbuildings,
  title     = {One Shared Bounded-Beta PPO Policy for Zero-Shot Commercial HVAC Control},
  author    = {Sidharthan S},
  booktitle = {NeurIPS 2026 Smart Buildings Challenge},
  year      = {2026}
}
```

The benchmarks this work trains and evaluates on:

```bibtex
@article{goldfeder2024smart,
  title   = {The Smart Buildings Control Suite: A Diverse Open Source Benchmark to Evaluate and Scale HVAC Control Policies for Sustainability},
  author  = {Goldfeder, Judah and Dean, Victoria and Jiang, Zixin and Wang, Xuezheng and Lipson, Hod and Sipple, John and others},
  journal = {arXiv preprint arXiv:2410.03756},
  year    = {2024}
}

@article{goldfeder2026smart,
  title   = {The Smart Buildings Challenge: Learning Foundation Models for Real-World Optimization at Scale},
  author  = {Goldfeder, Judah and Rai, Ayush and Wyder, Philippe M and Li, Na and Dean, Victoria and Vakalis, Donna and Lin, Jiong and Dong, Bing and Rossetti, Michael and Kutz, Nathan and others},
  journal = {arXiv preprint arXiv:2610.18511},
  year    = {2026}
}

@misc{pufferlib,
  title        = {{PufferLib}: Fast and Sane Simplifying Reinforcement Learning for Complex Game Environments},
  author       = {Joseph Suarez},
  howpublished = {\url{https://github.com/PufferAI/PufferLib}},
  year         = {2024},
  note         = {GitHub repository}
}
```

# One shared PPO policy for heterogeneous commercial building HVAC control

Code, data recipes, checkpoints and results for our entry to the **NeurIPS 2026 Smart Buildings
Challenge**. One PPO policy, trained only on the competition reward, controls every building it is
given: 41 to 81 rooms, one to four air handlers, six climates, no per-building tuning and no weather,
month or building identifier in its input.

| policy | run | step | 90 months | 54 months | public leaderboard |
|---|---|---|---|---|---|
| **Final** | `r139` 30-scenario | 199,065,600 | **0.0165** | **0.0098** | 0.0294 |
| Mid | `r139` 30-scenario | 159,252,480 | 0.0185 | 0.0112 | 0.0175 |
| Long | `r122` 30-scenario | 89,579,520 | 0.0253 | 0.0192 | 0.0182 |
| PID (`pid_water`) | organiser baseline | | 0.0461 | 0.0437 | 0.0640 |

Lower is better. The score is the competition return
`-0.6 * productivity + 0.2 * energy_cost + 0.2 * carbon`, with productivity stored signed, so a
negative number is discomfort. Zero is the ideal no controller reaches. The 90-month and 54-month
panels are held-out weather windows of buildings the policy was trained on; the public row is one
third of the hidden evaluation, which we do not see.

**Final** is the reported policy: it leads both held-out panels. **Mid** is the checkpoint that
leads our public leaderboard entry. Both are shipped, each named with the measure it wins. One seed.

---

## Requirements

| | |
|---|---|
| GPU | one NVIDIA card with 24 GB, compute capability 8.6 or newer. The paper's training run used one L40S (46 GB) |
| driver / runtime | a driver new enough for CUDA 13, plus `nvidia-container-toolkit` |
| container | `pufferai/puffertank:5.0`, pulled by `./run.sh setup`. The host `nvcc` is not used: it is too old for this code |
| Python | 3.12 in `starter-kit/.venv` (the organiser simulator needs `numba`, which has no 3.13 wheels), 3.11 or newer for our tools |
| PufferLib | a checkout of `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2` plus `complete_port/patches/pufferlib_core.diff`, created by `./run.sh setup` |
| disk | 10 GB for the code and the 30 baked scenarios, 30 GB if you also bake the evaluation sets |
| network | the first `setup` clones PufferLib and the starter kit; after that everything is local. Both clones retry three times over HTTP/1.1, and `SB_PL_ROOT` can point at a directory that already holds `plclean` |

CPU-only review is possible for the paper and for the analysis of the shipped scores. Training,
evaluation and the bit-exactness gates need the GPU. The gates were last run end to end from a
clean copy on a 12 GB RTX 3060: 67 passed, 0 failed, 33 minutes (`docs/VERIFICATION.md`).

---

## Six commands

Everything runs from the repository root. Each command prints what it is doing and fails loudly.

```bash
./run.sh setup      # container, Python 3.12 oracle env, PufferLib at the pinned commit + our patch
./run.sh data       # bake the 30 shipped scenarios into artifacts/train (4.6 GB, 5 to 10 min)
./run.sh verify     # build both simulator variants and run the 35 bit-exactness gates (~20 min)
./run.sh train      # PPO, 30 scenarios, 960 agents, 199.9M steps
./run.sh eval       # deterministic fp64 month for every scenario in a set
./run.sh score      # the 90-month panel for one checkpoint, with per-step recordings
```

| command | what it really runs | wall time |
|---|---|---|
| `setup` | `docker pull`, `uv sync` in `starter-kit/`, `git clone` + `deploy_to_pufferlib.sh --apply-patch` | 10 min once |
| `data` | `tools/bake_training_set.sh`, then `tools/make_artifacts.sh` (oracle traces and CPU references) | 5 to 10 min |
| `verify` | `tools/verify_all.sh --trainer`, ends with `ALL CHECKS PASSED` | 18 to 25 min |
| `train` | `PUFFERLIB=... tools/train.sh` with `config/smart_buildings.ini` | see below |
| `eval` | `tools/eval_policy.sh <set> <out.json> <checkpoint>` | 2 min per 18 scenarios |
| `score` | `tools/score_eval90.py` over three parts, merged by `tools/merge_eval_json.py` | 10 min per 90 months |

Training budget: 199,895,040 steps at 17.4K steps/s measured with 960 agents on the exact fp64 build
is 3.2 h; the paper quotes 4.6 h on one L40S. A 24 GB card runs the same batch in 16 to 18 GB.

Evaluation is deterministic: `env.desync=0`, one fresh episode per scenario, 8,640 steps, fp64 solve.
The same checkpoint always produces the same number, which is why the panels in the table can be
re-derived years from now.

---

## What is where

```
run.sh                     the six commands, plus reproduce and paper
starter-kit/               the organiser's simulator, cloned by ./run.sh setup at a pinned commit, unmodified, Apache-2.0
complete_port/             the GPU port, the trainer patch, the tools, the policy
  gpu/                     CUDA kernels: physics, reward, observation, action decode
  csim/                    the CPU core the GPU is diffed against, bit for bit
  patches/                 our PufferLib 5.0 patch; the base commit is in patches/pufferlib_base_commit.txt
  pysim/                   bake a scenario into a .sbb blob, dump oracle traces
  tools/                   train, eval, verify, bake, score, TensorBoard
  config/                  smart_buildings.ini: the recipe, flag by flag
  submission/              the exported Kaggle policies, numpy only
  docs/                    HISTORY.md (what was measured when) and PRIORS.md (every human prior)
kaggle_evaluation/         the scoring gateway, as shipped by the organisers
data/                      what a scenario is, the scenario lists, blob hashes, the data card
experiments/               the ablation library, the reported runs, the score panels, the ledger
paper/                     the paper source, the figures and both compiled PDFs
docs/                      hardware, release edits, licences
MANIFEST.md, SHA256SUMS    every file, its size and its digest
```

---

## Verification: what "bit for bit" means here

`./run.sh verify` is the gate, and it prints one PASS or FAIL line per check. It builds the same
source twice, once with the default fast solve and once with `SB_LU_EXACT=1`, and then compares the
GPU against the unmodified Python simulator and against the CPU C core:

| gate group | what it proves |
|---|---|
| oracle traces | the first 30 to 40 steps of a scenario match the Python simulator exactly: zero differing observation values, reward and done included |
| multi-episode | 300 to 900 steps across several episodes match the CPU C core, with host resets and with in-step restarts |
| device solve | the device linear solve equals `sb_lu_solve` bit for bit at 5 and 40 environments |
| convection | the device convection kernel equals `sb_conv_apply` bit for bit, including the extremes `p=1e-4`, `0.5`, `1.0` |
| multi-scenario | six buildings in one group, solo against interleaved against concurrent streams, bit identical; weathers of one building merged into one group, bit identical |
| CUDA graph | the same diffs hold when the step is captured and replayed, which is how training runs |
| trainer | the mixed-head PPO kernels match a finite-difference reference; the asynchronous checkpoint restores optimizer, RNG, simulator and pending rollout |
| policy | the Beta head, the per-head ratio, the mask, the permutation equivariance of the shared trunk, all against reference implementations |
| viewer | a replay snapshot renders, so a trained policy can be watched |

The one documented difference is the default linear solve, which agrees with the CPU replay to about
`2e-14` rather than exactly. The exact build removes even that, at about 26% lower throughput, and is
used by the gates that demand zero difference.

---

## Data

A scenario is a building plus a weather strip. The 30 shipped scenarios are defined by
`starter-kit/smart_buildings/training_set.yaml`: six buildings, one weather strip per station, no
strip reused.

The paper's panels are wider:

| panel | buildings | weather windows | months |
|---|---|---|---|
| 90 months | `train_1, train_2, train_3, train_6, train_8` | 18 per building | 90 |
| 54 months | `train_4, train_5, train_7` | 18 per building | 54 |
| public | the hidden evaluation | | 30 |

`train_3` and `train_6` are shipped buildings that are not in the training set, and every panel uses
weather windows outside the five each building is trained on. The lists are in
`data/scenarios/sets/`. Nothing from the hidden evaluation is in this repository.

Scenarios are baked to `.sbb` blobs, which are a cache: `bake_scenario.py` packs a `config.gin` plus
the weather into the binary layout the GPU reads. The blobs are 0.4 to 0.9 GB each, so they are not
shipped; `data/blobs_manifest.tsv` records every blob's name and size, `data/train_blobs.sha256`
records the digest of the 30, and `./run.sh data` rebuilds them deterministically.

---

## Experiments

`experiments/` holds 60 run folders, one per run that decided the recipe, named after the change each
one makes:

```
recipe_01_coherent_beta_cn_tau_12/       r84, the sampler that ended the cold basin
recipe_06_linear_lr_anneal/              r94b, the run the paper calls the starting point
recipe_07_headwise_ratio/                r115, per-head ratio and the 50M budget
evidence_01_beta_head_vs_gaussian/       r79, why the head is a Beta
ablation_04_zone_mlp_on_vs_off/          r123, policy.zone_mlp 1 -> 0
ablation_12_muon_role_adam_vs_adam/      r130, adam=1 + role_adam=0
ablation_16_hidden_size_256_vs_128/      r138, hidden_size 256 -> 128
rejected_09_popart/                      r42, 0.926 against 0.171
rejected_26_minibatch_tile_9216/         the tile that drove KL to zero
17_composition_r139_bestof/              the five winning mechanisms combined
18_scale_r122_30_scenario/               30 scenarios, 960 agents
19_final_r139_30_scenario_200m/          the reported policy
```

Each folder carries the run's full argument list, its launch command, the startup banner, the last
epochs, every TensorBoard scalar as `tag,step,value`, and the run's checkpoints. `experiments/README.md`
is the index and `experiments/ABLATIONS.md` is the ledger: what each arm cost, what it bought, and
which claims it kills.

The sixteen ablations are one flag off the `r116` baseline, trained from scratch, one seed, one
batch, one binary, 50,135,040 steps. The seven `recipe_*` folders are the chain that built that
baseline, each adding one flag to the previous. The three `evidence_*` folders are the pre-r116
levers that were adopted with a control. The thirty `rejected_*` folders are the levers that were
measured and not adopted, each with the number that killed it. The panel is the training read at
each save, which is a changing policy; it is a learning-curve comparison, not a score. The scores are
in `experiments/scores/`.

---

## Reproducing a claim

| claim in the paper | command |
|---|---|
| the 90-month panel, Table 2 | `./run.sh score r139 199065600` |
| one row of Table 2, any checkpoint | `./run.sh eval eval5_heldout18 final_90m.json <ckpt>` |
| the 54-month panel | `./run.sh score r139 199065600 --set eval3_heldout18` |
| the training curve of any arm | `experiments/<arm>/tb/scalars.csv.gz` |
| the recipe of any arm | `experiments/<arm>/args.txt` against `experiments/00_baseline_r116_s75_s76/*/args.txt` |
| bit-exactness | `./run.sh verify` |
| the figures | `paper/figures/`, rebuilt by `paper/build.sh` for the PDF and `paper/figures/*.py` for the plots |

`./run.sh reproduce` runs the shortest end-to-end path: data, then the 90-month panel for the shipped
Final checkpoint, then the score.

---

## Honest limits

- **One seed.** Every run in the paper is seed 75, apart from the r116 baseline pair. The seed spread
  we measured is 6%; the gap against the baseline is larger than that, the gap to the best single arm
  is not.
- **Noise remains.** The last five panel readings of the reported run are 0.026 to 0.028, and no
  ablation was repeated at 200M scale.
- **Simulator only.** The policy is evaluated inside the organiser's calibrated simulator, not in a
  real building. Nothing here is a sim-to-real claim.
- **The public row is one third** of the hidden evaluation. A leaderboard position is not the score
  the paper reports.

---

## Licences and citation

Our code is Apache-2.0. `starter-kit/` is the organisers' Apache-2.0 work and stays unmodified.
`kaggle_evaluation/` is theirs as well. The paper is the authors'.

```bibtex
@inproceedings{siddharth2026smartbuildings,
  title  = {One Shared PPO Policy for Heterogeneous Commercial Building HVAC Control},
  author = {Sidharthan S},
  booktitle = {NeurIPS 2026 Smart Buildings Challenge},
  year   = {2026}
}
```

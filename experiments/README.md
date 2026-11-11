# Experiments

Every run that decided the recipe, one folder each, named after the change it makes. The archive is
in five families, in the order the work happened.

## Reading order

| family | folders | what it is |
|---|---|---|
| `00_baseline_r116_s75_s76/` | 1 | the recipe every arm is measured against, on two seeds |
| `recipe_01` to `recipe_07` | 7 | the chain that built r116 from r84, one flag per step, ending at r94b (step 6), the run the paper calls the starting point |
| `evidence_01` to `evidence_03` | 3 | the pre-r116 levers that were adopted with a control: the Beta head, the shape floor, episode desync |
| `ablation_01` to `ablation_16` | 16 | one flag off the baseline, one binary, one seed, 50,135,040 steps |
| `17` to `19` | 3 | the winning composition, the 30-scenario scale-up, and the reported policy |
| `rejected_01` to `rejected_30` | 30 | the graveyard: every lever that was measured and not adopted, with the number that killed it |

`scores/` holds the frozen evaluation. `ABLATIONS.md` is the ledger these folders are indexed against,
and `EXPERIMENTS.md` is the longer development log in time order.

## The naming rule

```
family_<two digits>_<what changed>_<control>_vs_<arm>
```

`ablation_04_zone_mlp_on_vs_off` is the arm with `policy.zone_mlp` at 0 against the baseline where it
is 1. The control value comes first. Numbers use `p` notation: `ablation_13_clip_coef_0p2_vs_0p1`,
`ablation_15_gamma_0p99_vs_0p995`, `rejected_05_proportional_band_water_reset`.

`recipe_NN` steps are cumulative: each one adds a single flag to the previous, so the diff between
two neighbouring `args.txt` files is one line. `rejected_NN` levers are independent, measured against
the recipe of their era.

## What a panel number is

`env/episode_score_scen_mean` at a save is the mean competition return of the episodes that finished
at that point, on a policy that is still changing, with each environment's episode cut at a random
phase. It is a learning-curve read: comparable inside a family, not a score and not comparable to the
frozen numbers in `../README.md`.

Frozen scores come from `scores/`, which holds the deterministic fp64 evaluation of fixed checkpoints
over whole months.

## Checkpoints

| family | what is shipped |
|---|---|
| baseline, recipe, evidence, ablations, compositions, scale, final | every save the run wrote (6 to 16 weights, 1.8 MB each) |
| rejected | the final save only, because the evidence for a rejection is its curve |

`--final-only` on the assembler ships only the last save everywhere, at about 300 MB less.

Weights are the trainer's flat `master_weights` dump. The `.mb` momentum sidecars are not shipped: they
are only needed to continue training, which the release does not do.

## Files in every folder

| file | what |
|---|---|
| `README.md` | the change, what the ledger records, the verdict, and the run facts read back from the artifacts |
| `args.txt` | the complete argument list; diff two of these to see exactly one change |
| `command.txt` | the launch command, with development paths marked `<workspace>` and `<pufferlib>` |
| `train.log.head` | startup banners: which mechanisms were on, in the trainer's own words |
| `train.log.tail` | the last epochs, including the final panel read |
| `tb/scalars.csv.gz` | every epoch scalar as `tag,step,value`; the whole curve, nothing removed |
| `ckpt/` | `master_weights` dumps with manifests, and the sha256 of each |

The full per-visit diagnostics (`SB_UPDATE`, `SB_ZONE_ACTION`, `SB_CREDIT` and the rest, 40 MB to
450 MB per run) are not shipped; `tb/scalars.csv.gz` carries the same series at epoch resolution.

## Reproducing one arm

```bash
diff 00_baseline_r116_s75_s76/r116_zone_mlp_headwise_s75/args.txt \
     ablation_04_zone_mlp_on_vs_off/args.txt
```

shows one changed token. `../complete_port/config/smart_buildings.ini` holds the defaults and
`../run.sh train` trains with them.

## Commands that read this archive

| command | arguments | what it does |
|---|---|---|
| `../run.sh score <run> <step> [set]` | folder name, checkpoint step, one of `eval5_heldout18` or `eval3_heldout18` | the frozen panel of that checkpoint, with recordings |
| `../run.sh eval <set> <out.json> <ckpt> [run]` | the run folder supplies the decode flags, without which the loader refuses the weights | one month per scenario, no recordings |
| `../complete_port/tools/decode_flags.py <run>` | any folder here | prints the flags that decide how a checkpoint is decoded, from `args.txt` or `flags.txt` |
| `../complete_port/tools/tbread.py <run>/tb <tag>` | a tag substring, e.g. `env/episode_score` | prints a series from the raw event files; `tb/scalars.csv.gz` already holds every epoch of it |

A checkpoint folder holds `args.txt` (the exact launch), `command.txt` (the launch as it ran), the
head and tail of the log, the epoch scalars and the weights. `diff` two `args.txt` files to see the one
flag that changed.

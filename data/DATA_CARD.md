# Data card

What a scenario is, where the inputs come from, and how to rebuild every blob.

## The scenario

A scenario is one building plus one weather strip, baked into a `.sbb` blob. The blob holds the
floor plan, the materials, the terminal units, the air handlers, the plant, the occupancy and the
weather at 5-minute resolution, packed into the binary layout the GPU kernel reads. A blob is a
cache, not a source of truth: `pysim/bake_scenario.py` produces it from a `config.gin` and the
weather files, both of which ship with the simulator.

A month is 8,640 steps of 300 s, starting at 06:00 on the first Sunday of the window.

## Inputs, all from the organiser

| input | where |
|---|---|
| six building definitions (rooms, walls, materials, AHUs, VAV boxes) | `starter-kit/smart_buildings/buildings/*.yaml` |
| forty weather strips per station, six stations | `starter-kit/smart_buildings/weather/` |
| the 30 shipped scenario pairs | `starter-kit/smart_buildings/training_set.yaml` |
| per-scenario simulator configs | `starter-kit/smart_buildings/training_scenarios/<building>/<weather>/config.gin` |
| furniture, setpoints, schedules | the same configs and `smart_control/` |

Nothing in this repository replaces those inputs. `data/scenarios/training_set.yaml` is a copy of
the organiser's list, kept here so the release can be read without cloning the simulator.

## The panels in the paper

| panel | buildings | weather windows per building | months |
|---|---|---|---|
| 90 months | `building_train_1`, `_2`, `_3`, `_6`, `_8` | 18 | 90 |
| 54 months | `building_train_4`, `_5`, `_7` | 18 | 54 |
| public | the hidden evaluation, which we do not have | | 30 |

The 30 training scenarios give each building five weather strips, one per station, so the 18
evaluation strips per building are outside the five it was trained on. `building_train_3` and
`building_train_6` appear in no training scenario. The exact lists are in `data/scenarios/sets/`.

## Generated buildings

Three buildings outside the shipped six were generated to test that the port is not tuned to one
floor plan:

| building | rooms | purpose |
|---|---|---|
| `building_0500` | 120 AHU-coupled zones, 278 continuous actions | the wide-action path, `SB_CONT_BIG` |
| `building_0501` | new floor plan, new weather | a second unseen layout |
| `building_0502` | new floor plan, new weather | a third |

Their `config.gin` files are in `data/generated_buildings/`. They are produced by
`complete_port/tools/new_scenarios.sh --seeds ... --weathers ...` from the same generator the
organiser ships, and the verification gate `generated buildings 0502/0511` bakes them and diffs them
against the Python simulator.

No generated building is used for any score in the paper. They exist for the port gates only.

## Rebuilding the blobs

```bash
./run.sh data                                    # 30 shipped scenarios, ~5 min, 4.6 GB
complete_port/pysim/bake_scenario.py --out <dir> --name <building>__<weather> \
    <path to config.gin>                         # any other scenario
```

`data/blobs_manifest.tsv` records the name and size of every blob that existed when the release was
assembled. `data/train_blobs.sha256` records the digest of the 30 training blobs, so a rebuild can be
checked byte for byte:

```bash
cd complete_port/artifacts/train && sha256sum -c ../../../data/train_blobs.sha256
```

## What is not here

- No blob is shipped: the 265 blobs in the development workspace total 103 GB, and every one of them
  is regenerable from inputs that are in the simulator.
- No per-step recording is shipped: one checkpoint's 90-month recording is 7.6 GB. The scores it
  produced are in `experiments/scores/` and the recorder that produces it is
  `tools/eval_policy.sh` with `SB_RECORD` set.
- No hidden-evaluation scenario, building or weather strip is present, and none was used.

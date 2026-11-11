# data

What defines a scenario, and what this repository ships instead of a 103 GB blob store.

| path | what |
|---|---|
| `DATA_CARD.md` | the full card: what a scenario is, which buildings and weather strips the panels use, how to rebuild a blob, what is deliberately absent |
| `scenarios/training_set.yaml` | the organiser's 30 shipped scenario pairs |
| `scenarios/sets/*.txt` | one scenario per line, `<building>__<weather>.sbb`: the 90-month panel (`eval5_heldout18`) and the 54-month panel (`eval3_heldout18`) |
| `generated_buildings/*.yaml` | the buildings the panels use that the starter kit does not ship: `building_train_3` (floor-generator seed 1018) and `building_train_6` (1012), plus the generated buildings the port gates use |
| `blobs_manifest.tsv` | every blob that existed when the release was assembled: name and bytes |
| `train_blobs.sha256` | the digest of the 30 training blobs, so a rebuild can be checked byte for byte |
| `blob_meta/*.json` | per-scenario metadata: room names, grid size, interior mass, the config it was baked from |

## Commands that use it

| command | what it bakes |
|---|---|
| `./run.sh data` | the 30 training scenarios, the oracle traces and the CPU references |
| `./run.sh data eval5_heldout18 [jobs]` | the same, plus the 90-month set (90 blobs, 32.6 GB) |
| `./run.sh data eval3_heldout18 [jobs]` | the same, plus the 54-month set (54 blobs, 20 GB) |
| `complete_port/tools/bake_eval_set.sh data/scenarios/sets/<name>.txt [jobs]` | one set on its own; `jobs` is how many scenarios bake in parallel (default 2) |

A set needs a `config.gin` for every pair, and the starter kit ships configs only for the 30 training
scenarios, so `bake_eval_set.sh` first installs any generated building from `generated_buildings/` and
then runs the organiser's scenario generator for the missing weather strips. A rebuild of
`building_train_3__atlanta_4.sbb` is byte-identical to the development blob, so the panels are exactly
reproducible.

Check a rebuild:

```bash
cd complete_port/artifacts/train && sha256sum -c ../../../data/train_blobs.sha256
```

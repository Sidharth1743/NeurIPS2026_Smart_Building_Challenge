# Paper

The source of the submitted paper and its PDF.

| file | what |
|---|---|
| `SBchallenge26_round1.tex` | the paper: body (pp. 1–8), references (9–10), appendices A–J (11–27) |
| `SBchallenge26_round1.pdf` | the compiled paper, 27 pages |
| `references.bib` | 32 entries, numeric, citation order, the two benchmark papers first |
| `neurips_2026.sty` | the style file |
| `figures/` | every figure as PDF, plus the scripts that produced them |
| `build.sh` | rebuilds the PDF with `tectonic` |

```bash
./build.sh                                   # rebuild SBchallenge26_round1.pdf
./build.sh SBchallenge26_round1.tex          # name the source explicitly
tectonic -X compile SBchallenge26_round1.tex # or directly
../run.sh paper                              # the same from the repository root
```

## The figures

`figures/*.py` are the plotting scripts and they read the per-step recordings, which are 7.6 GB per
checkpoint and are not shipped. What they read is in `../experiments/scores/`; the recordings
themselves come from `../complete_port/tools/eval_policy.sh` with `SB_RECORD` set. The compiled
figures in this folder are the ones the paper uses, so the paper builds without them.

`results_fig_data.py` reads `SB_ROOT` to point at a workspace that holds the recordings, and defaults
to the directory above `complete_port/`. From a fresh clone the recordings are absent, so the scripts
need `SB_ROOT` set to a workspace that has them.

The figure scripts take no arguments: each reads its data, writes its PDF beside itself, and is run as
`python make_results_figs_v2.py` from this folder. `results_fig_data.py` is the shared data layer and
reads `SB_ROOT` if the recordings live outside the checkout.

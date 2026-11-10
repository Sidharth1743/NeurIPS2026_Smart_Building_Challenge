# Paper

The source of the submitted paper and both PDFs.

| file | what |
|---|---|
| `neurips_3_method.tex` | the paper, 8 content pages and 2 pages of references |
| `neurips_3_method.pdf` | the compiled paper |
| `neurips_3_method_appendix.tex` | the same main text, then appendices A to I |
| `neurips_3_method_appendix.pdf` | the compiled appendix, 27 pages, Appendix A on page 11 |
| `references.bib` | 31 entries, numeric, citation order, the two benchmark papers first |
| `neurips_2026.sty` | the style file |
| `figures/` | every figure as PDF, plus the scripts that produced them |
| `build.sh` | rebuilds both PDFs with `tectonic` |

```bash
./build.sh                       # both PDFs, in place
tectonic -X compile neurips_3_method.tex
```

## The figures

`figures/*.py` are the plotting scripts and they read the per-step recordings, which are 7.6 GB per
checkpoint and are not shipped. What they read is in `../experiments/scores/`; the recordings
themselves come from `../complete_port/tools/eval_policy.sh` with `SB_RECORD` set. The compiled
figures in this folder are the ones the paper uses, so the paper builds without them.

`results_fig_data.py` takes `SB_CP` to point at a workspace that holds the recordings, and defaults to
the directory above `complete_port/`.

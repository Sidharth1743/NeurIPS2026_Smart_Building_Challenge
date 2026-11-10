"""Appendix figures: F1 (the 90 scored months), G1 (coverage grid), G2 (climate plane)."""
import os
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
SB_ROOT = Path(os.environ.get("SB_ROOT", next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents]
     if (p / 'complete_port').is_dir()), Path(__file__).resolve().parent)))


OUT = Path(__file__).resolve().parent / "figures"
REPO = SB_ROOT
KIT = REPO / "starter-kit/smart_buildings"
BLUE, RED, GREY, TEAL = "#0F4D92", "#B64342", "#8A8A8A", "#2F7F8A"
SEEN, UNSEEN = "#DCE8F4", "#F2D9B8"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.6, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

# ------------------------------------------------------------------ F1: the 90 months
d = json.load(open(REPO / "complete_port/results/study_r139_eval90/gpu/r139/"
                   "0000000199065600/score.json"))
vals = sorted(-v["mean_reward"] for v in d["scenarios"].values())
x = np.arange(1, len(vals) + 1)
v = np.array(vals)
mean, med = v.mean(), np.median(v)

fig = plt.figure(figsize=(5.5, 1.95))
ax = fig.add_axes([0.085, 0.215, 0.895, 0.700])
ax.axvspan(85.5, 90.5, color="#F2D9B8", lw=0, zorder=0)
ax.plot(x, v, color=BLUE, lw=0.9, zorder=3)
ax.plot(x, v, marker="o", ms=1.5, mfc=BLUE, mec="none", ls="", zorder=4)
ax.axhline(mean, color=RED, lw=0.8, ls=(0, (4, 2)), zorder=2)
ax.axhline(med, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.set_xlim(0, 91)
ax.set_ylim(0, 0.092)
ax.set_xlabel("scenario, sorted by score", labelpad=1.6)
ax.set_ylabel("score of one month", labelpad=1.6)
ax.set_xticks([1, 30, 60, 90])
ax.text(3, mean + 0.0024, "mean 0.0165", color=RED, fontsize=6.1, va="bottom")
ax.text(3, med - 0.0045, "median 0.0094", color=GREY, fontsize=6.1, va="top")
ax.text(6, 0.079, "the five worst months are one building\nand carry a fifth of the total",
        fontsize=5.9, color="#8A5A1E", ha="left", va="top")
fig.savefig(OUT / "fig_appendix_scores.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_scores.png", dpi=500, facecolor="white")

# --------------------------------------------------------------- G1: coverage grid
TRAIN_B = ["train_1", "train_2", "train_4", "train_5", "train_7", "train_8"]
TRAIN_S = ["atlanta", "denver", "fairbanks", "phoenix", "singapore"]
EVAL_B = ["train_1", "train_2", "train_3", "train_6", "train_8"]
EVAL_S = ["atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore"]
EVAL_COUNT = {"atlanta": 2, "denver": 2, "fairbanks": 2, "minneapolis": 8,
              "phoenix": 2, "singapore": 2}

fig2 = plt.figure(figsize=(5.5, 2.30))
axa = fig2.add_axes([0.108, 0.315, 0.315, 0.585])
axb = fig2.add_axes([0.545, 0.315, 0.395, 0.585])


def grid(ax, rows, cols, counts, unseen_rows=(), unseen_cols=(), title=""):
    nr, nc = len(rows), len(cols)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            n = counts(c)
            seen = (r not in unseen_rows) and (c not in unseen_cols)
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1,
                                   facecolor=SEEN if seen else UNSEEN,
                                   edgecolor="white", linewidth=1.2))
            ax.text(j + 0.5, nr - 1 - i + 0.5, str(n), ha="center", va="center",
                    fontsize=6.4, color="#1A1A1A")
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks(np.arange(nc) + 0.5)
    ax.set_xticklabels(cols, fontsize=6.1, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(nr)[::-1] + 0.5)
    ax.set_yticklabels(rows, fontsize=6.0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=6.6, pad=3)


grid(axa, TRAIN_B, TRAIN_S, lambda c: 1,
     title="(a) training: 6 buildings \u00d7 5 stations = 30")
grid(axb, EVAL_B, EVAL_S, lambda c: EVAL_COUNT[c],
     unseen_rows=("train_3", "train_6"), unseen_cols=("minneapolis",),
     title="(b) evaluation: 5 buildings \u00d7 18 windows = 90")

handles = [Patch(facecolor=SEEN, edgecolor="white", label="building and station both trained"),
           Patch(facecolor=UNSEEN, edgecolor="white", label="building or station never trained")]
fig2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=2, frameon=False, fontsize=6.1, handlelength=1.1, handleheight=0.8,
            columnspacing=1.6)
fig2.savefig(OUT / "fig_appendix_coverage.pdf", facecolor="white")
fig2.savefig(OUT / "fig_appendix_coverage.png", dpi=500, facecolor="white")

# ------------------------------------------------------------------ G2: climate
import yaml

cache = {}


def station_rows(st):
    if st in cache:
        return cache[st]
    rows = []
    with open(KIT / f"weather/csv/{st}.csv") as f:
        for row in csv.DictReader(f):
            rows.append((datetime.strptime(row["Time"], "%Y%m%d-%H%M"),
                         float(row["TempC"])))
    cache[st] = rows
    return rows


def hdd_cdd(st, start, hours=720):
    rows = station_rows(st)
    idx = {dt: i for i, (dt, t) in enumerate(rows)}
    i = idx[start]
    win = rows[i:i + hours]
    hdd = sum(max(0.0, 18.0 - t) for _, t in win) / 24.0
    cdd = sum(max(0.0, t - 21.0) for _, t in win) / 24.0
    return hdd, cdd


specs = {}
for p in sorted((KIT / "weather").glob("*.yaml")):
    s = yaml.safe_load(open(p))
    specs[s["name"]] = (p.stem.split("_")[0], s["start_timestamp"])

train_ids = []
for line in open(KIT / "training_set.yaml"):
    if line.strip().startswith("weather:"):
        train_ids.append(line.split(":", 1)[1].strip())
eval_ids = [r["weather"] for r in csv.DictReader(open(REPO / "results/eval90_analysis/data/windows.csv"))]

pts = []
for wid in train_ids:
    st, ts = specs[wid]
    h, c = hdd_cdd(st, datetime.strptime(ts, "%Y-%m-%d %H:%M"))
    pts.append((h, c, st, "train"))
for wid in eval_ids:
    st, ts = specs[wid]
    h, c = hdd_cdd(st, datetime.strptime(ts, "%Y-%m-%d %H:%M"))
    pts.append((h, c, st, "eval"))

COL = {"atlanta": "#4E79A7", "denver": "#59A14F", "fairbanks": "#8C6D31",
       "minneapolis": "#B07AA1", "phoenix": "#E8A33D", "singapore": "#D37295"}

fig3 = plt.figure(figsize=(5.5, 2.18))
axh = fig3.add_axes([0.175, 0.255, 0.320, 0.670])
axc = fig3.add_axes([0.610, 0.255, 0.320, 0.670])

ORDER = ["minneapolis", "fairbanks", "denver", "atlanta", "phoenix", "singapore"]
rng = np.random.default_rng(7)
by = {}
for h, c, st, kind in pts:
    by.setdefault(st, []).append((h, c, kind))

for k, st in enumerate(ORDER):
    row = len(ORDER) - 1 - k
    tr = [p for p in by[st] if p[2] == "train"]
    ev = [p for p in by[st] if p[2] == "eval"]
    if tr:
        hs = [p[0] for p in tr]
        cs = [p[1] for p in tr]
        axh.plot([min(hs), max(hs)], [row, row], color="#CFCFCF", lw=2.4,
                 solid_capstyle="round", zorder=1)
        axc.plot([min(cs), max(cs)], [row, row], color="#CFCFCF", lw=2.4,
                 solid_capstyle="round", zorder=1)
    for h, c, _ in tr:
        dy = rng.uniform(-0.15, 0.15)
        axh.plot(h, row + dy, "o", ms=3.1, mfc=COL[st], mec="white", mew=0.4, zorder=3)
        axc.plot(c, row + dy, "o", ms=3.1, mfc=COL[st], mec="white", mew=0.4, zorder=3)
    for h, c, _ in ev:
        dy = rng.uniform(-0.24, 0.24)
        axh.plot(h, row + dy, "o", ms=3.7, mfc="white", mec=COL[st], mew=0.95, zorder=4)
        axc.plot(c, row + dy, "o", ms=3.7, mfc="white", mec=COL[st], mew=0.95, zorder=4)

axh.set_xlim(-20, 1250)
axh.set_ylim(-0.65, 5.80)
axc.set_xlim(-8, 505)
axc.set_ylim(-0.65, 5.80)

axh.set_yticks(range(len(ORDER)))
axh.set_yticklabels(ORDER[::-1], fontsize=6.4)
axc.set_yticks([])
axh.tick_params(axis="y", length=0)

axh.set_xlabel("heating degree-days, base 18 $^{\\circ}$C", labelpad=1.6)
axc.set_xlabel("cooling degree-days, base 21 $^{\\circ}$C", labelpad=1.6)
axh.set_title("(a) heating demand", fontsize=6.6, pad=3)
axc.set_title("(b) cooling demand", fontsize=6.6, pad=3)

axh.text(410, 5.58, "station never trained", fontsize=5.8, ha="center", va="bottom",
         color=COL["minneapolis"])

handles3 = [plt.Line2D([], [], marker="o", ls="", ms=3.6, mfc=GREY, mec=GREY,
                       label="a trained month (30)"),
            plt.Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=GREY,
                       label="a scored month (18 windows), grey bar = trained range")]
fig3.legend(handles=handles3, loc="lower center", bbox_to_anchor=(0.5, -0.01),
            ncol=2, frameon=False, fontsize=6.0, handletextpad=0.3, columnspacing=1.6)
fig3.savefig(OUT / "fig_appendix_climate.pdf", facecolor="white")
fig3.savefig(OUT / "fig_appendix_climate.png", dpi=500, facecolor="white")
print("wrote scores, coverage, climate")

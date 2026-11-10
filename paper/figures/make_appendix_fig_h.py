"""Appendix Figure H1: per-head spread by role, and where the zone deficit sits."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
DATA = Path(os.environ.get("SB_ROOT", Path(__file__).resolve().parents[1])) / "results" / "eval90_analysis" / (
            "r139_199065600/09_behaviour.json")
BLUE, RED, GREY = "#0F4D92", "#B64342", "#9A9A9A"
COLS = ["#0F4D92", "#2F7F8A", "#7FA6CC", "#8A8A8A", "#C97B5A"]

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.6, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

d = json.load(open(DATA))
ROLES = [r["role"] for r in d["roles"]]
heads = d["heads"]
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(5.5, 2.35))
axa = fig.add_axes([0.205, 0.215, 0.335, 0.660])
axb = fig.add_axes([0.640, 0.215, 0.335, 0.660])

# ------------------------------------------------------------------ (a) heads
for k, role in enumerate(ROLES):
    v = [h["sd"] for h in heads if h["role"] == k]
    y = np.full(len(v), len(ROLES) - 1 - k) + rng.uniform(-0.16, 0.16, len(v))
    axa.plot(v, y, "o", ms=2.6, mfc=BLUE, mec="white", mew=0.35, alpha=0.85, zorder=3)
axa.axvline(0.05, color=RED, lw=0.7, ls=(0, (3, 2)), zorder=2)
axa.set_xscale("log")
axa.set_xlim(0.006, 1.1)
axa.set_ylim(-0.6, len(ROLES) - 0.4)
axa.set_yticks(range(len(ROLES)))
axa.set_yticklabels(ROLES[::-1], fontsize=6.1)
axa.tick_params(axis="y", length=0)
axa.set_xticks([0.01, 0.05, 0.2, 0.5])
axa.set_xticklabels(["0.01", "0.05", "0.2", "0.5"])
axa.set_xlabel("spread of one head over a month", labelpad=1.6)
axa.text(0.054, -0.48, "static", fontsize=5.8, color=RED, va="bottom")
axa.set_title("(a) every head moves", fontsize=6.6, pad=3)

# ------------------------------------------------------------------ (b) zones
NAMES = {"building_train_1": "train_1", "building_train_2": "train_2",
         "building_train_3": "train_3", "building_train_6": "train_6",
         "building_train_8": "train_8"}
for i, b in enumerate(d["buildings"]):
    pz = np.sort(np.array(b["pz"]))[::-1]          # worst first
    cum = np.cumsum(pz) / pz.sum()
    x = np.arange(1, len(pz) + 1) / len(pz)
    xx = np.concatenate([[0], x])
    yy = np.concatenate([[0], cum])
    col = RED if b["building"] == "building_train_3" else COLS[i]
    lw = 1.2 if b["building"] == "building_train_3" else 0.9
    axb.plot(xx, yy, color=col, lw=lw, zorder=3 if b["building"] == "building_train_3" else 2)
    axb.plot([], [], color=col, lw=1.0, label=NAMES[b["building"]])
axb.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.7, ls=(0, (2, 2)), zorder=1)
axb.set_xlim(0, 1)
axb.set_ylim(0, 1)
axb.set_xticks([0, 0.25, 0.5, 0.75, 1])
axb.set_xticklabels(["0", "25", "50", "75", "100"])
axb.set_yticks([0, 0.25, 0.5, 0.75, 1])
axb.set_yticklabels(["0", "25", "50", "75", "100"])
axb.set_xlabel("zones, worst first (%)", labelpad=1.6)
axb.set_ylabel("share of the deficit (%)", labelpad=1.6)
axb.set_title("(b) where the deficit sits", fontsize=6.6, pad=3)
axb.legend(loc="lower right", frameon=False, fontsize=5.9, ncol=2,
           handlelength=1.3, columnspacing=1.0, handletextpad=0.4,
           borderaxespad=0.2)

fig.savefig(OUT / "fig_appendix_behaviour.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_behaviour.png", dpi=500, facecolor="white")
print("wrote fig_appendix_behaviour")

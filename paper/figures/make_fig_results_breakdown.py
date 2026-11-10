"""Figure 2: score by building and by climate, ours against pid_water.

Dumbbell per row: the controller as a red square, the policy as a blue dot, the line
between them the gain. Log axis labelled in plain decimals, rows sorted by our score,
the two unseen buildings shaded and starred. House style per the scientific-figure-making
skill: two hues, marker shape as the second channel, spines off, vector output.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter

OUT = Path(__file__).resolve().parent / "figures"
BLUE, RED, GREY = "#0F4D92", "#B64342", "#9A9A9A"
BAND = "#EDF3FA"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})

BUILDINGS = [("train_2", 0.0078, 0.0379, -79, False),
             ("train_8", 0.0085, 0.0392, -78, False),
             ("train_1", 0.0094, 0.0354, -74, False),
             ("train_6*", 0.0127, 0.0491, -74, True),
             ("train_3*", 0.0443, 0.0689, -36, True)]
CLIMATES = [("phoenix", 0.0112, 0.0278, -60, False),
            ("atlanta", 0.0130, 0.0316, -59, False),
            ("singapore", 0.0158, 0.0487, -68, False),
            ("minneapolis*", 0.0180, 0.0520, -65, True),
            ("denver", 0.0181, 0.0490, -63, False),
            ("fairbanks", 0.0184, 0.0500, -63, False)]

XLIM = (0.0030, 0.22)
TICKS = [0.01, 0.02, 0.05]
LABELS = ["0.01", "0.02", "0.05"]

fig = plt.figure(figsize=(5.5, 2.5))
panels = [(fig.add_axes([0.150, 0.255, 0.315, 0.570]), BUILDINGS, "by building"),
          (fig.add_axes([0.610, 0.255, 0.315, 0.570]), CLIMATES, "by climate")]

for ax, rows, title in panels:
    n = len(rows)
    ys = np.arange(n)[::-1]
    for (name, ours, pid, pct, unseen), y in zip(rows, ys):
        if unseen:
            ax.axhspan(y - 0.46, y + 0.46, color=BAND, zorder=0, lw=0)
        ax.plot([pid, ours], [y, y], color=GREY, lw=1.1, solid_capstyle="round",
                zorder=2)
        ax.plot(pid, y, marker="s", ms=4.0, mfc=RED, mec="white", mew=0.5, ls="",
                zorder=4)
        ax.plot(ours, y, marker="o", ms=4.4, mfc=BLUE, mec="white", mew=0.5, ls="",
                zorder=5)
        ax.annotate(f"{ours:.4f}", (ours, y), textcoords="offset points",
                    xytext=(-4.5, 0), ha="right", va="center", fontsize=5.7,
                    color=BLUE)
        ax.annotate(f"{pid:.4f}", (pid, y), textcoords="offset points",
                    xytext=(4.5, 0), ha="left", va="center", fontsize=5.7,
                    color=RED)

    ax.set_yscale("linear")
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(LABELS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="x", labelsize=6.2, length=2.5, pad=2, width=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.grid(axis="x", color="#EAEAEA", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.045, title, transform=ax.transAxes, fontsize=7.0,
            ha="left", va="bottom", color="#1A1A1A")
    ax.set_xlabel("score", fontsize=6.8, labelpad=2)
    for s in ("left",):
        ax.spines[s].set_visible(False)

handles = [Line2D([], [], marker="o", ms=4.4, mfc=BLUE, mec="white", mew=0.5, ls="",
                  label="ours"),
           Line2D([], [], marker="s", ms=4.0, mfc=RED, mec="white", mew=0.5, ls="",
                  label="pid_water"),
           Patch(facecolor=BAND, edgecolor="none", label="unseen in training")]
fig.text(0.048, 0.972, "left is better", fontsize=7.6, fontweight="bold",
         ha="left", va="center", color="#1A1A1A")
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.63, 0.99), ncol=3,
           frameon=False, fontsize=6.6, handletextpad=0.4, columnspacing=1.5)


fig.text(0.52, 0.045, "building and weather characteristics, and how the structures differ, are listed in the appendix",
         ha="center", va="center", fontsize=5.7, color="#5A5A5A")
fig.savefig(OUT / "fig_results_breakdown.pdf", facecolor="white")
fig.savefig(OUT / "fig_results_breakdown.png", dpi=500, facecolor="white")
print("wrote fig_results_breakdown.pdf/.png")

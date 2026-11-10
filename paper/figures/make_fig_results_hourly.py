"""Figure 3: score per step by hour of day, ours against pid_water, 90 held-out months.

The curve carries the argument of 4.3: cheaper in every occupied hour, dearer only in
the two hours before occupancy. Log axis, occupied window shaded, two hues plus marker
shape, vector output (scientific-figure-making house style).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "figures"
BLUE, RED, GREY = "#0F4D92", "#B64342", "#8A8A8A"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})

o = pd.read_csv("/tmp/opencode/hour_ours.csv", index_col=0).iloc[:, 0]
q = pd.read_csv("/tmp/opencode/hour_pid.csv", index_col=0).iloc[:, 0]
hours = list(range(24))

fig = plt.figure(figsize=(3.5, 2.35))
ax = fig.add_axes([0.175, 0.185, 0.800, 0.760])

ax.axvspan(6, 19, color="#EDF3FA", zorder=0, lw=0)
ax.text(12.5, 0.00092, "occupied", fontsize=5.8, color="#5A6B80", ha="center", va="bottom")

ax.plot(hours, q.values, color=RED, lw=1.2, marker="s", ms=2.9, mfc="white",
        mew=0.8, zorder=3, label="pid_water")
ax.plot(hours, o.values, color=BLUE, lw=1.3, marker="o", ms=2.9, mfc=BLUE,
        mec="white", mew=0.4, zorder=4, label="ours")

ax.set_yscale("log")
ax.set_ylim(4.5e-4, 0.26)
ax.set_xlim(-0.6, 23.6)
ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
ax.tick_params(labelsize=6.2, length=2.5, pad=2, width=0.7)
ax.set_yticks([0.001, 0.01, 0.1])
ax.set_yticklabels(["0.001", "0.01", "0.1"])
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("hour of day", fontsize=6.8, labelpad=2)
ax.set_ylabel("score per step", fontsize=6.8, labelpad=2)


handles = [Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=3.0, label="ours"),
           Line2D([], [], color=RED, lw=1.2, marker="s", ms=3.0, mfc="white",
                  mew=0.8, label="pid_water")]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.3,
          handlelength=1.4, labelspacing=0.3, borderpad=0.1)

fig.savefig(OUT / "fig_results_hourly.pdf", facecolor="white")
fig.savefig(OUT / "fig_results_hourly.png", dpi=500, facecolor="white")
print("wrote fig_results_hourly.pdf/.png")

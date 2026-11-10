"""Experimental: one figure, six panels, hour of day.

score / comfort / energy / carbon for both policies, gas under the policy, and the
occupancy that drives all of it. Shared clock, shared shading, one hue per policy.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "figures"
BLUE, RED, GREY = "#0F4D92", "#B64342", "#8A8A8A"
BAND = "#EDF3FA"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})

d = pd.read_csv("/tmp/opencode/hour_profile.csv")
h = d.hour.to_numpy()

PANELS = [
    ("score (log)", "o_score", "p_score", True, "log"),
    ("comfort", "o_comfort", "p_comfort", True, "linear"),
    ("energy", "o_energy", "p_energy", True, "linear"),
    ("carbon", "o_carbon", "p_carbon", True, "linear"),
    ("gas (kWh)", "o_gas", None, False, "linear"),
]

fig = plt.figure(figsize=(3.55, 4.75))
axes = []
H = 0.118
for i, (name, oc, pc, both, scale) in enumerate(PANELS):
    bottom = 0.855 - i * (H + 0.028)
    ax = fig.add_axes([0.185, bottom, 0.775, H])
    ax.axvspan(6, 19, color=BAND, zorder=0, lw=0)
    ax.plot(h, d[oc], color=BLUE, lw=1.1, marker="o", ms=2.1, mec="white", mew=0.3,
            zorder=4)
    if both and pc:
        ax.plot(h, d[pc], color=RED, lw=1.0, marker="s", ms=1.9, mfc="white",
                mew=0.6, zorder=3)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_ylabel(name, fontsize=6.4, labelpad=2)
    ax.tick_params(labelsize=5.9, length=2, pad=1.5, width=0.6)
    if scale == "log":
        ax.set_yscale("log")
        ax.set_yticks([0.001, 0.01, 0.1])
        ax.set_yticklabels(["0.001", "0.01", "0.1"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticklabels([])
    axes.append(ax)

# occupancy, bottom panel, drawn as context
ax = fig.add_axes([0.185, 0.855 - 5 * (H + 0.028), 0.775, H])
ax.axvspan(6, 19, color=BAND, zorder=0, lw=0)
ax.fill_between(h, 0, d.occ, color="#C9D6E4", lw=0, zorder=2)
ax.plot(h, d.occ, color="#5A6B80", lw=0.8, zorder=3)
ax.set_xlim(-0.5, 23.5)
ax.set_ylim(0, 56)
ax.set_xticks([0, 6, 12, 18])
ax.set_ylabel("occupants", fontsize=6.4, labelpad=2)
ax.tick_params(labelsize=5.9, length=2, pad=1.5, width=0.6)
ax.set_xlabel("hour of day", fontsize=6.6, labelpad=2)

handles = [Line2D([], [], color=BLUE, lw=1.1, marker="o", ms=2.4, label="ours"),
           Line2D([], [], color=RED, lw=1.0, marker="s", ms=2.2, mfc="white", mew=0.6,
                  label="pid_water")]
axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
               frameon=False, fontsize=6.0, handlelength=1.4, columnspacing=1.4,
               borderpad=0.0)

fig.savefig(OUT / "fig_results_profile.pdf", facecolor="white")
fig.savefig(OUT / "fig_results_profile.png", dpi=400, facecolor="white")
print("wrote fig_results_profile.pdf/.png")

"""Appendix Figure D1: the shape of one control head.

(a) the Beta family on the action interval, (b) the sensitivity of the action to the
latent draw for a tanh-squashed Gaussian against the affine Beta map, (c) the spread a
head can reach as a function of alpha+beta, with the concentration cap at 12.
"""
from math import gamma, log
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
BLUE, TEAL, RED, GREY = "#0F4D92", "#2F7F8A", "#B64342", "#9A9A9A"
LIGHT = "#EAF1F8"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.6, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})


def beta_pdf_a(a, al, be):
    """Beta(al, be) on x in (0,1), mapped to the action a = 2x - 1; density in a."""
    x = np.clip((a + 1.0) / 2.0, 1e-9, 1 - 1e-9)
    logb = log(gamma(al)) + log(gamma(be)) - log(gamma(al + be))
    p = np.exp((al - 1) * np.log(x) + (be - 1) * np.log1p(-x) - logb)
    return 0.5 * p


fig = plt.figure(figsize=(5.5, 1.72))

# ------------------------------------------------------------------ (a) the family
ax = fig.add_axes([0.062, 0.245, 0.275, 0.615])
a = np.linspace(-1, 1, 1400)
for (al, be), col, lw in (((0.5, 0.5), BLUE, 1.0), ((2, 2), TEAL, 1.0)):
    y = beta_pdf_a(a, al, be)
    ax.plot(a, np.minimum(y, 2.9), color=col, lw=lw, zorder=3)
ax.set_xlim(-1.02, 1.02)
ax.set_ylim(0, 2.9)
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(["$-$1", "0", "1"])
ax.set_yticks([0, 1, 2])
ax.set_xlabel("action", labelpad=1.5)
ax.set_ylabel("density", labelpad=1.5)
ax.set_title("(a) the family", fontsize=6.8, pad=2.5)

ax.text(-0.94, 1.30, "$\\alpha,\\beta=0.5$", color=BLUE, fontsize=6.0, va="bottom")
ax.text(-0.94, 1.06, "holds a stop", color=BLUE, fontsize=5.8, va="bottom")
ax.text(0.44, 1.62, "$2,2$", color=TEAL, fontsize=6.0, va="bottom")
ax.text(0.44, 1.40, "a hump", color=TEAL, fontsize=5.8, va="bottom")

# ------------------------------------------------------------- (b) the derivative
axb = fig.add_axes([0.408, 0.245, 0.225, 0.615])
aa = np.linspace(-0.999, 0.999, 900)
axb.plot(aa, 1 - aa ** 2, color=RED, lw=1.1, zorder=3)
axb.plot([-1, 1], [2, 2], color=BLUE, lw=1.1, zorder=3)
axb.set_xlim(-1.02, 1.02)
axb.set_ylim(0, 2.35)
axb.set_xticks([-1, 0, 1])
axb.set_xticklabels(["$-$1", "0", "1"])
axb.set_yticks([0, 1, 2])
axb.set_xlabel("action", labelpad=1.5)
axb.set_ylabel("$\\partial$ action / $\\partial$ draw", labelpad=1.5)
axb.set_title("(b) the map", fontsize=6.8, pad=2.5)

axb.annotate("0 at the stop", xy=(0.965, 0.070), xytext=(0.30, 0.42),
             fontsize=5.9, color=RED,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=RED))
axb.text(-0.94, 1.74, "2 everywhere", fontsize=5.9, color=BLUE, va="bottom")
axb.text(-0.96, 0.30, "Gaussian,\ntanh", fontsize=5.9, color=RED, va="bottom")

# ------------------------------------------------------------------- (c) the cap
axc = fig.add_axes([0.728, 0.245, 0.248, 0.615])
c = np.linspace(2, 40, 500)
axc.axvspan(12, 40, color=LIGHT, lw=0, zorder=0)
axc.plot(c, 1 / np.sqrt(c + 1), color=BLUE, lw=1.1, zorder=3)
axc.plot([12], [1 / np.sqrt(13)], marker="o", ms=3.4, color=BLUE, zorder=4)
axc.axvline(12, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=2)
axc.set_xlim(2, 40)
axc.set_ylim(0, 0.62)
axc.set_xticks([2, 12, 20, 30, 40])
axc.set_yticks([0, 0.2, 0.4, 0.6])
axc.set_xlabel("$\\alpha+\\beta$", labelpad=1.5)
axc.set_ylabel("spread of one head", labelpad=1.5)
axc.set_title("(c) the cap", fontsize=6.8, pad=2.5)
axc.text(13.2, 0.325, "cap at 12\nspread 0.28", fontsize=5.9, color=BLUE, va="bottom")
axc.text(26.0, 0.085, "not reachable", fontsize=5.8, color=GREY, ha="center")


fig.savefig(OUT / "fig_appendix_beta.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_beta.png", dpi=500, facecolor="white")
print("wrote fig_appendix_beta")

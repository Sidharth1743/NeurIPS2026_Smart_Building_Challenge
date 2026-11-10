"""Appendix Figures E1 (the 50M change set) and E2 (the flagship training panel).

E1: every row is a setting as shipped. The dot is the score when that one setting is
flipped and everything else is left alone. The dashed line is the same recipe with nothing
flipped. Two settings score off the axis and are drawn as arrows at the break.
E2: the training panel of the reported run on a logarithmic ordinate, so 0.01 and 0.02 get
the same room as 0.1 and 0.2.
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
SB_ROOT = Path(os.environ.get("SB_ROOT", next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents]
     if (p / 'complete_port').is_dir()), Path(__file__).resolve().parent)))


OUT = Path(__file__).resolve().parent / "figures"
RUNS = SB_ROOT / "complete_port/results/runs"
BLUE, TEAL, RED, GREY = "#0F4D92", "#2F7F8A", "#B64342", "#8A8A8A"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.4, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

REF = 0.0330          # the same recipe with nothing flipped
LO, HI = 0.0310, 0.0350   # 6 % seed spread around the reference

# (setting as shipped, score when that one setting is flipped, delta %, saves of 6)
ROWS = [
    ("without staggered starts", 2.213, 6606, 0),
    ("without the per-head clip", 0.152, 361, 1),
    ("without the cap", 0.054, 64, 0),
    ("without the zone MLP", 0.052, 58, 1),
    ("trust region 0.1", 0.051, 55, 1),
    ("role split off (Muon)", 0.049, 48, 0),
    ("signed deficits", 0.044, 33, 0),
    ("without the anneal", 0.038, 15, 1),
    ("$\\gamma=0.995$", 0.038, 15, 3),
    ("grad clip 1.0", 0.034, 3, 3),
    ("width 128", 0.031, -6, 4),
    ("Beta floor 1.0", 0.031, -6, 5),
    ("without coherent draws", 0.031, -6, 6),
    ("without the critic delay", 0.030, -9, 6),
    ("Adam everywhere", 0.029, -12, 6),
    ("all adopted changes", 0.0280, -15.2, 5),
]
n = len(ROWS)

fig = plt.figure(figsize=(5.5, 2.96))
ax = fig.add_axes([0.268, 0.128, 0.452, 0.790])

ax.axvline(REF, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(REF + 0.0024, n - 0.42, "unmodified", fontsize=6.1, color=GREY, va="center")

for i, (name, panel, delta, saves) in enumerate(ROWS):
    y = n - 1 - i
    if panel > HI:
        col = RED
    elif panel < LO:
        col = TEAL
    else:
        col = GREY
    if panel > 0.06:
        ax.plot(0.0588, y, marker=">", ms=4.4, mfc=RED, mec=RED, ls="", zorder=5)
        ax.text(0.0554, y, f"{panel:.3g}", fontsize=5.9, color=RED, ha="right",
                va="center")
    else:
        ax.plot(panel, y, marker="o", ms=3.8, mfc=col, mec="white", mew=0.45, ls="",
                zorder=4)

    fy = 0.128 + 0.790 * (y + 0.7) / (n + 0.4)
    txt = f"{delta:+.0f} %".replace("-", "$-$")
    fig.text(0.826, fy, txt, fontsize=6.4, ha="right", color=col, va="center")
    fig.text(0.848, fy, f"{saves}/6", fontsize=6.4, ha="left", color=col, va="center")

ax.set_yticks(list(range(n - 1, -1, -1)))
ax.set_yticklabels([r[0] for r in ROWS], fontsize=6.6)
ax.tick_params(axis="y", length=0)

ax.set_xlim(0, 0.0605)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xticks([0.01 * k for k in range(6)])
ax.set_xticklabels(["0", "0.01", "0.02", "0.03", "0.04", "0.05"])
ax.set_xlabel("training panel, lower is better", fontsize=6.6, labelpad=1.6)

# broken axis: two settings score past 0.06
for x0 in (0.0566, 0.0576):
    ax.plot([x0, x0 + 0.0007], [-0.012, 0.012], color="#333333", lw=0.7,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)


fig.text(0.268, 0.950, "score of that one change", fontsize=6.6, ha="left")
fig.text(0.826, 0.950, "vs unmodified", fontsize=6.6, ha="right")
fig.text(0.848, 0.950, "saves", fontsize=6.6, ha="left")

fig.savefig(OUT / "fig_appendix_ablation.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_ablation.png", dpi=500, facecolor="white")

# --------------------------------------------------------------- E2 the curve
steps_re = re.compile(r"Steps\s+([0-9.]+)([KM]?)")
panel_re = re.compile(r"episode_score_scen_mean\s+(-?[0-9.]+)")

rows, step = [], None
for line in (RUNS / "r139_train30_200m_s75/train.log").open(errors="ignore"):
    m = steps_re.search(line)
    if m:
        step = float(m.group(1)) * {"K": 1e3, "M": 1e6}.get(m.group(2), 1)
    m = panel_re.search(line)
    if m and step is not None:
        rows.append((step, float(m.group(1))))
rows = [(s, v) for s, v in rows if s > 8e6 and v > 0]
x = np.array([r[0] for r in rows]) / 1e6
y = np.array([r[1] for r in rows])

fig2 = plt.figure(figsize=(5.5, 1.90))
ax2 = fig2.add_axes([0.098, 0.225, 0.875, 0.685])
ax2.plot(x, y, color=BLUE, lw=0.85, zorder=3)

for s in (39.813, 79.626, 119.439, 159.252, 199.066, 199.895):
    i = int(np.argmin(np.abs(x - s)))
    ax2.plot(x[i], y[i], marker="o", ms=3.0, mfc="white", mec=BLUE, mew=0.8, zorder=5)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(8, 250)
ax2.set_ylim(0.0075, 0.34)
ax2.set_xticks([10, 20, 50, 100, 200])
ax2.set_xticklabels(["10", "20", "50", "100", "200"])
ax2.minorticks_off()
ax2.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2])
ax2.set_yticklabels(["0.01", "0.02", "0.05", "0.10", "0.20"])
ax2.set_xlabel("training steps (millions)", labelpad=1.6)
ax2.set_ylabel("training panel", labelpad=1.6)

ax2.text(210, 0.0112, "0.011", fontsize=6.1, color=BLUE, va="center")

fig2.savefig(OUT / "fig_appendix_curve.pdf", facecolor="white")
fig2.savefig(OUT / "fig_appendix_curve.png", dpi=500, facecolor="white")
print("wrote fig_appendix_ablation and fig_appendix_curve")

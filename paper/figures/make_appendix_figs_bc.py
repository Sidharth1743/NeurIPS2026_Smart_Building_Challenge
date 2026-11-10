"""Appendix figures B1 (983-value slot map) and C1 (memory ring)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
BLUE, TEAL, GREY, RED = "#0F4D92", "#2F7F8A", "#5A5A5A", "#B64342"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def cell(ax, x, y, s, fc, ec=BLUE, lw=0.35):
    ax.add_patch(Rectangle((x, y), s, s, facecolor=fc, edgecolor=ec, linewidth=lw))


# ------------------------------------------------------------------ B1 slot map
fig = plt.figure(figsize=(5.5, 1.45))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 1.45)
ax.set_axis_off()

S, P = 0.047, 0.006
gx, gy = 0.25, 0.47
for r in range(9):
    for c in range(10):
        cell(ax, gx + c * (S + P), gy + r * (S + P), S, "#EAF1F8")

xa = gx + 10 * (S + P) + 0.09
for r in range(9):
    cell(ax, xa, gy + r * (S + P), S, "#CBDCEF")

xp = xa + S + 0.09
for i in range(2):
    cell(ax, xp, gy + (7 - i) * (S + P), S, "#9FBBD8")



ax.add_patch(FancyArrowPatch((1.03, gy + 0.21), (1.55, gy + 0.21),
                             arrowstyle="-|>", mutation_scale=7, lw=0.9, color=GREY))
ax.text(1.60, gy + 0.21, "983", fontsize=11, fontweight="bold", ha="left", va="center")
ax.text(2.02, gy + 0.21, "values", fontsize=6.8, color="#3A3A3A", ha="left", va="center")

LINES = [("#EAF1F8", "90 zone slots", "× 9", "810"),
         ("#CBDCEF", "9 air-handler slots", "× 16", "144"),
         ("#9FBBD8", "1 plant slot", "× 10", "10"),
         ("#9FBBD8", "1 global slot", "× 19", "19")]
for i, (col, name, mult, tot) in enumerate(LINES):
    y = 0.93 - i * 0.20
    ax.add_patch(Rectangle((2.55, y - 0.045), 0.09, 0.09, facecolor=col,
                           edgecolor=BLUE, linewidth=0.4))
    ax.text(2.72, y, name, fontsize=6.6, ha="left", va="center")
    ax.text(3.60, y, mult, fontsize=6.6, ha="left", va="center")
    ax.text(4.30, y, tot, fontsize=6.6, ha="right", va="center", fontweight="bold")
ax.plot([2.55, 4.30], [0.13, 0.13], color="#BBBBBB", lw=0.6)
ax.text(2.72, 0.05, "one value per feature; the first of each slot is presence",
        fontsize=5.8, color=GREY, ha="left", va="center")

fig.savefig(OUT / "fig_appendix_slots.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_slots.png", dpi=500, facecolor="white")

# ------------------------------------------------------------------ C1 ring
fig = plt.figure(figsize=(5.5, 1.00))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 1.00)
ax.set_axis_off()

N, W, H, y = 48, 0.0665, 0.150, 0.52
rx = 0.30
for i in range(N):
    fc = "#FFFFFF" if i < N - 12 else "#CBDCEF"
    ax.add_patch(Rectangle((rx + i * W, y), W - 0.003, H, facecolor=fc,
                           edgecolor=BLUE, linewidth=0.3))
nx = rx + N * W

# whole ring, above
ax.plot([rx, rx, nx, nx], [y + H + 0.02, y + H + 0.10, y + H + 0.10, y + H + 0.02],
        color="#B9B9B9", lw=0.8)
ax.text(rx, y + H + 0.115, "48 steps = 4 hours", fontsize=6.2, color=GREY,
        ha="left", va="bottom")

# last hour, below
b0 = rx + (N - 12) * W
ax.plot([b0, b0, nx, nx], [y - 0.02, y - 0.075, y - 0.075, y - 0.02],
        color=GREY, lw=0.8)
ax.text((b0 + nx) / 2, y - 0.090, "last hour (12 steps)", ha="center", va="top",
        fontsize=6.0)
ax.text(rx, y - 0.175, "oldest step", fontsize=5.9, color=GREY, ha="left", va="top")
ax.text(nx, y - 0.245, "current step", fontsize=5.9, color=GREY, ha="right", va="top")

# projection
px = nx + 0.16
ax.add_patch(FancyArrowPatch((px, y + H / 2), (px + 1.30, y + H / 2),
                             arrowstyle="-|>", mutation_scale=8, lw=1.1, color=TEAL))
ax.text(px, y + H + 0.115, "projected to the billing horizon", fontsize=6.3,
        color=TEAL, ha="left", va="bottom")
ax.text(px, y + H + 0.030, "60 min in comfort, up to 240 min before", fontsize=6.0,
        color="#3A3A3A", ha="left", va="bottom")

fig.savefig(OUT / "fig_appendix_memory.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_memory.png", dpi=500, facecolor="white")
print("wrote fig_appendix_slots and fig_appendix_memory")

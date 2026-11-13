"""Appendix Figure B1: the 983-value vector as an address map plus one expanded slot."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "figures"
BLUE, GREY = "#0F4D92", "#5A5A5A"
SEG = ["#EAF1F8", "#CBDCEF", "#9FBBD8", "#7FA6CC"]

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
})

fig = plt.figure(figsize=(5.5, 1.80))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.5)
ax.set_ylim(0.15, 1.95)
ax.set_axis_off()

# ---------------------------------------------------------------- address map
x0, x1 = 0.60, 5.10
W = x1 - x0
BOUND = [(0, 810, "0–809", "zone slots", "stride 9 · 90 slots"),
         (810, 954, "810–953", "air handlers", "stride 16 · 9 slots"),
         (954, 964, "954–963", "plant", "stride 10"),
         (964, 983, "964–982", "global", "stride 19")]
ybar, H = 1.50, 0.16
for i, (a, b, rng, name, note) in enumerate(BOUND):
    xa = x0 + W * a / 983
    xb = x0 + W * b / 983
    ax.add_patch(Rectangle((xa, ybar), xb - xa, H, facecolor=SEG[i],
                           edgecolor=BLUE, linewidth=0.5))
    ax.plot([xa, xa], [ybar - 0.03, ybar + H + 0.03], color=GREY, lw=0.5)
ax.plot([x1, x1], [ybar - 0.03, ybar + H + 0.03], color=GREY, lw=0.5)
for a, lab in ((0, "0"), (810, "810"), (983, "983")):
    ax.text(x0 + W * a / 983, ybar + H + 0.05, lab, ha="center", va="bottom",
            fontsize=5.9, color=GREY)

# legend under the bar, two columns
for i, (a, b, rng, name, note) in enumerate(BOUND):
    lx = 0.60 if i % 2 == 0 else 3.05
    ly = 1.16 - (i // 2) * 0.20
    ax.add_patch(Rectangle((lx, ly - 0.045), 0.10, 0.10, facecolor=SEG[i],
                           edgecolor=BLUE, linewidth=0.4))
    ax.text(lx + 0.17, ly, rng, fontsize=6.0, va="center", color=GREY)
    ax.text(lx + 0.72, ly, name, fontsize=6.3, va="center")
    ax.text(lx + 1.55, ly, note, fontsize=6.0, va="center", color="#3A3A3A")

# ------------------------------------------------------------- one zone slot
C, GX, GY = 0.245, 0.72, 0.40
for i in range(9):
    ax.add_patch(Rectangle((GX + i * C, GY), C - 0.014, 0.22, facecolor="#EAF1F8",
                           edgecolor=BLUE, linewidth=0.4))
    ax.text(GX + i * C + (C - 0.014) / 2, GY - 0.03, str(i), ha="center", va="top",
            fontsize=6.0, color="#1A1A1A")

ax.plot([GX, GX, GX + 9 * C - 0.014, GX + 9 * C - 0.014],
        [GY + 0.26, GY + 0.31, GY + 0.31, GY + 0.26], color=GREY, lw=0.7)
ax.text(GX + (9 * C) / 2, GY + 0.33, "one zone slot = 9 values, × 90 slots = 810",
        ha="center", va="bottom", fontsize=6.2)

fig.savefig(OUT / "fig_appendix_slots.pdf", facecolor="white")
fig.savefig(OUT / "fig_appendix_slots.png", dpi=500, facecolor="white")
print("wrote fig_appendix_slots")

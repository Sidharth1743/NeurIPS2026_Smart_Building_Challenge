"""Figure 1: heterogeneous buildings -> one canonical tensor -> one shared policy.

Final layout:
  * three region headers on one line: buildings / one canonical layout / one readout per role;
  * each building carries its zone count, with the word "zones" under the row;
  * the temperature scale that colours every zone is drawn once, under the buildings;
  * the conversion is a tapered block arrow labelled "pad to 90";
  * the network is drawn as a network (two layers, recurrent self-loop);
  * dark ink labels, teal only on the readouts.

Drawn at the printed width so the 6 pt labels stay 6 pt.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

INK = "#111111"
GREY = "#3F3F3F"
PAD = "#EBEBEB"
TEAL = "#2F7F8A"
LINK = "#C4C4C4"
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

CMAP = plt.get_cmap("coolwarm")
VMIN, VMAX = 291.5, 298.5


def temp(rows, cols, seed, base=294.4, spread=2.6):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    core = np.exp(-(((yy - (rows - 1) / 2) / max(rows / 2, 1)) ** 2
                    + ((xx - (cols - 1) / 2) / max(cols / 2, 1)) ** 2))
    return base + spread * (core - 0.42) + rng.normal(0, 0.30, (rows, cols))


def rooms(ax, x, y, mask, values, cell, lw=0.4):
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            fc = CMAP((values[r, c] - VMIN) / (VMAX - VMIN)) if mask[r, c] else PAD
            ax.add_patch(Rectangle(
                (x + c * cell, y + (rows - 1 - r) * cell), cell, cell,
                facecolor=fc, edgecolor="white", linewidth=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, lw=0.9, color=INK, head=6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))


def taper(ax, x0, x1, cy, half, waist, color="#CFCFCF"):
    head = (x1 - x0) * 0.42
    pts = [(x0, cy - waist), (x1 - head, cy - waist), (x1 - head, cy - half),
           (x1, cy), (x1 - head, cy + half), (x1 - head, cy + waist),
           (x0, cy + waist)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=1))


fig = plt.figure(figsize=(5.2, 1.00))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.2)
ax.set_ylim(0, 1.00)
ax.set_axis_off()

CY, CELL, FC = 0.620, 0.038, 0.045
BASE = CY - 9 * FC / 2
NROW = 0.365          # zone counts, "pad to 90", "983 values", network label

# ------------------------------------------------------------------- buildings
for counts, x, seed in (([9] * 9, 0.05, 13),
                        ([8] * 7, 0.43, 12),
                        ([8, 8, 8, 8, 9], 0.77, 11)):
    rows, cols = len(counts), max(counts)
    mask = np.zeros((rows, cols), bool)
    for r, n in enumerate(counts):
        mask[r, :n] = True
    rooms(ax, x, BASE, mask, temp(rows, cols, seed), CELL)
    ax.add_patch(Rectangle((x, BASE), cols * CELL, rows * CELL, facecolor="none",
                           edgecolor=INK, linewidth=0.6, zorder=3))
    ax.text(x + cols * CELL / 2, NROW, str(sum(counts)), fontsize=6.0, color=INK,
            ha="center", va="center")

ax.text(0.05, 0.298, "zone counts", fontsize=5.5, color=GREY, ha="left", va="center")

ax.text(0.05, 0.895, "buildings", fontsize=5.7, color=GREY, ha="left", va="center")


# ------------------------------------------------- the temperature colour scale
CB_X, CB_W, CB_Y, CB_H = 4.30, 0.82, 0.140, 0.028
for i in range(160):
    ax.add_patch(Rectangle((CB_X + CB_W * i / 160, CB_Y), CB_W / 160 + 0.002, CB_H,
                           facecolor=CMAP(i / 159), edgecolor="none", zorder=2))
ax.add_patch(Rectangle((CB_X, CB_Y), CB_W, CB_H, facecolor="none",
                       edgecolor="#B0B0B0", linewidth=0.4, zorder=3))
for frac, lab in ((0.0, "292"), (0.5, "295"), (1.0, "298")):
    ax.text(CB_X + CB_W * frac, CB_Y - 0.020, lab, fontsize=5.2, color=GREY,
            ha="center", va="top")
ax.text(CB_X, CB_Y + CB_H + 0.018, "zone temperature (K)", fontsize=5.4,
        color=GREY, ha="left", va="bottom")

# ------------------------------------------------------------- the conversion
taper(ax, 1.150, 1.560, CY, 0.115, 0.058)
ax.text(1.355, 0.300, "pad to 90", fontsize=5.4, color=INK, ha="center", va="center")

# ------------------------------------------------- the one frame, padded to 90
FX = 1.585
rows, cols = 9, 10
mask = np.zeros((rows, cols), bool)
counts = [8, 8, 8, 8, 9]
for r, n in enumerate(counts):
    mask[r, :n] = True
vals = np.full((rows, cols), VMIN)
vals[:5, :9] = temp(5, 9, 11)
rooms(ax, FX, BASE, mask, vals, FC)
ax.add_patch(Rectangle((FX, BASE), cols * FC, rows * FC, facecolor="none",
                       edgecolor=INK, linewidth=0.7, zorder=3))

ax.text(FX, 0.895, "one canonical layout", fontsize=6.9, color=INK,
        fontweight="bold", ha="left", va="center")
ax.text(FX, NROW, "983 values", fontsize=5.7, color=INK, ha="left", va="center")

arrow(ax, FX + cols * FC + 0.040, CY, 2.255, CY)

# ------------------------------------------------------------------ MinGRU x2
G1, G2 = 2.31, 2.79
ys = np.linspace(CY - 0.112, CY + 0.112, 7)
for y0 in ys:
    for y1 in ys:
        ax.plot([G1, G2], [y0, y1], color=LINK, lw=0.22, zorder=1)
for x in (G1, G2):
    for y in ys:
        ax.add_patch(Circle((x, y), 0.0115, facecolor="white", edgecolor=INK,
                            linewidth=0.55, zorder=2))
ax.add_patch(FancyArrowPatch(
    (G2 + 0.006, ys[-1] + 0.020), (G2 - 0.006, ys[-1] + 0.020),
    arrowstyle="-|>", mutation_scale=4.2, linewidth=0.55, color=INK,
    connectionstyle="arc3,rad=1.9", shrinkA=0, shrinkB=0, zorder=3))
ax.text((G1 + G2) / 2, NROW, "2 × MinGRU, 256", fontsize=5.7, color=INK,
        ha="center", va="center")

# --------------------------------------------------------------------- readouts
RX, RW, BH, GAP = 3.03, 0.90, 0.072, 0.040
top = CY + (3 * BH + 2 * GAP) / 2
for i, (name, n) in enumerate((("zone", 90), ("AHU", 9), ("plant", 1))):
    y = top - (i + 1) * BH - i * GAP
    ax.add_patch(Rectangle((RX, y), RW, BH, facecolor="white", edgecolor=TEAL,
                           linewidth=0.9, zorder=2))
    ax.text(RX + 0.075, y + BH / 2, name, fontsize=6.1, color=INK,
            ha="left", va="center")
    ax.text(RX + RW - 0.075, y + BH / 2, str(n), fontsize=6.1, color=TEAL,
            ha="right", va="center", fontweight="bold")
    arrow(ax, G2 + 0.075, CY, RX - 0.02, y + BH / 2, lw=0.7, color=GREY, head=4.5)

ax.text(RX, 0.895, "one readout per role", fontsize=5.7, color=GREY, ha="left",
        va="center")

# ------------------------------------------------------------------------ heads
HX = 4.22
arrow(ax, RX + RW + 0.045, CY, HX - 0.045, CY)
ax.text(HX, CY + 0.050, "219 heads", fontsize=6.6, color=INK, ha="left",
        va="center", fontweight="bold")
ax.text(HX, CY - 0.046, "180 + 36 + 3", fontsize=5.9, color=INK, ha="left",
        va="center")

OUT.mkdir(parents=True, exist_ok=True)
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(OUT / f"fig_architecture.{ext}", bbox_inches="tight",
                pad_inches=0.02, facecolor="white", **kw)
print("wrote", OUT / "fig_architecture.pdf")

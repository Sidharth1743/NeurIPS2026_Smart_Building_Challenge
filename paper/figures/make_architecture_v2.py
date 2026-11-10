"""Figure 1 (corrected): buildings -> canonical layout -> backbone -> readouts -> heads.

Fixes over the previous draft, all checked against the scenario blobs:
  * zone counts are 79 / 54 / 39, not 81 / 56 / 41 - room_names carries two
    non-zone entries (exterior_space, interior_wall) per building, so the
    n_phys_zones field is the right count;
  * the 54-zone building has 6 AHUs, so it fills the 9x10 frame as six rows of
    nine: 54 real + 36 padded, not a row-major block;
  * one fixed colour range (291.5-298.5 K) for every grid, so the colourbar is
    valid for all panels and the 54-zone building shows the same colours in (a)
    and (b);
  * the arrow into the backbone carries the 80-value pooled summary, not the 983 values;
  * the readouts read the second layer's output, not the first;
  * a second, dashed path shows that each readout also reads its own slot's values;
  * padding is labelled "zero", which is what the implementation writes.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 7})

OUT = Path(__file__).resolve().parent / "figures"
W, H = 740, 290
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

TEAL, DK, GR = "#2B7F8A", "#222222", "#777777"
CMAP = plt.get_cmap("coolwarm")
VMIN, VMAX = 291.5, 298.5
rng = np.random.default_rng(3)


def field(rows, cols, seed, base=294.6, spread=2.3):
    """Plausible zone temperatures: warm core, cool perimeter, smooth noise."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    core = np.exp(-(((yy - (rows - 1) / 2) / max(rows / 2, 1)) ** 2
                    + ((xx - (cols - 1) / 2) / max(cols / 2, 1)) ** 2))
    f = base + spread * (core - 0.45) + 0.9 * gaussian_filter(r.normal(size=(rows, cols)), 0.8)
    return np.clip(f, VMIN + 0.02, VMAX - 0.02)


def draw(x0, y0, vals, cell, pad_mask=None):
    """vals: 2D array of temperatures. pad_mask: True where the slot is padding."""
    rows, cols = vals.shape
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * cell, y0 + (rows - 1 - r) * cell
            if pad_mask is not None and pad_mask[r, c]:
                ax.add_patch(Rectangle((x, y), cell, cell, fc="#EDEDED", ec="#BDBDBD",
                                       lw=0.4, hatch="////"))
            else:
                v = (vals[r, c] - VMIN) / (VMAX - VMIN)
                ax.add_patch(Rectangle((x, y), cell, cell, fc=CMAP(v), ec="white", lw=0.5))
    ax.add_patch(Rectangle((x0, y0), cols * cell, rows * cell, fc="none", ec=DK, lw=1))


def arrow(p, q, lw=1.3, c=DK, **k):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=c, shrinkA=0, shrinkB=0, **k))


def head(x, letter, t, sub):
    ax.text(x, 262, f"({letter}) {t}", fontweight="bold", fontsize=7.8, ha="left", color=DK)
    ax.text(x, 251, sub, fontsize=6.4, ha="left", color=GR)


# ------------------------------------------------------------------ (a) buildings
head(8, "a", "Buildings", "variable zone count")

# zones per AHU, from the blobs: train_5 is 8 AHUs, train_4 is 6, train_1 is 4
buildings = [
    (79, [10, 10, 10, 10, 10, 10, 10, 9], 8, 11),
    (54, [9, 9, 9, 9, 9, 9], 77, 12),
    (39, [10, 10, 10, 9], 140, 13),
]
CELL_A = 6.0
vals54 = None
for n, counts, x0, seed in buildings:
    rows, cols = len(counts), max(counts)
    v = field(rows, cols, seed)
    pad = np.array([[c >= n_r for c in range(cols)] for n_r in counts])
    draw(x0, 120, v, CELL_A, pad)
    if n == 54:
        vals54, pad54 = v, pad
    ax.text(x0 + cols * CELL_A / 2, 108, f"{n}", ha="center", fontsize=9, fontweight="bold")
    ax.text(x0 + cols * CELL_A / 2, 97, "zones", ha="center", fontsize=6.5, color=GR)

# (a) -> (b): tapered arrow, the conversion
ax.add_patch(FancyArrowPatch((212, 150), (240, 150),
                             arrowstyle="simple,head_width=14,head_length=10,tail_width=6",
                             fc="#CFCFCF", ec="none"))
ax.text(226, 168, "pad to\n90 slots", ha="center", va="bottom", fontsize=6.5,
        color=DK, linespacing=1.15)

# ------------------------------------------------------- (b) canonical 9x10 frame
head(245, "b", "Canonical layout", "fixed 90 slots")

vals = np.full((9, 10), VMIN)
pad = np.ones((9, 10), bool)
vals[:6, :9] = vals54
pad[:6, :9] = pad54
draw(245, 110, vals, 9.0, pad)

ax.text(290, 97, "54 real + 36 padded", ha="center", fontsize=6.5, color=DK)
ax.text(290, 87, "(example: 54-zone building)", ha="center", fontsize=6, color=GR)

ax.add_patch(Rectangle((245, 66), 8, 8, fc=CMAP((294.5 - VMIN) / (VMAX - VMIN)),
                       ec="white", lw=.5))
ax.text(257, 70, "real zone", va="center", fontsize=6.2)
ax.add_patch(Rectangle((305, 66), 8, 8, fc="#EDEDED", ec="#BDBDBD", lw=.4, hatch="////"))
ax.text(317, 70, "padding (zero)", va="center", fontsize=6.2)

cb = fig.add_axes([30 / W, 38 / H, 104 / W, 5 / H])
cb.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CMAP,
          extent=[VMIN, VMAX, 0, 1])
cb.set_yticks([])
cb.set_xticks([VMIN, (VMIN + VMAX) / 2, VMAX])
cb.set_xticklabels(["291.5", "295", "298.5"], fontsize=6)
cb.tick_params(length=2, pad=1)
for s in cb.spines.values():
    s.set_visible(False)
ax.text(30, 52, "zone temperature (K)", fontsize=6.4, color=GR)

# ------------------------------------------------------------------- (b) -> (c)
arrow((337, 128), (384, 128))
ax.text(356, 133, "pooled", ha="center", va="bottom", fontsize=6.5, color=DK,
        linespacing=1.1)

# ------------------------------------------------------------------- (c) backbone
head(384, "c", "Backbone", "recurrent over time")
ax.add_patch(FancyBboxPatch((376, 105), 82, 90, boxstyle="round,pad=0,rounding_size=4",
                            fc="#F6F6F6", ec="#999999", lw=0.8, ls="--"))
layers = [(113, "MinGRU 1"), (158, "MinGRU 2")]
for y, t in layers:
    ax.add_patch(FancyBboxPatch((386, y), 52, 30, boxstyle="round,pad=0,rounding_size=3",
                                fc="white", ec=DK, lw=1))
    ax.text(412, y + 15, t, ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((438, y + 22), (438, y + 8),
                                 connectionstyle="arc3,rad=-2.2", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=GR))
arrow((412, 143), (412, 158))          # layer 1 -> layer 2
ax.text(417, 93, "hidden size 256", ha="center", fontsize=6.5, color=DK)
ax.text(417, 83, "(h\u209c carried across steps)", ha="center", fontsize=6, color=GR)

# --------------------------------------------------------- (d) readouts, (e) heads
head(490, "d", "Readouts", "one per role")
head(588, "e", "Output heads", "135 in the building shown")

ys = [195, 150, 105]
roles = [("Zone", "90 slots", "90 × 2", "180"),
         ("AHU", "9 units", "9 × 4", "36"),
         ("Plant", "1 unit", "1 × 3", "3")]
for y, (nm, cnt, calc, tot) in zip(ys, roles):
    arrow((458, 173), (488, y), lw=1.1, c="#444444")   # out of the second layer
    ax.add_patch(FancyBboxPatch((490, y - 17), 84, 34, boxstyle="round,pad=0,rounding_size=3",
                                fc="white", ec=TEAL, lw=1.6))
    ax.text(532, y + 5, f"{nm} readout", ha="center", va="center", fontsize=7,
            fontweight="bold", color=DK)
    ax.text(532, y - 7, cnt, ha="center", va="center", fontsize=6.5, color=TEAL)
    arrow((576, y), (590, y), lw=1.1, c="#444444")
    ax.text(594, y + 5, f"{tot} heads", va="center", fontsize=7.5, fontweight="bold", color=DK)
    ax.text(594, y - 7, calc, va="center", fontsize=6.5, color=GR)

# (b) -> (d): each readout also reads its own slot's values, past the trunk
ax.plot([335, 335, 532], [191, 219, 219], color=TEAL, lw=1.0, ls=(0, (2.4, 1.6)))
arrow((532, 219), (532, 213), lw=1.0, c=TEAL, linestyle=(0, (2.4, 1.6)))

ax.plot([660, 666, 666, 660], [ys[0] + 12, ys[0] + 12, ys[2] - 12, ys[2] - 12], color=DK, lw=1)
arrow((666, 150), (682, 150), lw=1.2)
ax.text(686, 158, "219", fontsize=13, fontweight="bold", va="bottom")
ax.text(686, 148, "heads", fontsize=8, va="top")
ax.text(686, 136, "180+36+3", fontsize=6.4, color=GR, va="top")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_architecture_v2.pdf")
fig.savefig(OUT / "fig_architecture_v2.png", dpi=400)
print("wrote", OUT / "fig_architecture_v2.pdf", "and .png")

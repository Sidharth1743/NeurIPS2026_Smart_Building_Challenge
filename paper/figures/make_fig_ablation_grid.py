"""Figure 7: one panel per claim, curves against the unmodified recipe with its seed band.

Panel labels sit under each panel; the figure caption carries the encoding and the final
numbers. Only the four large effects and the six small ones share axes.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(os.environ.get("SB_ROOT", Path(__file__).resolve().parents[1])) / "complete_port" / "results" / "runs"
OUT = Path(__file__).resolve().parent / "figures"
RED, TEAL, GREY, DARK = "#B64342", "#2F7F8A", "#8A8A8A", "#3A3A3A"
BAND = "#ECECEC"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.7, "ytick.major.size": 1.7,
    "xtick.labelsize": 5.4, "ytick.labelsize": 5.4,
})

STEP = re.compile(r"Steps\s+([0-9.]+)([KM]?)")
PANEL = re.compile(r"episode_score_scen_mean\s+(-?[0-9.]+)")


def load(name):
    steps, vals = [], []
    s = None
    for line in (RUNS / name / "train.log").open(errors="ignore"):
        m = STEP.search(line)
        if m:
            s = float(m.group(1)) * {"K": 1e3, "M": 1e6}.get(m.group(2), 1)
        m = PANEL.search(line)
        if m and s:
            v = float(m.group(1))
            if v > 0:
                steps.append(s / 1e6)
                vals.append(v)
    return np.array(steps), np.array(vals)


BASE = ["r116_zone_mlp_headwise_s75", "r116_zone_mlp_headwise_s76",
        "r116_zone_mlp_headwise_s77"]
bx, by = load(BASE[0])
bs = [by] + [np.interp(bx, *load(n)) for n in BASE[1:]]
lo, hi = np.min(bs, axis=0), np.max(bs, axis=0)
BASE_FINAL = float(np.median([s[-1] for s in bs]))

ARMS = [
    ("a", "staggered starts off", "r129_no_desync_s75", 2.213),
    ("b", "per-head clip off", "r125_no_headwise_s75", 0.152),
    ("c", "cap off", "r127_no_conc_cap_s75", 0.054),
    ("d", "zone MLP off", "r123_no_zone_mlp_s75", 0.052),
    ("e", "trust region 0.1", "r135_clip01_s75", 0.051),
    ("f", "role split off, Muon", "r128_no_role_adam_s75", 0.049),
    ("g", "signed deficits", "r134_zone_signed_s75", 0.044),
    ("h", "anneal off", "r131_no_anneal_s75", 0.038),
    ("i", "$\\gamma=0.995$", "r137_gamma995_s75", 0.038),
    ("j", "grad clip 1.0", "r136_maxgrad10_s75", 0.034),
    ("k", "width 128", "r138_hidden128_s75", 0.031),
    ("l", "Beta floor 1.0", "r133_floor05_s75", 0.031),
    ("m", "coherent draws off", "r124_no_coherence_s75", 0.031),
    ("n", "critic delay off", "r132_no_value_delay_s75", 0.030),
    ("o", "Adam everywhere", "r130_adam_s75", 0.029),
    ("p", "all changes", "r139_bestof_s75", 0.028),
]

SAVES = [10.027, 20.054, 30.081, 40.108, 45.121, 50.135]


def colour(final):
    if final > BASE_FINAL * 1.06:
        return RED
    if final < BASE_FINAL * 0.94:
        return TEAL
    return GREY


W, H = 5.5, 4.35
fig = plt.figure(figsize=(W, H))

AXW, AXH = 1.22, 0.66
LEFT, TOP, COL, ROW = 0.42, 0.52, 1.30, 0.87

for k, (letter, name, run, final) in enumerate(ARMS):
    r, c = divmod(k, 4)
    x0 = LEFT + c * COL
    y0 = TOP + r * ROW
    ax = fig.add_axes([x0 / W, 1 - (y0 + AXH) / H, AXW / W, AXH / H])

    x, y = load(run)
    y = np.interp(bx, x, y)
    col = colour(final)
    ax.fill_between(bx, lo, hi, color=BAND, lw=0, zorder=1)
    ax.plot(bx, by, color=DARK, lw=0.8, zorder=2)
    ax.plot(bx, y, color=col, lw=1.0, zorder=3)
    ax.plot(SAVES, np.interp(SAVES, bx, y), "o", ms=1.8, mfc=col, mec="white",
            mew=0.3, zorder=4)

    ax.set_xlim(0, 52)
    ax.set_ylim(0.02, 3.0)
    ax.set_yscale("log")
    ax.set_xticks([10, 30, 50])
    ax.set_yticks([0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    if c:
        ax.set_yticklabels([])
    if r < 3:
        ax.set_xticklabels([])

    dy = -0.30 if r == 3 else -0.17
    ax.text(0.0, dy, f"$\\mathbf{{{letter}}}$  {name}", transform=ax.transAxes,
            fontsize=5.9, va="top", ha="left")

# shared axis labels
fig.text(0.5, 0.028, "training steps (millions)", ha="center", fontsize=6.2)
fig.text(0.028, 0.55, "training panel", va="center", rotation=90, fontsize=6.2)

# legend
h1, = plt.plot([], [], color=DARK, lw=0.9)
h2 = plt.Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none")
h3, = plt.plot([], [], "o", ms=2.0, mfc=DARK, mec="white", mew=0.3, ls="")
fig.legend([h1, h2, h3], ["unmodified recipe", "range of its three seeds", "the six saves"],
           loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
           fontsize=6.0, handlelength=1.4, columnspacing=1.4, handletextpad=0.5)

fig.savefig(OUT / "fig_ablation_A.pdf", facecolor="white")
fig.savefig(OUT / "fig_ablation_A.png", dpi=400, facecolor="white")
print("wrote A")

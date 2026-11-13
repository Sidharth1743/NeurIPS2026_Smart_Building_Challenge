"""Appendix figure: the 144 scored months as comfort / energy / carbon, sorted by score,
with the building and the cold-versus-hot character of each month.

Reads the per-step recordings of the reported checkpoint (study_r139_eval90 for the 90
months, study_eval457 for the 54) and writes figures/fig_appendix_terms.pdf/.png plus a
CSV of the row data beside it. Not referenced by the .tex yet.

    python make_fig_appendix_terms.py [--cp /path/to/complete_port] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

WEATHERS = ["atlanta_4", "atlanta_7", "denver_1", "denver_4", "fairbanks_3", "fairbanks_6",
            "minneapolis_1", "minneapolis_2", "minneapolis_3", "minneapolis_4", "minneapolis_5",
            "minneapolis_6", "minneapolis_7", "minneapolis_8", "phoenix_2", "phoenix_7",
            "singapore_2", "singapore_5"]
SETS = [("90", "complete_port/results/study_r139_eval90/gpu/r139/0000000199065600",
         ["building_train_1", "building_train_2", "building_train_3", "building_train_6", "building_train_8"]),
        ("54", "complete_port/results/study_eval457/r139/0000000199065600",
         ["building_train_4", "building_train_5", "building_train_7"])]

BAND_LO, BAND_HI = 293.15, 297.15      # the benchmark's comfort window, 20-24 C
STEPS = 8640

BUILDING_ORDER = [f"building_train_{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8)]
BUILDING_COL = dict(zip(BUILDING_ORDER, ["#1F6FB2", "#E07B39", "#2CA02C", "#D62728",
                                         "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]))
TERM_COL = {"comfort": "#0F4D92", "energy": "#5B9BD5", "carbon": "#A6CEE3"}


def month_rows(cp: Path):
    rows = []
    for tag, rel, buildings in SETS:
        for b in buildings:
            rec = cp / rel / b / "record"
            parts = np.memmap(rec / "parts.bin", dtype=np.float64, mode="r").reshape(STEPS, 18, 8)
            obs = np.memmap(rec / "obs.bin", dtype=np.float32, mode="r").reshape(STEPS, 18, 1024)
            z = obs[:, :, :810].reshape(STEPS, 18, 90, 9)
            present = z[:, :, :, 0] > 0.5
            temp = z[:, :, :, 1] * 5.0 + 295.0
            occupied = obs[:, :, 982] > 0
            mask = occupied[:, :, None] & present
            cold = np.clip(BAND_LO - temp, 0.0, None) * mask
            hot = np.clip(temp - BAND_HI, 0.0, None) * mask
            for a, weather in enumerate(WEATHERS):
                p = parts[:, a, :]
                comfort = 0.6 * abs(p[:, 0].mean())
                energy = 0.2 * p[:, 1].mean()
                carbon = 0.2 * p[:, 2].mean()
                c, h = cold[:, a].sum(), hot[:, a].sum()
                rows.append(dict(set=tag, building=b, weather=weather,
                                 score=comfort + energy + carbon,
                                 comfort=comfort, energy=energy, carbon=carbon,
                                 cold_share=float(c / (c + h)) if c + h else 0.5,
                                 deviation=float((c + h) / max(mask[:, a].sum(), 1))))
    rows.sort(key=lambda r: r["score"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=Path,
                    default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = ap.parse_args()

    rows = month_rows(args.cp)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "fig_appendix_terms.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    third = n // 3
    bounds = [(0, third), (third, 2 * third), (2 * third, n)]
    labels = ["low score", "middle", "high score"]

    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "font.size": 7, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.05))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.26, 0.95], hspace=0.10,
                          left=0.075, right=0.985, top=0.93, bottom=0.21)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    x = np.arange(n)

    # (a) the three terms, stacked, one bar per month
    bottom = np.zeros(n)
    for key in ("comfort", "energy", "carbon"):
        v = np.array([r[key] for r in rows])
        ax1.bar(x, v, bottom=bottom, width=1.0, color=TERM_COL[key], linewidth=0,
                label=f"{key} {v.mean():.4f}")
        bottom += v
    for (a, b), lab in zip(bounds, labels):
        m = np.mean([r["score"] for r in rows[a:b]])
        ax1.axvspan(a - 0.5, b - 0.5, color="#000000", alpha=0.035, zorder=0)
        ax1.text((a + b) / 2 - 0.5, 1.015, f"{lab} ({m:.4f})", transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=6.6, color="#333333")
    ax1.set_ylabel("score per step")
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(0, max(r["score"] for r in rows) * 1.06)
    ax1.legend(loc="upper left", fontsize=6.2, frameon=False, ncol=3, handlelength=1.0,
               columnspacing=1.1, bbox_to_anchor=(0.005, 1.0), title="mean over the 144 months",
               title_fontsize=6.2)
    ax1.tick_params(labelbottom=False, labelsize=6.4)

    # (b) building strip
    for i, r in enumerate(rows):
        ax2.add_patch(plt.Rectangle((i - 0.5, 0), 1.0, 1.0,
                                    color=BUILDING_COL[r["building"]], linewidth=0))
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.tick_params(labelbottom=False)
    ax2.set_ylabel("building", rotation=0, ha="right", va="center", fontsize=6.4, labelpad=6)

    # (c) cold against hot, diverging about zero
    cold = np.array([r["cold_share"] for r in rows])
    ax3.bar(x, cold, width=1.0, color="#2C7FB8", linewidth=0)
    ax3.bar(x, -(1 - cold), width=1.0, color="#D95F4B", linewidth=0)
    ax3.axhline(0.0, color="white", lw=0.6)
    for (a, b), lab in zip(bounds, labels):
        ax3.axvspan(a - 0.5, b - 0.5, color="#000000", alpha=0.035, zorder=0)
    ax3.set_ylim(-1, 1)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["hot", "", "cold"], fontsize=6.2)
    ax3.set_xlabel("the 144 scored months, sorted by score", labelpad=5)
    ax3.set_xlim(-0.5, n - 0.5)
    ax3.tick_params(labelsize=6.2)

    handles = [Patch(color=BUILDING_COL[b], label=b.replace("building_", "")) for b in BUILDING_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=6.0,
               handlelength=0.8, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    for ax in (ax1, ax2, ax3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.savefig(args.out / "fig_appendix_terms.pdf", facecolor="white")
    fig.savefig(args.out / "fig_appendix_terms.png", dpi=450, facecolor="white")
    print(f"wrote {args.out/'fig_appendix_terms.pdf'}  ({n} months)")


if __name__ == "__main__":
    main()

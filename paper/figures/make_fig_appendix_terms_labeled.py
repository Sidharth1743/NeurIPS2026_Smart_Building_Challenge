"""Companion to fig_appendix_terms: the same 144 bars, with the weather station of
every month added as a second strip so a reader can name any bar.

Same data as make_fig_appendix_terms.py (imported, not recomputed), same order of
months, same terms and cold/hot panel. Writes fig_appendix_terms_labeled.pdf/.png.

    python make_fig_appendix_terms_labeled.py [--cp DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from make_fig_appendix_terms import (BAND_HI, BAND_LO, BUILDING_COL, BUILDING_ORDER,
                                     TERM_COL, WEATHERS, month_rows)

WEATHER_COL = {w: c for w, c in zip(WEATHERS, plt.get_cmap("tab20").colors)}
SHORT = {w: w.replace("minneapolis", "mpls").replace("fairbanks", "fair")
         .replace("singapore", "sing").replace("phoenix", "phx")
         .replace("atlanta", "atl").replace("denver", "den").replace("_", "") for w in WEATHERS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = ap.parse_args()

    rows = month_rows(args.cp)
    n = len(rows)
    third = n // 3
    bounds = [(0, third), (third, 2 * third), (2 * third, n)]
    labels = ["low score", "middle", "high score"]

    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "font.size": 8.0, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(5.6, 5.25))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 0.24, 0.24, 0.85], hspace=0.10,
                          left=0.105, right=0.988, top=0.955, bottom=0.30)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i]) for i in range(4))
    x = np.arange(n)

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
                 ha="center", va="bottom", fontsize=7.8, color="#333333")
    ax1.set_ylabel("score per step")
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(0, max(r["score"] for r in rows) * 1.06)
    ax1.legend(loc="upper left", fontsize=7.4, frameon=False, ncol=3, handlelength=1.0,
               columnspacing=1.1, bbox_to_anchor=(0.005, 1.0), title="mean over the 144 months",
               title_fontsize=7.4)
    ax1.tick_params(labelbottom=False, labelsize=7.4)

    for i, r in enumerate(rows):
        ax2.add_patch(plt.Rectangle((i - 0.5, 0), 1.0, 1.0, color=BUILDING_COL[r["building"]],
                                    linewidth=0))
        ax3.add_patch(plt.Rectangle((i - 0.5, 0), 1.0, 1.0, color=WEATHER_COL[r["weather"]],
                                    linewidth=0))
    for ax, lab in ((ax2, "building"), (ax3, "station")):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7.4, labelpad=6)

    cold = np.array([r["cold_share"] for r in rows])
    ax4.bar(x, cold, width=1.0, color="#2C7FB8", linewidth=0)
    ax4.bar(x, -(1 - cold), width=1.0, color="#D95F4B", linewidth=0)
    ax4.axhline(0.0, color="white", lw=0.6)
    for (a, b), lab in zip(bounds, labels):
        ax4.axvspan(a - 0.5, b - 0.5, color="#000000", alpha=0.035, zorder=0)
    ax4.set_ylim(-1, 1)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(["hot", "", "cold"], fontsize=7.4)
    ax4.set_xlabel("the 144 scored months, sorted by score", labelpad=5, fontsize=7.8)
    ax4.set_xlim(-0.5, n - 0.5)
    ax4.tick_params(labelsize=7.4)

    bh = [Patch(color=BUILDING_COL[b], label=b.replace("building_", "")) for b in BUILDING_ORDER]
    l1 = fig.legend(handles=bh, loc="lower center", ncol=8, frameon=False, fontsize=7.0,
                    handlelength=0.8, columnspacing=0.8, bbox_to_anchor=(0.5, 0.13),
                    title="building", title_fontsize=7.0)
    wh = [Patch(color=WEATHER_COL[w], label=w) for w in WEATHERS]
    fig.legend(handles=wh, loc="lower center", ncol=6, frameon=False, fontsize=6.6,
               handlelength=0.7, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0),
               title="weather station of each month", title_fontsize=6.9)
    fig.add_artist(l1)
    for ax in (ax1, ax2, ax3, ax4):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.savefig(args.out / "fig_appendix_terms_labeled.pdf", facecolor="white")
    fig.savefig(args.out / "fig_appendix_terms_labeled.png", dpi=450, facecolor="white")
    print(f"wrote {args.out/'fig_appendix_terms_labeled.pdf'}  ({n} months)")


if __name__ == "__main__":
    main()

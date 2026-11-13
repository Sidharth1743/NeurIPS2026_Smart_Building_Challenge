#!/usr/bin/env python3
"""Figure 3, bold variant: same data and encoding as fig_results_breakdown4_v2,
with saturated colours and a distinct filled shape per policy so the series can be
told apart at print size and in greyscale.

Writes figures/fig_results_breakdown5.{pdf,png}; the shipped figure is untouched.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import results_fig_data as D

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 9.4, "axes.labelsize": 9.4, "axes.titlesize": 9.4,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.4, "legend.fontsize": 7.6,
    "axes.linewidth": 1.0, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
})

FIG = D.CP.parent / "paper-writing/figures"

# saturated, colour-blind-safe set, one shape per policy
BOLD = {"PID": "#B2182B", "SAC": "#762A83", "Long": "#404040", "Mid": "#1B7A8C", "Final": "#0B3D7A"}
SHAPE = {"PID": "P", "SAC": "D", "Long": "^", "Mid": "s", "Final": "o"}
SIZE = {"PID": 4.6, "SAC": 5.2, "Long": 4.6, "Mid": 4.6, "Final": 5.6}


def dec(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def stat(v):
    return (float(np.mean(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), float(np.max(v)))


def main() -> None:
    s = D.load_scores(); s54 = D.load_scores54()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.66),
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})
    off = {p: (i - (len(D.POL) - 1) / 2) * 0.16 for i, p in enumerate(D.POL)}
    pos = {b: i for i, b in enumerate(D.BUILDINGS)}
    for i, b in enumerate(D.EXT):
        pos[b] = 6.0 + i

    def panel(ax, groups, xs_of, stats):
        for g in groups:
            if g in ("building_train_3", "building_train_6"):
                ax.axvspan(xs_of(g) - 0.48, xs_of(g) + 0.48, color="#EFEFEF", zorder=0)
        for p in D.POL:
            xs = [xs_of(g) + off[p] for g in groups]
            mus = [stats[g][p][0] for g in groups]
            los = [stats[g][p][1] for g in groups]
            his = [stats[g][p][2] for g in groups]
            wst = [stats[g][p][3] for g in groups]
            ax.vlines(xs, his, wst, color=BOLD[p], lw=0.6, alpha=0.28, zorder=1)
            ax.vlines(xs, los, his, color=BOLD[p], lw=1.5, alpha=0.60, zorder=2)
            ax.plot(xs, wst, "_", ms=5.0, mew=1.3, color=BOLD[p], alpha=0.8, zorder=3)
            ax.plot(xs, mus, SHAPE[p], ms=SIZE[p], color=BOLD[p], mec="white", mew=0.6, zorder=5)
        for g in groups:
            ax.plot([xs_of(g) + off[p] for p in D.POL], [stats[g][p][0] for p in D.POL],
                    "-", color="#8A8A8A", lw=0.8, alpha=0.55, zorder=4)

    groups = D.BUILDINGS + D.EXT
    st = {}
    for b in groups:
        d = {}
        for p in D.POL:
            v = ([s54[p][k] for k in s54[p] if k.startswith(b + "__")] if b in D.EXT
                 else [s[p][k][0] for k in s[p] if k.startswith(b + "__")])
            d[p] = stat(v)
        st[b] = d
    panel(ax, groups, lambda b: pos[b], st)
    ax.axvline(5.0, color="#BBBBBB", lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.006, 0.135); ax.set_xlim(-0.7, 9.2)
    dec(ax, [0.01, 0.02, 0.05, 0.1])
    ax.set_xticks([pos[b] for b in groups])
    ax.set_xticklabels(["1", "2", "3*", "6*", "8", "4", "5", "7"])
    ax.set_xlabel("building (train_N; * never trained)")
    ax.set_ylabel("score over\nthe 18 windows", labelpad=1.5)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title("90 held-out months                         54-month extension",
                 fontsize=7.6, color="#444444", pad=3)

    st2 = {}
    for stn in D.STATIONS:
        d = {}
        for p in D.POL:
            v = [s[p][k][0] for k in s[p] if D.station(k) == stn]
            d[p] = stat(v)
        st2[stn] = d
    panel(bx, D.STATIONS, lambda g: D.STATIONS.index(g), st2)
    bx.axvspan(D.STATIONS.index("minneapolis") - 0.48,
               D.STATIONS.index("minneapolis") + 0.48, color="#EFEFEF", zorder=0)
    for g in D.STATIONS:
        bx.plot([D.STATIONS.index(g) + off[p] for p in D.POL],
                [st2[g][p][0] for p in D.POL], "-", color="#8A8A8A", lw=0.8, alpha=0.55, zorder=4)
    bx.set_yscale("log"); bx.set_ylim(0.006, 0.135); bx.set_xlim(-0.7, 5.7)
    dec(bx, [0.01, 0.02, 0.05, 0.1])
    bx.set_xticks(range(6))
    bx.set_xticklabels(["Atlanta", "Denver", "Fairbanks", "Mpls*", "Phoenix", "Singapore"],
                       rotation=28, ha="right")
    bx.set_xlabel("climate (station)")
    bx.yaxis.grid(True, color="#EDEDED", lw=0.6, zorder=0); bx.set_axisbelow(True)

    h = [Line2D([], [], color=BOLD[p], marker=SHAPE[p], ms=SIZE[p] + 0.6, lw=1.2, label=p)
         for p in D.POL]
    enc = [Line2D([], [], color="#555555", marker="o", ms=4.0, lw=0, label="mean"),
           Line2D([], [], color="#555555", lw=1.2, label="10th--90th"),
           Line2D([], [], color="#555555", marker="_", ms=5.5, mew=1.3, lw=0,
                  label="worst window")]
    order = [h[0], enc[0], h[1], enc[1], h[2], enc[2], h[3], h[4]]
    fig.legend(handles=order, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.075),
               handlelength=1.1, columnspacing=1.4, labelspacing=0.35, fontsize=7.2)
    fig.tight_layout(pad=0.4, rect=(0.01, 0.0, 1, 0.90))

    fig.savefig(FIG / "fig_results_breakdown5.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "fig_results_breakdown5.png", bbox_inches="tight", pad_inches=0.02,
                dpi=300)
    print("wrote fig_results_breakdown5")
    plt.close(fig)


if __name__ == "__main__":
    main()

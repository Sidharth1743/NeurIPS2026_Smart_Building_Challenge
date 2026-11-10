#!/usr/bin/env python3
"""Results figures 2 and 3, second pass: same encodings as the captions, clearer read.

Writes fig_results_scores_v2 and fig_results_breakdown4_v2; the originals are untouched.
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
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.linewidth": 1.0, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "svg.fonttype": "none",
    "figure.dpi": 200, "savefig.dpi": 300,
})

FIG = D.CP.parent / "paper-writing/figures"
TERM = ["#0F4D92", "#6C9BD2", "#BCD0EA"]
TERM_NAMES = ["comfort", "energy", "carbon"]
ORDER = ["Final", "Mid", "Long", "PID"]
GREY = "#8A8A8A"


def dec(ax, ticks, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    print("wrote", name)


def fig_scores():
    s = D.load_scores()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.50),
                                 gridspec_kw={"width_ratios": [1.0, 1.35]})

    # ---- (a) the score as the sum of its three terms -----------------------
    y = np.arange(len(ORDER))[::-1]
    left = np.zeros(len(ORDER))
    comfort = []
    for k, (c, nm) in enumerate(zip(TERM, TERM_NAMES)):
        vals = np.array([np.mean([v[1 + k] for v in s[p].values()]) for p in ORDER])
        ax.barh(y, vals, left=left, height=0.62, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        if k == 0:
            comfort = vals
        left += vals
    for yi, p, cf in zip(y, ORDER, comfort):
        i = ORDER.index(p)
        ax.text(cf / 2, yi, f"{100 * cf / left[i]:.0f}%", ha="center", va="center",
                fontsize=7.0, color="white", zorder=4)
        ax.text(left[i] + 0.0016, yi, f"{left[i]:.4f}", va="center", ha="left",
                fontsize=8.0, color="#1A1A1A", fontweight="bold" if p == "Final" else "normal")
    ax.set_yticks(y); ax.set_yticklabels(ORDER, fontsize=8.5)
    ax.set_xlim(0, 0.058)
    dec(ax, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05], axis="x")
    ax.set_xlabel("score (sum of the three terms)")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in TERM]
    ax.legend(handles, TERM_NAMES, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3,
              handlelength=0.8, handleheight=0.8, columnspacing=0.9, borderaxespad=0.0)

    # ---- (b) score against training steps ----------------------------------
    pid90 = 0.04609
    bx.axhline(pid90, color=D.COL["PID"], lw=1.3, ls=(0, (1.4, 1.9)), zorder=2)
    bx.text(4, 0.0443, "PID, 90 months", color=D.COL["PID"], fontsize=7.2, va="top")

    loc = {"Long": [(89.579520, 0.02532), (119.43936, 0.02723)],
           "Mid": [(159.25248, 0.01853)],
           "Final": [(199.0656, 0.01651)]}
    pub = {k: D.PUBLIC[k] for k in ("Long", "Mid", "Final")}

    # the two saves of the later run: one solid (90 months), one open (public)
    bx.plot([159.25248, 199.0656], [0.01853, 0.01651], "-", color=D.COL["Final"], lw=1.0,
            alpha=0.45, zorder=3)
    bx.plot([159.25248, 199.0656], [0.01750, 0.02944], "--", color=D.COL["Final"], lw=1.0,
            alpha=0.45, zorder=3)

    for name in ("Long", "Mid", "Final"):
        xs = [x for x, _ in loc[name]]; ys = [v for _, v in loc[name]]
        bx.plot(xs, ys, "-", color=D.COL[name], lw=1.4, zorder=4)
        bx.plot(xs, ys, "o", ms=6.0, mfc=D.COL[name], mec="white", mew=0.8, zorder=6)
        px = sorted(pub[name]); py = [pub[name][x] for x in px]
        bx.plot(px, py, "--", color=D.COL[name], lw=1.2, zorder=4)
        bx.plot(px, py, "o", ms=6.0, mfc="white", mec=D.COL[name], mew=1.3, zorder=6)

    bx.text(243, 0.0240, "Long", color=D.COL["Long"], fontsize=7.6, ha="left", va="center")
    bx.text(152.5, 0.0181, "Mid", color=D.COL["Mid"], fontsize=7.6, ha="right", va="center")
    bx.text(203, 0.0170, "Final", color=D.COL["Final"], fontsize=7.6, ha="left", va="center")

    bx.text(300, 0.0322, "Long budget 1.5B,\nstopped at 507.6M", fontsize=6.8,
            color="#6A6A6A", ha="left", va="center", linespacing=1.3)

    bx.set_yscale("log"); bx.set_ylim(0.0125, 0.052); bx.set_xlim(0, 525)
    dec(bx, [0.02, 0.03, 0.04, 0.05])
    bx.set_xticks([0, 100, 200, 300, 400, 500])
    bx.yaxis.grid(True, color="#EDEDED", lw=0.6, zorder=0); bx.set_axisbelow(True)
    bx.set_xlabel("training steps (M)"); bx.set_ylabel("score (lower is better)")
    h = [Line2D([], [], color="#4A4A4A", marker="o", ms=5.5, lw=1.2, label="90 months"),
         Line2D([], [], color="#4A4A4A", marker="o", ms=5.5, lw=1.2, mfc="white", label="public")]
    bx.legend(handles=h, loc="lower right", handlelength=1.6, borderaxespad=0.4,
              labelspacing=0.35)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_results_scores_v2")
    plt.close(fig)


def stat(vals):
    v = np.sort(np.asarray(vals))
    return v.mean(), np.percentile(v, 10), np.percentile(v, 90), v.max()


def fig_breakdown():
    s = D.load_scores(); s54 = D.load_scores54()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.45),
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})
    off = {"PID": -0.26, "Long": -0.09, "Mid": 0.09, "Final": 0.26}
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
            ax.vlines(xs, his, wst, color=D.COL[p], lw=0.5, alpha=0.18, zorder=1)
            ax.vlines(xs, los, his, color=D.COL[p], lw=1.0, alpha=0.38, zorder=2)
            ax.plot(xs, wst, "_", ms=5.0, mew=1.0, color=D.COL[p], alpha=0.6, zorder=3)
            ax.plot(xs, mus, "o", ms=4.4 if p == "Final" else 3.6, color=D.COL[p],
                    mec="white", mew=0.45, zorder=5)
        # the staircase: the four means of each group, joined
        for g in groups:
            ax.plot([xs_of(g) + off[p] for p in D.POL], [stats[g][p][0] for p in D.POL],
                    "-", color="#9A9A9A", lw=0.7, alpha=0.55, zorder=4)

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
                 fontsize=7.2, color="#444444", pad=3)

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
                [st2[g][p][0] for p in D.POL], "-", color="#9A9A9A", lw=0.7, alpha=0.55, zorder=4)
    bx.set_yscale("log"); bx.set_ylim(0.006, 0.135); bx.set_xlim(-0.7, 5.7)
    dec(bx, [0.01, 0.02, 0.05, 0.1])
    bx.set_xticks(range(6))
    bx.set_xticklabels(["Atlanta", "Denver", "Fairbanks", "Mpls*", "Phoenix", "Singapore"],
                       rotation=28, ha="right")
    bx.set_xlabel("climate (station)")
    bx.yaxis.grid(True, color="#EDEDED", lw=0.6, zorder=0); bx.set_axisbelow(True)

    h = [Line2D([], [], color=D.COL[p], marker="o", ms=4.0, lw=1.0, label=p) for p in D.POL]
    enc = [Line2D([], [], color="#555555", marker="o", ms=3.6, lw=0, label="mean"),
           Line2D([], [], color="#555555", lw=1.0, label="10th--90th"),
           Line2D([], [], color="#555555", marker="_", ms=5.0, mew=1.0, lw=0,
                  label="worst window")]
    order = [h[0], enc[0], h[1], enc[1], h[2], enc[2], h[3]]
    fig.legend(handles=order, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.055),
               handlelength=1.1, columnspacing=1.4, labelspacing=0.35, fontsize=6.8)
    fig.tight_layout(pad=0.4, rect=(0.01, 0.0, 1, 0.90))
    save(fig, "fig_results_breakdown4_v2")
    plt.close(fig)


if __name__ == "__main__":
    fig_scores()
    fig_breakdown()

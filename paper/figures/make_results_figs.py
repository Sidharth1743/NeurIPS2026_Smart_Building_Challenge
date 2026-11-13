#!/usr/bin/env python3
"""Three Results figures, four policies, figures4papers style (v2)."""
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
    "axes.linewidth": 1.1, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "svg.fonttype": "none",
    "figure.dpi": 200, "savefig.dpi": 300,
})

FIG = D.CP.parent / "paper-writing/figures"
TERM = ["#0F4D92", "#5B8FC9", "#A8C4E8"]
TERM_NAMES = ["comfort", "energy", "carbon"]
ORDER = ["Final", "Mid", "Long", "PID"]


def dec(ax, ticks):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    print("wrote", name)


def fig_scores():
    s = D.load_scores()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.3),
                                 gridspec_kw={"width_ratios": [1.0, 1.3]})

    y = np.arange(len(ORDER))[::-1]
    left = np.zeros(len(ORDER))
    for k, (c, nm) in enumerate(zip(TERM, TERM_NAMES)):
        vals = np.array([np.mean([v[1 + k] for v in s[p].values()]) for p in ORDER])
        ax.barh(y, vals, left=left, height=0.60, color=c, edgecolor="white", linewidth=1.0)
        left += vals
    for yi, p in zip(y, ORDER):
        ax.text(left[list(ORDER).index(p)] + 0.0012, yi, f"{left[list(ORDER).index(p)]:.4f}",
                va="center", ha="left", fontsize=7.5, color="#272727")
    ax.set_yticks(y); ax.set_yticklabels(ORDER, fontsize=8.5)
    ax.set_xlim(0, 0.061)
    ax.set_xlabel("score (sum of the three terms)")
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in TERM]
    ax.legend(handles, TERM_NAMES, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=0.9, columnspacing=1.0, borderaxespad=0.2)

    pid90 = 0.04609
    bx.axhline(pid90, color=D.COL["PID"], lw=1.2, ls=(0, (1.5, 2.0)))
    pl = D.PUBLIC["Long"]
    xp = np.array(sorted(pl)); yp = np.array([pl[x] for x in xp])
    bx.plot([89.57952, 119.43936], [0.02532, 0.02723], "-s", color=D.COL["Long"], lw=1.3, ms=4.5)
    bx.plot(xp, yp, "--s", color=D.COL["Long"], lw=1.1, ms=4.5, mfc="white", mec=D.COL["Long"])
    for p, x, yl, ypub in (("Mid", 159.25248, 0.01853, 0.01750), ("Final", 199.0656, 0.01651, 0.02944)):
        bx.plot(x, yl, "o", ms=6.5, mfc=D.COL[p], mec=D.COL[p], zorder=5)
        bx.plot(x, ypub, "o", ms=6.5, mfc="white", mec=D.COL[p], mew=1.3, zorder=5)
    bx.plot([159.25248, 199.0656], [0.01853, 0.01651], "-", color=D.COL["Final"], lw=1.2, zorder=4)
    bx.plot([159.25248, 199.0656], [0.01750, 0.02944], "--", color=D.COL["Final"], lw=1.2, zorder=4)

    bx.text(6, pid90 * 1.015, "PID", color=D.COL["PID"], fontsize=7.5, va="bottom")
    bx.text(62, 0.0265, "Long", color=D.COL["Long"], fontsize=7.5)
    bx.text(148, 0.0157, "Mid", color=D.COL["Mid"], fontsize=7.5)
    bx.text(204, 0.0171, "Final", color=D.COL["Final"], fontsize=7.5)
    bx.text(300, 0.0335, "budget 1.5B, stopped at 507.6M", fontsize=7, color=D.COL["Long"])
    bx.text(238, 0.0206, "199.9M, annealed to zero", fontsize=7, color=D.COL["Final"])

    bx.set_yscale("log"); bx.set_ylim(0.013, 0.06); bx.set_xlim(0, 520)
    dec(bx, [0.02, 0.03, 0.04, 0.05])
    bx.set_xlabel("training steps (M)"); bx.set_ylabel("score (lower is better)")
    h = [Line2D([], [], color="#555555", marker="s", ms=4.5, lw=0, label="90 months"),
         Line2D([], [], color="#555555", marker="s", ms=4.5, mfc="white", lw=0, label="public")]
    bx.legend(handles=h, loc="lower right", handlelength=0.8, borderaxespad=0.3)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_results_scores")
    plt.close(fig)


def stat(vals):
    v = np.sort(np.asarray(vals))
    return v.mean(), np.percentile(v, 10), np.percentile(v, 90), v.max()


def fig_breakdown():
    s = D.load_scores(); s54 = D.load_scores54()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                 gridspec_kw={"width_ratios": [1.5, 1.0]})
    off = {p: (i - (len(D.POL) - 1) / 2) * 0.16 for i, p in enumerate(D.POL)}
    pos = {b: i for i, b in enumerate(D.BUILDINGS)}
    for i, b in enumerate(D.EXT):
        pos[b] = 6.0 + i
    for b in D.BUILDINGS + D.EXT:
        if b in ("building_train_3", "building_train_6"):
            ax.axvspan(pos[b] - 0.5, pos[b] + 0.5, color="#EDEDED", zorder=0)
    for b in D.EXT:
        ax.axvspan(pos[b] - 0.5, pos[b] + 0.5, color="#F7F7F7", zorder=0)
    for p in D.POL:
        xs, mus, los, his, wst = [], [], [], [], []
        for b in D.BUILDINGS:
            v = [s[p][k][0] for k in s[p] if k.startswith(b + "__")]
            m, lo, hi, mx = stat(v); xs.append(pos[b] + off[p]); mus.append(m)
            los.append(lo); his.append(hi); wst.append(mx)
        for b in D.EXT:
            v = [s54[p][k] for k in s54[p] if k.startswith(b + "__")]
            m, lo, hi, mx = stat(v); xs.append(pos[b] + off[p]); mus.append(m)
            los.append(lo); his.append(hi); wst.append(mx)
        ax.vlines(xs, los, his, color=D.COL[p], lw=0.8, alpha=0.35)
        ax.plot(xs, mus, "o", ms=4.2, color=D.COL[p], zorder=4)
        ax.plot(xs, wst, "_", ms=4.5, color=D.COL[p], mew=1.0, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.006, 0.13); ax.set_xlim(-0.7, 9.2)
    dec(ax, [0.01, 0.02, 0.05, 0.1])
    xt = [pos[b] for b in D.BUILDINGS + D.EXT]
    ax.set_xticks(xt)
    ax.set_xticklabels(["1", "2", "3*", "6*", "8", "4", "5", "7"])
    ax.set_xlabel("building (train_N; * never trained)")
    ax.set_ylabel("score over the 18 windows")
    tr = ax.get_xaxis_transform()
    ax.text(2, -0.30, "90 months", transform=tr, ha="center", va="top", fontsize=7,
            color="#555555")
    ax.text(7, -0.30, "54-month extension", transform=tr, ha="center", va="top", fontsize=7,
            color="#555555")

    for i, st in enumerate(D.STATIONS):
        if st == "minneapolis":
            bx.axvspan(i - 0.5, i + 0.5, color="#EDEDED", zorder=0)
    for p in D.POL:
        xs, mus, los, his, wst = [], [], [], [], []
        for i, st in enumerate(D.STATIONS):
            v = [s[p][k][0] for k in s[p] if D.station(k) == st]
            m, lo, hi, mx = stat(v); xs.append(i + off[p]); mus.append(m)
            los.append(lo); his.append(hi); wst.append(mx)
        bx.vlines(xs, los, his, color=D.COL[p], lw=0.8, alpha=0.35)
        bx.plot(xs, mus, "o", ms=4.2, color=D.COL[p], zorder=4)
        bx.plot(xs, wst, "_", ms=4.5, color=D.COL[p], mew=1.0, zorder=3)
    bx.set_yscale("log"); bx.set_ylim(0.006, 0.13); bx.set_xlim(-0.7, 5.7)
    dec(bx, [0.01, 0.02, 0.05, 0.1])
    bx.set_xticks(range(6))
    bx.set_xticklabels(["Atlanta", "Denver", "Fairbanks", "Mpls*", "Phoenix", "Singapore"],
                       rotation=25, ha="right")
    bx.set_xlabel("climate (station)")
    h = [Line2D([], [], color=D.COL[p], marker="o", ms=4.2, lw=1.0, label=p) for p in D.POL]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
    save(fig, "fig_results_breakdown4")
    plt.close(fig)


def fig_time():
    st = D.load_steps(); s = D.load_scores()
    months = {n: D.month_of(n) for n in s["Final"]}
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(7.0, 2.15),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
    for p in D.POL:
        r = np.concatenate([st[p][n] for n in sorted(st[p])])
        hrs = np.tile(D.HOUR, len(st[p]))
        vals = np.array([-r[hrs == h].mean() for h in range(24)])
        ax.plot(range(24), vals, "-o", ms=2.8, lw=1.3, color=D.COL[p], label=p)
    ax.axvspan(6, 19, color="#EDEDED", zorder=0)
    ax.set_yscale("log"); ax.set_ylim(4e-4, 3e-1); ax.set_xlim(-0.5, 23.5)
    dec(ax, [0.001, 0.01, 0.1])
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlabel("hour of day")
    ax.set_ylabel("score per step")
    ax.text(12.5, 1.4e-1, "occupied", fontsize=6.8, color="#555555", ha="center")
    ax.annotate("06:00", xy=(6, 0.128), xytext=(8.4, 0.10), fontsize=6.8,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"))

    mo = sorted(set(months.values()))
    for p in D.POL:
        vals = [np.mean([s[p][n][0] for n in s[p] if months[n] == m]) for m in mo]
        cx.plot(range(len(mo)), vals, "-o", ms=3.2, lw=1.3, color=D.COL[p], label=p)
    cx.set_yscale("log"); cx.set_ylim(0.006, 0.12)
    dec(cx, [0.01, 0.02, 0.05, 0.1])
    cx.set_xticks(range(len(mo)))
    cx.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"][:len(mo)], fontsize=6.2, rotation=60, ha="right")
    cx.set_xlabel("month the window starts")
    h = [Line2D([], [], color=D.COL[p], marker="o", ms=3.2, lw=1.3, label=p) for p in D.POL]
    fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03),
               handlelength=0.9, columnspacing=1.0, fontsize=6.8)
    fig.tight_layout(pad=0.5, rect=(0, 0.06, 1, 0.92))
    for lab, a in zip(("a", "b"), (ax, cx)):
        pos = a.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.012, f"({lab})", ha="center", va="bottom", fontsize=8)
    save(fig, "fig_results_time4")
    plt.close(fig)


if __name__ == "__main__":
    fig_scores(); fig_breakdown(); fig_time()

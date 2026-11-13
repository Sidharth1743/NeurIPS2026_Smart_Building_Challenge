#!/usr/bin/env python3
"""SAC design ablation: the 50M arms as one panel, Figure 9 style."""
from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT = "/data/neurips-2026-smart-buildings/paper-writing/figures"
CURVES = json.load(open("sac_curves.json"))

REF = ("r04_utd4_async_960_1m_s75", "reference: 4 passes, 256 batch", "#4D4D4D", "--", 1.2)
ARMS = [
    ("r05_utd4_async_960_1m_frozen_s75", "frozen normalizer", "#B64342", "-", 1.2),
    ("r06_utd15_async_960_1m_s75", "15 passes", "#42949E", "-", 1.2),
    ("r07_utd15_ln_960_1m_s75", "15 passes + LayerNorm", "#6FA36F", "-", 1.2),
    ("r11_structured_b2048_s75", "structured, batch 2048", "#0F4D92", "-", 1.6),
    ("r13_structured_zm_s75", "structured, zone memory", "#8C8C8C", "-", 1.2),
    ("r15_structured_beta_b2048_s75", "structured, Beta head", "#9A4D8E", "-", 2.2),
]
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 8.5,
    "axes.linewidth": 1.1,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
})


def curve(run):
    s = np.array(CURVES[run]["steps"])[1:]
    v = np.array(CURVES[run]["values"])[1:]
    k = np.ones(5) / 5
    return np.convolve(s, k, "valid"), np.convolve(v, k, "valid")


def main():
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    for run, label, col, ls, lw in [REF] + ARMS:
        s, v = curve(run)
        ax.plot(s, v, color=col, ls=ls, lw=lw, label=label)
    ax.set_yscale("log")
    ax.set_xlim(0, 55)
    ax.set_ylim(0.015, 0.2)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_yticks([0.015, 0.02, 0.03, 0.05, 0.1, 0.2])
    ax.set_yticklabels(["0.015", "0.02", "0.03", "0.05", "0.1", "0.2"], fontsize=7.2)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("training steps (M)", fontsize=9)
    ax.set_ylabel("training panel (lower is better)", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=4,
              fontsize=6.6, handlelength=2.0, columnspacing=1.6,
              labelspacing=0.35)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.765, bottom=0.185)
    fig.savefig(f"{OUT}/fig_sac_ablation.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/fig_sac_ablation.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_sac_ablation (single panel, 50M)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SAC curve for the appendix: the Gaussian 200M run's four scored saves, with the
Beta-head arm of the same 50M budget marked as a star."""
from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import results_fig_data as D

OUT = D.CP.parent / "paper-writing" / "figures"
GAUSS = D.CP.parent / "SAC/results/r14_structured_200m_s75/scored90"
BETA = D.CP.parent / "SAC/results/r15_structured_beta_b2048_s75/scored90/50135040.mean.json"
STEPS = [49.9728, 89.95104, 119.93472, 159.91296]
FILES = ["49972800", "89951040", "119934720", "159912960"]
BETA_STEP, BETA_LABEL_STEP = 50.135, 50.135


def main() -> None:
    sac = [json.loads((GAUSS / f"{f}.mean.json").read_text())["score"] for f in FILES]
    beta = json.loads(BETA.read_text())["score"]
    s = D.load_scores()
    ppo = float(np.mean([v[0] for v in s["Final"].values()]))
    pid = float(np.mean([v[0] for v in s["PID"].values()]))

    fig, ax = plt.subplots(figsize=(3.60, 1.95))
    ax.axhline(ppo, color=D.COL["Final"], ls="--", lw=1.0, zorder=2)
    ax.axhline(pid, color=D.COL["PID"], ls="--", lw=1.0, zorder=2)
    ax.plot(STEPS, sac, "-o", color=D.COL["SAC"], ms=3.8, lw=1.4, zorder=3)
    ax.plot([BETA_STEP], [beta], "*", color=D.COL["SAC"], ms=11, mec="white", mew=0.7, zorder=4)

    ax.annotate(f"{sac[0]:.4f}", (STEPS[0], sac[0]), xytext=(-6, -13), textcoords="offset points",
                ha="right", fontsize=6.4, color=D.COL["SAC"])
    ax.annotate(f"{sac[1]:.4f}", (STEPS[1], sac[1]), xytext=(-5, 4), textcoords="offset points", ha="right",
                fontsize=6.4, color=D.COL["SAC"])
    ax.annotate(f"{beta:.4f}", (BETA_STEP, beta), xytext=(9, -3), textcoords="offset points",
                fontsize=6.4, color=D.COL["SAC"], fontweight="bold")

    handles = [
        Line2D([], [], color=D.COL["SAC"], lw=1.4, marker="o", ms=3.8,
               label="Gaussian head, 200M run"),
        Line2D([], [], color=D.COL["SAC"], lw=0, marker="*", ms=9, mec="white", mew=0.6,
               label="Beta head, 50M, same recipe"),
        Line2D([], [], color=D.COL["PID"], ls="--", lw=1.0, label="PID, 90 months"),
        Line2D([], [], color=D.COL["Final"], ls="--", lw=1.0, label="PPO, 90 months"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.0, ncol=2,
           frameon=False, handlelength=1.5, labelspacing=0.25, columnspacing=1.2)

    ax.set_xlim(0, 170)
    ax.set_ylim(0, 0.085)
    ax.set_xlabel("training steps (M)", fontsize=7.8)
    ax.set_ylabel("score, 90 months", fontsize=7.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(labelsize=6.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.975, bottom=0.375)
    fig.savefig(OUT / "fig_sac_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_sac_curve.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    print("wrote fig_sac_curve", [round(v, 4) for v in sac], "beta", round(beta, 4),
          "ppo", round(ppo, 4), "pid", round(pid, 4))


if __name__ == "__main__":
    main()

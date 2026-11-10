"""Figure 2: one point per held-out month, ours against pid_water.

Positive score convention (mean absolute error against zero, lower is better), so a
point below the identity line is a month the policy wins. Colour encodes the building
on a single sequential blue ramp; the two buildings never seen in training are drawn
open, so "unseen" reads as a shape difference and does not need its own hue.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
SB_ROOT = Path(os.environ.get("SB_ROOT", next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parent.parents]
     if (p / 'complete_port').is_dir()), Path(__file__).resolve().parent)))


REPO = SB_ROOT
D = REPO / 'results/eval90_analysis/r139_199065600/data'
OUT = REPO / 'paper-writing/figures'

plt.rcParams.update({
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
})

s = pd.read_csv(D / 'scenarios_r139.csv')
ORDER = ['building_train_2', 'building_train_8', 'building_train_1',
         'building_train_6', 'building_train_3']
LABEL = {'building_train_2': 'train_2', 'building_train_8': 'train_8',
         'building_train_1': 'train_1', 'building_train_6': r'train_6$^*$',
         'building_train_3': r'train_3$^*$'}
UNSEEN = {'building_train_6', 'building_train_3'}

ramp = plt.get_cmap('Blues')
shade = dict(zip(ORDER, [0.42, 0.55, 0.68, 0.80, 0.95]))
lo, hi = 2.5e-4, 0.2

fig = plt.figure(figsize=(3.35, 2.55))
ax = fig.add_axes([0.155, 0.155, 0.825, 0.825])

ax.plot([lo, hi], [lo, hi], color='#9A9A9A', lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.fill_between([lo, hi], [lo, hi], [lo, lo], color='#0F4D92', alpha=0.035, zorder=0)

for b in ORDER:
    d = s[s.building == b]
    c = ramp(shade[b])
    if b in UNSEEN:
        ax.scatter(d.score_pid, d.score, s=15, facecolor='white', edgecolor=c,
                   linewidth=0.9, zorder=4)
    else:
        ax.scatter(d.score_pid, d.score, s=15, facecolor=c, edgecolor='white',
                   linewidth=0.35, zorder=3)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks([1e-3, 1e-2, 1e-1])
ax.set_yticks([1e-3, 1e-2, 1e-1])
ax.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
ax.set_yticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
ax.tick_params(labelsize=6.4, length=2.5, pad=2, width=0.7)
ax.grid(True, which='major', color='#EDEDED', lw=0.6, zorder=0)
ax.set_axisbelow(True)

ax.set_xlabel('pid_water', fontsize=7.4, labelpad=3)
ax.set_ylabel('ours', fontsize=7.4, labelpad=3)

ax.set_frame_on(False)
ax.minorticks_off()
ax.tick_params(which='major', length=3.0, width=0.7)
ax.text(0.975, 0.045, '89 of 90 months below the line', transform=ax.transAxes,
        fontsize=5.9, color='#3A3A3A', ha='right', va='bottom')

handles = [Line2D([], [], marker='o', ls='', markersize=4.2, markerfacecolor=ramp(shade[b]),
                  markeredgecolor='white' if b not in UNSEEN else ramp(shade[b]),
                  markeredgewidth=0.6 if b not in UNSEEN else 0.9,
                  markerfacecoloralt='white', label=LABEL[b]) for b in ORDER]
for h, b in zip(handles, ORDER):
    if b in UNSEEN:
        h.set_markerfacecolor('white')
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=False,
          fontsize=5.9, handletextpad=0.3, labelspacing=0.22, borderpad=0.0,
          title='building ($^*$ unseen)', title_fontsize=5.9)

fig.savefig(OUT / 'fig_results_scatter.pdf', facecolor='white')
fig.savefig(OUT / 'fig_results_scatter.png', dpi=500, facecolor='white')
print('wrote fig_results_scatter.pdf/.png')

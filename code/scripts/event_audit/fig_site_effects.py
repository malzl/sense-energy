"""Every site's reaction to heat waves with its own placebo interval (argument: heatwave or
cold). Sites in order of size. Bar: 95 % interval from random blocks of the site's ordinary
days in the same season. Filled point: different from zero after the adjustment for the number
of sites (false discovery rate 5 %). Small black point: the estimate after shrinkage towards
the mean of its group. Sites with complete meter data; no site is named.

analysis/event_audit.py (site_effect_tests.csv).
"""

import sys

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

kind = "cold spell" if len(sys.argv) > 1 and sys.argv[1].startswith("cold") else "heatwave"
t = table("site_effect_tests")
t = t[t["type"] == kind].sort_values("effect_pct").reset_index(drop=True)
x = np.arange(len(t))
fig, ax = style.figure(style.WIDE_SHORT)
style.reference_line(ax, 0.0)
for group in style.PLANT_GROUP_ORDER:
    d = t[t["chp"] == group]
    color = style.PLANT_GROUP_COLORS[group]
    ax.vlines(d.index, d["lo"], d["hi"], color=color, linewidth=0.7, alpha=0.6)
    sig = d["q"] < 0.05
    ax.scatter(d.index[sig], d.loc[sig, "effect_pct"], s=9, color=color, zorder=3, label=group)
    ax.scatter(
        d.index[~sig],
        d.loc[~sig, "effect_pct"],
        s=9,
        facecolor="white",
        edgecolor=color,
        linewidth=0.6,
        zorder=3,
    )
ax.scatter(x, t["shrunk_pct"], s=2.5, color="black", zorder=4, label="after shrinkage")
ax.set_xticks([])
ax.set_ylim(-45, 75)
ax.set_xlabel("Sites, in order of the reaction")
ax.set_ylabel("Reaction of demand (%)")
style.style_axis(ax, grid="y", wide_short=True)
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.14, bottom=0.22)
style.save_figure(fig, OUT / f"site_effects_{kind.replace(' ', '_')}", data=t)

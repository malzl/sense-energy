"""The cold and heat effects (specification D: date effects plus sunshine, wind and rain) with each
thermometer, on the site-days all thermometers cover. One plot.

95 % intervals two-way clustered by site and date (analysis/thermometers.py). Doses offset
horizontally.
"""

import numpy as np
from _common import OUT, thermometers

from sense_energy.visualization import style

e = thermometers("effects")
e = e[(e["block"] == "specs") & (e["spec"] == "D")]
names = [n for n in style.THERMOMETER_ORDER if n in set(e["thermometer"])]

fig, ax = style.figure(style.SINGLE_COLUMN)
offsets = {"hdd14": -0.12, "cdd20": 0.12}
for dose, off in offsets.items():
    d = e[e["term"] == dose].set_index("thermometer").reindex(names)
    x = np.arange(len(names)) + off
    ax.errorbar(
        x,
        d["coef_pct"],
        yerr=[d["coef_pct"] - d["lo_pct"], d["hi_pct"] - d["coef_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=style.LINEWIDTH["reference"],
        color=style.DOSE_COLORS[dose],
        label=style.DOSE_NAMES[dose],
    )
style.reference_line(ax, 0.0)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([style.THERMOMETER_NAMES[n] for n in names])
ax.set_xlabel("Site thermometer")
ax.set_ylabel("Demand change (% per °C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
print(e.pivot_table(index="thermometer", columns="term", values="coef_pct").round(2).to_string())
style.save_figure(fig, OUT / "thermometer_effects", data=e)

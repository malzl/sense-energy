"""Effect of a degree of cold (below 14 degC) and of heat (above 20 degC) on daily demand at sites
without CHP, across nested fixed-effect specifications. One plot.

A: site x year-month, site x weekday and calendar controls; B: + date effects (only the day's
cross-site temperature gradient identifies the effect); C: + region x date. 95 % intervals
two-way clustered by site and date (analysis/weather_causal.py).
"""

import numpy as np
from _common import OUT, causal

from sense_energy.visualization import style

s = causal("twfe_specs")
s = s[s["group"] == "no CHP"]

fig, ax = style.figure(style.SINGLE_COLUMN)
offsets = {"hdd14": -0.12, "cdd20": 0.12}
for dose, off in offsets.items():
    d = s[s["term"] == dose].set_index("spec").reindex(style.SPEC_ORDER)
    x = np.arange(len(style.SPEC_ORDER)) + off
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
ax.set_xticks(range(len(style.SPEC_ORDER)))
ax.set_xticklabels([style.SPEC_NAMES[k] for k in style.SPEC_ORDER])
ax.set_xlabel("Fixed effects")
ax.set_ylabel("Demand change (% per °C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
print(s[["spec", "term", "coef_pct", "lo_pct", "hi_pct", "p"]].round(3).to_string(index=False))
style.save_figure(fig, OUT / "causal_specs", data=s)

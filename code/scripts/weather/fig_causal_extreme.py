"""Extra demand on days whose maximum reaches 28-30 or 30 degC and above, beyond the linear heat
dose, at sites without CHP, across the nested specifications. One plot.

Indicators added to the specifications of analysis/weather_causal.py; 95 % intervals two-way
clustered by site and date. Indicators offset horizontally.
"""

import numpy as np
from _common import OUT, causal

from sense_energy.visualization import style

e = causal("twfe_extreme")
e = e[(e["group"] == "no CHP") & e["term"].isin(style.EXTREME_DAY_NAMES)]

fig, ax = style.figure(style.SINGLE_COLUMN)
offsets = {"hot28": -0.12, "hot30": 0.12}
for term, off in offsets.items():
    d = e[e["term"] == term].set_index("spec").reindex(style.SPEC_ORDER)
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
        color=style.EXTREME_DAY_COLORS[term],
        label=style.EXTREME_DAY_NAMES[term],
    )
style.reference_line(ax, 0.0)
ax.set_xticks(range(len(style.SPEC_ORDER)))
ax.set_xticklabels([style.SPEC_NAMES[k] for k in style.SPEC_ORDER])
ax.set_xlabel("Fixed effects")
ax.set_ylabel("Extra demand beyond the heat dose (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
print(
    e[["spec", "term", "coef_pct", "lo_pct", "hi_pct", "p", "site_days_30plus"]]
    .round(2)
    .to_string(index=False)
)
style.save_figure(fig, OUT / "causal_extreme", data=e)

"""Extra demand on the 2nd and 3rd+ consecutive hot or cold day, beyond that day's own temperature. One plot.

Hot day: T_max at or above the site's Met Office heatwave threshold; cold day: T_mean <= 2 degC.
Coefficients from the binned fixed-effect regression with consecutive-day terms
(analysis/weather_response.py); 95 % intervals clustered by week. Groups offset horizontally.
"""

import numpy as np
from _common import OUT, response

from sense_energy.visualization import style

p = response("persistence")
terms = list(style.PERSISTENCE_NAMES)
p = p[p["term"].isin(terms)]

fig, ax = style.figure(style.SINGLE_COLUMN)
offsets = np.linspace(-0.2, 0.2, len(style.SITE_GROUP_ORDER))
for off, g in zip(offsets, style.SITE_GROUP_ORDER, strict=True):
    d = p[p["group"] == g].set_index("term").reindex(terms)
    x = np.arange(len(terms)) + off
    ax.errorbar(
        x,
        d["coef_pct"],
        yerr=[d["coef_pct"] - d["lo_pct"], d["hi_pct"] - d["coef_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=style.LINEWIDTH["reference"],
        color=style.SITE_GROUP_COLORS[g],
        label=g,
    )
style.reference_line(ax, 0.0)
ax.set_xticks(range(len(terms)))
ax.set_xticklabels([style.PERSISTENCE_NAMES[t] for t in terms])
ax.set_ylabel("Extra demand beyond same-day temperature (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
print(p.pivot_table(index="term", columns="group", values="coef_pct").round(2).to_string())
style.save_figure(fig, OUT / "weather_persistence", data=p)

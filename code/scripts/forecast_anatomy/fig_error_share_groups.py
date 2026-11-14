"""Share of the panel's metered demand and share of its forecast error (kWh) by site group,
best model (Chronos-2 with forecast weather), 91 sites, 69 test days. The demand of a CHP site
is its grid import.

analysis/forecast_anatomy.py (error_by_group.csv).
"""

import numpy as np
import pandas as pd
from _common import BEST, FA, OUT

from sense_energy.visualization import style

g = pd.read_csv(FA / "error_by_group.csv")
g = g[g["model"] == BEST].set_index("group").reindex(style.SITE_GROUP_ORDER)
fig, ax = style.figure(style.SINGLE_COLUMN)
x = np.arange(len(g))
ax.bar(x - 0.2, g["demand_share_pct"], width=0.38, color=style.ROLE_COLORS["zero_effect"])
ax.bar(
    x + 0.2,
    g["error_share_pct"],
    width=0.38,
    color=[style.SITE_GROUP_COLORS[k] for k in g.index],
)
handles = [
    ax.bar([0], [0], color=style.ROLE_COLORS["zero_effect"]),
    ax.bar([0], [0], color="#000000"),
]
ax.set_xticks(x)
ax.set_xticklabels(list(g.index))
ax.set_ylim(0, 100)
ax.set_ylabel("Share of the panel (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, handles=handles, labels=["metered demand", "forecast error"])
style.save_figure(
    fig,
    OUT / "error_share_groups",
    data=g.reset_index()[["group", "sites", "demand_share_pct", "error_share_pct", "nmae_pct"]],
)

"""Daily demand against daily footfall at the three sites with counts (2024), both with month,
weekday, weather and holidays taken out. Values beyond the axes are drawn at the edge.

analysis/forecast_anatomy.py (mobility_partial.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.analysis import weather_response as wr
from sense_energy.visualization import style

m = pd.read_csv(FA / "mobility_partial.csv")
m["group"] = m["site_code"].map(wr.site_groups())
m["x"], m["y"] = m["footfall_pct"].clip(-60, 60), m["demand_pct"].clip(-40, 40)
fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0)
for group in ("on-site CHP", "outpatient", "inpatient"):
    d = m[m["group"] == group]
    if d.empty:
        continue
    ax.plot(
        d["x"],
        d["y"],
        linestyle="none",
        marker="o",
        markersize=2,
        markeredgewidth=0,
        alpha=0.5,
        color=style.SITE_GROUP_COLORS[group],
        label=group,
    )
ax.set_xlim(-60, 60)
ax.set_ylim(-40, 40)
ax.set_xlabel("Footfall against its expected value (%)")
ax.set_ylabel("Demand against its expected value (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False, bottom=0.26)
style.save_figure(
    fig,
    OUT / "footfall_partial",
    data=m[["site_code", "group", "date", "footfall_pct", "demand_pct"]],
)

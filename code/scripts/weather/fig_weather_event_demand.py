"""Electricity demand during each national weather event relative to same-weekday reference days. One plot.

One point per event with at least 20 site-days of demand data; ratio = median over affected
site-days of event-day demand / median of the same weekday 1 and 2 weeks either side.
Points are offset horizontally by event date within each type (no random jitter).
"""

import matplotlib.ticker as mticker
import numpy as np
from _common import OUT, national_events

from sense_energy.visualization import style

MIN_SITE_DAYS = 20
nat = national_events()
nat = nat[nat["demand_site_days"] >= MIN_SITE_DAYS].copy()

fig, ax = style.figure(style.SINGLE_COLUMN)
order = list(style.WEATHER_EVENT_ORDER)
for i, kind in enumerate(order):
    g = nat[nat["type"] == kind].sort_values("start")
    if g.empty:
        continue
    x = i + (np.linspace(-0.25, 0.25, len(g)) if len(g) > 1 else np.zeros(1))
    ax.plot(
        x,
        g["demand_ratio_median"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.WEATHER_EVENT_COLORS[kind],
    )
style.reference_line(ax, 1.0)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_ylabel("Event-day demand / reference (×)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
style.style_axis(ax, grid="y")
print(nat.groupby("type")["demand_ratio_median"].agg(["count", "median"]).round(3).to_string())
style.save_figure(fig, OUT / "weather_event_demand", data=nat)

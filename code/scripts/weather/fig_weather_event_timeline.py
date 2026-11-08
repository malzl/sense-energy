"""National extreme-weather events by type over time; bar height = share of sites affected. One plot.

Dashed line: last day of the half-hourly electricity record (events after it have no demand data).
"""

import json

import pandas as pd
from _common import EVENTS, OUT, national_events

from sense_energy.visualization import style

nat = national_events()
summary = json.loads((EVENTS / "summary.json").read_text())
n_sites = summary["sites"]
elec_end = pd.Timestamp(summary["electricity_window"][1])

fig, ax = style.figure(style.WIDE_SHORT)
order = list(style.WEATHER_EVENT_ORDER)
for i, kind in enumerate(order):
    g = nat[nat["type"] == kind]
    for e in g.itertuples():
        h = 0.8 * e.sites / n_sites
        ax.bar(
            e.start,
            h,
            width=pd.Timedelta(days=e.days),
            bottom=i - h / 2,
            align="edge",
            color=style.WEATHER_EVENT_COLORS[kind],
            linewidth=0,
        )
style.reference_line(ax, elec_end, axis="x")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlabel("Date")
style.style_axis(ax, wide_short=True)
print(nat["type"].value_counts().reindex(order).to_string())
style.save_figure(fig, OUT / "weather_event_timeline", data=nat)

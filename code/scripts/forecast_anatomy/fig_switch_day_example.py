"""One site with on-site CHP on a day when its plant stops (27 January 2026, a stored test day
of the site with the second largest error): observed grid import, the day-ahead median of the
best model and its 80 % interval.

analysis/forecast_anatomy.py (example_days.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

SITE, DAY = "RXN02", "2026-01-27"
e = pd.read_csv(FA / "example_days.csv", parse_dates=["target"])
d = e[(e["site_code"] == SITE) & (e["day"] == DAY)].sort_values("target").copy()
local = d["target"].dt.tz_convert("Europe/London")
d["hour"] = local.dt.hour + local.dt.minute / 60.0
c = style.method_color("Chronos-2 + IFS ENS")
fig, ax = style.figure(style.SINGLE_COLUMN)
ax.fill_between(d["hour"], d["q10"], d["q90"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
ax.plot(d["hour"], d["point"], color=c, linewidth=style.LINEWIDTH["primary"], label="forecast")
ax.plot(
    d["hour"], d["y"], color="#000000", linewidth=style.LINEWIDTH["secondary"], label="observed"
)
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_ylim(0, None)
ax.set_xlabel("Hour of the day (local time)")
ax.set_ylabel("Grid import (kW)")
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(
    fig,
    OUT / "switch_day_example",
    data=d[["site_code", "day", "hour", "y", "point", "q10", "q90"]],
)

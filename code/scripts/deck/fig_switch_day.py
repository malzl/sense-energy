"""Finding 2, chart: one site with on-site generation on a day when its plant stops. Observed
grid import against the day-ahead forecast of the best model with its 80 % band. The day is one
of the stored test days of the site with the second largest error (27 January 2026): the plant
is off from 09:00 to 18:00 and the import is three times the forecast."""

import pandas as pd
from _deck import BLUE, CHARCOAL, REPORTS, figure, legend_below, save

SITE, DAY = "RXN02", "2026-01-27"
e = pd.read_csv(REPORTS / "forecast_anatomy" / "example_days.csv", parse_dates=["target"])
d = e[(e["site_code"] == SITE) & (e["day"] == DAY)].sort_values("target").copy()
local = d["target"].dt.tz_convert("Europe/London")
d["hour"] = local.dt.hour + local.dt.minute / 60.0
for c in ("y", "point", "q10", "q90"):
    d[c] = d[c] / 1000.0  # MW
fig, ax = figure(42, 39)
ax.fill_between(d["hour"], d["q10"], d["q90"], color=BLUE, alpha=0.22, linewidth=0)
ax.plot(d["hour"], d["point"], color=BLUE, linewidth=1.1, label="forecast, 80 % band")
ax.plot(d["hour"], d["y"], color=CHARCOAL, linewidth=1.1, label="measured")
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_ylim(0, None)
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Grid import (MW)")
ax.grid(axis="y")
legend_below(fig, ax, ncol=2, handlelength=1.2)
save(fig, "switch_day", data=d[["hour", "y", "point", "q10", "q90"]])
print(round(d["y"].max(), 2), "MW measured peak;", round(d["point"].max(), 2), "MW forecast peak")

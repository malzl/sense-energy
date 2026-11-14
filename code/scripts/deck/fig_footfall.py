"""Finding 3, right: daily visitors against daily demand at the two footfall sites without
on-site generation (2024), both as the difference from the value that month, weekday, weather
and holidays give. The line is the fit: its slope is the elasticity."""

import numpy as np
import pandas as pd
from _deck import BLUE, ORANGE, REPORTS, figure, save

SITES = ("RH802", "RH8K6")
m = pd.read_csv(REPORTS / "forecast_anatomy" / "mobility_partial.csv")
m = m[m["site_code"].isin(SITES)]
fig, ax = figure(42, 39)
ax.scatter(m["footfall_pct"], m["demand_pct"], s=2.2, color=BLUE, alpha=0.45, linewidth=0)
slope, intercept = np.polyfit(m["footfall_pct"], m["demand_pct"], 1)
x = np.array([-60.0, 60.0])
ax.plot(x, intercept + slope * x, color=ORANGE, linewidth=1.1)
ax.set_xlim(-60, 60)
ax.set_ylim(-20, 20)
ax.set_xticks([-60, -30, 0, 30, 60])
ax.set_yticks([-20, -10, 0, 10, 20])
ax.set_xlabel("Visitors against normal (%)")
ax.set_ylabel("Demand against normal (%)")
ax.grid(axis="y")
save(fig, "footfall", data=m)
inside = m["footfall_pct"].between(-60, 60) & m["demand_pct"].between(-20, 20)
print(len(m), "site-days; slope", round(slope, 4), "; outside the axes", int((~inside).sum()))

"""Sites above their connection's capacity with heat pumps under the weather of each year since
1960 (31 sites with whole-site meters and gas heating), the reference year's demand re-weathered
day for day.

analysis/design_winter.py (years.csv).
"""

import pandas as pd
from _common import DESIGN, OUT

from sense_energy.analysis.design_winter import RECENT, REF_YEAR
from sense_energy.visualization import style

y = pd.read_csv(DESIGN / "years.csv")
last = int(y["weather_year"].max())
y["role"] = "earlier winters"
y.loc[y["weather_year"] > last - RECENT, "role"] = "last 30 winters"
y.loc[y["weather_year"] == REF_YEAR, "role"] = "reference year"
fig, ax = style.figure(style.WIDE_SINGLE)
for role, color in style.WEATHER_YEAR_COLORS.items():
    d = y[y["role"] == role]
    ax.bar(
        d["weather_year"],
        d["hp_sites_above_capacity"],
        color=color,
        width=0.8,
        label=role,
        zorder=3,
    )
ax.set_ylim(0, int(y["hp_sites"].max()))
ax.set_xlabel("Weather year (April to March, by its first year)")
ax.set_ylabel("Sites above capacity with heat pumps (of 31)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, bottom=0.3)
style.save_figure(fig, OUT / "design_winter_sites", data=y)

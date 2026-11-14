"""Finding 1, chart: panel sites above their connection capacity with heat pumps, in the
weather of each year since 1960."""

import pandas as pd
from _deck import BLUE, MID_GREY, REPORTS, figure, save

from sense_energy.analysis.design_winter import RECENT

y = pd.read_csv(REPORTS / "design_winter" / "years.csv")
last = int(y["weather_year"].max())
recent = y["weather_year"] > last - RECENT
fig, ax = figure(42, 39)
ax.bar(
    y.loc[~recent, "weather_year"],
    y.loc[~recent, "hp_sites_above_capacity"],
    width=0.8,
    color=MID_GREY,
    label="1960 to 1995",
)
ax.bar(
    y.loc[recent, "weather_year"],
    y.loc[recent, "hp_sites_above_capacity"],
    width=0.8,
    color=BLUE,
    label="last 30 winters",
)
ax.set_ylim(0, int(y["hp_sites"].max()))
ax.set_yticks([0, 10, 20, 31])
ax.set_xticks([1960, 1980, 2000, 2020])
ax.set_xlabel("Weather year")
ax.set_ylabel("Sites above capacity (of 31)")
ax.grid(axis="y")
ax.legend(loc="upper right", ncol=1, labelspacing=0.25, handlelength=1.0)
save(fig, "headroom_winters", data=y[["weather_year", "hp_sites_above_capacity", "hp_sites"]])

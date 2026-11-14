"""Finding 4, map: English sites at postcodes with solar funds from Great British Energy, by
what the 2025/26 estates return reports."""

import pandas as pd
from _deck import LIGHT_BLUE, ORANGE, REPORTS, england_map, figure, legend_below, save

from sense_energy.config import INTERIM_DIR

r = pd.read_csv(REPORTS / "solar_programme" / "returns_check.csv")
g = pd.read_parquet(
    INTERIM_DIR / "eric_2025_26_site.parquet", columns=["site_code", "latitude", "longitude"]
)
r = r[r["funded"]].merge(g, on="site_code").dropna(subset=["latitude"])
r["shows"] = r["new"] | r["grew"]
fig, ax = figure(48, 56)
england_map(ax)
for flag, label, colour, size, z in (
    (False, "no change yet", LIGHT_BLUE, 7, 3),
    (True, "new or more solar", ORANGE, 13, 4),
):
    d = r[r["shows"] == flag]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=size,
        color=colour,
        edgecolor="white",
        linewidth=0.25,
        zorder=z,
        label=label,
    )
legend_below(fig, ax, ncol=2, reverse=True, handletextpad=0.1)
save(fig, "solar_programme_map", data=r[["site_code", "latitude", "longitude", "shows"]])
print(len(r), "sites,", int(r["shows"].sum()), "with new or more solar")

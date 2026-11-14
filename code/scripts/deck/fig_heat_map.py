"""Finding 2, map: the 83 sites without CHP by the sign of their response to hot days."""

import pandas as pd
from _deck import BLUE, MID_GREY, ORANGE, REPORTS, england_map, figure, legend_below, save

from sense_energy.config import GEO_DIR

f = pd.read_csv(REPORTS / "heat_drivers" / "site_features.csv")
g = pd.read_parquet(GEO_DIR / "sites_geo.parquet")[["site_code", "latitude", "longitude"]]
f = f.merge(g.drop_duplicates("site_code"), on="site_code").dropna(subset=["latitude"])
groups = {
    "neither": ("no clear change", MID_GREY),
    "falls with heat": ("uses less", BLUE),
    "rises with heat": ("uses more", ORANGE),
}
fig, ax = figure(48, 56)
england_map(ax)
for key, (label, colour) in groups.items():
    d = f[f["sign"] == key]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=13,
        color=colour,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
        label=label,
    )
legend_below(fig, ax, ncol=3, reverse=True, handletextpad=0.1)
save(fig, "heat_map", data=f[["site_code", "latitude", "longitude", "heat", "sign"]])
print(f["sign"].value_counts().to_dict())

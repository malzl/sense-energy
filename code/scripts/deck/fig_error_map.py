"""Finding 2, map: the 91 forecast sites, marker area by the forecast error of the site in kWh
(best model, 69 test days); sites with on-site generation in orange."""

import numpy as np
import pandas as pd
from _deck import BLUE, ORANGE, REPORTS, england_map, figure, legend_below, save

s = pd.read_csv(REPORTS / "forecast_anatomy" / "error_by_site.csv").dropna(subset=["latitude"])
s["plant"] = s["group"].eq("on-site CHP")
s = s.sort_values("error_mwh", ascending=False)
s["size"] = 1.5 + 330.0 * s["error_mwh"] / s["error_mwh"].max()
fig, ax = figure(48, 56)
england_map(ax)
for plant, colour, label in (
    (False, BLUE, "no on-site generation"),
    (True, ORANGE, "on-site generation"),
):
    d = s[s["plant"] == plant]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=d["size"],
        color=colour,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
        zorder=4 if plant else 3,
        label=label,
    )
leg = legend_below(fig, ax, ncol=2, handletextpad=0.1)
for h in leg.legend_handles:
    h.set_sizes([14])
save(fig, "error_map", data=s[["site_code", "latitude", "longitude", "error_mwh", "plant"]])
print(len(s), "sites;", round(s.loc[s["plant"], "error_mwh"].sum() / s["error_mwh"].sum(), 3))
print(np.round(s["error_share_pct"].head(3).sum(), 1), "% of the error at three sites")

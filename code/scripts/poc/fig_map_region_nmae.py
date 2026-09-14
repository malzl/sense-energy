"""Direct day-ahead nMAE of Chronos-2 + IFS ENS for each NHS commissioning region's total demand
(6 regions with selected sites), as a choropleth with the sites overlaid. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, RESULTS
from _maps import COAST, LAND, base, colorbar, nhs_regions, site_points

from sense_energy.visualization import style

MODEL = "chronos2_ifs"

per = pd.read_csv(RESULTS.with_name(RESULTS.name + "_region") / "scores_per_site.csv")
per = per[per["model"] == MODEL].rename(columns={"site_code": "region_id"})
regions = nhs_regions().merge(
    per[["region_id", "nmae", "crps_skill_vs_naive", "n"]], on="region_id", how="left"
)

fig, ax = style.figure(style.SQUARE)
base(ax)
regions.plot(
    ax=ax, facecolor=LAND, edgecolor=COAST, linewidth=0.5, zorder=2
)  # regions without sites
with_data = regions[regions["nmae"].notna()]
coll = with_data.plot(
    ax=ax,
    column=with_data["nmae"] * 100,
    cmap="viridis",
    vmin=float(np.floor(with_data["nmae"].min() * 100)),
    vmax=float(np.ceil(with_data["nmae"].max() * 100)),
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
selected = set(pd.read_csv(RESULTS / "scores_per_site.csv")["site_code"])  # the PoC's 91 sites
pts = site_points()
pts = pts[pts["site_code"].isin(selected)]
ax.scatter(
    pts["longitude"], pts["latitude"], s=4, color="#000000", alpha=0.6, linewidth=0, zorder=4
)
mappable = ax.collections[-2]
colorbar(fig, mappable, "MAE / mean demand (%)")
style.save_figure(
    fig,
    OUT / "map_region_nmae",
    data=pd.DataFrame(regions.drop(columns="geometry"))[
        ["region_id", "nmae", "crps_skill_vs_naive", "n"]
    ],
)

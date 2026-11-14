"""Title slide: the 134 panel sites on Oxford blue, marker area by mean demand; sites with
on-site CHP in gold."""

import numpy as np
import pandas as pd
from _deck import GOLD, OXFORD_BLUE, england_map, figure, save

from sense_energy.analysis import weather_response as wr
from sense_energy.config import GEO_DIR, PROCESSED_DIR

w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
g = pd.read_parquet(GEO_DIR / "sites_geo.parquet").drop_duplicates("site_code")
g = g[g["site_code"].isin(w.columns)].dropna(subset=["latitude"]).copy()
g["mean_kw"] = g["site_code"].map(w.mean())
g["chp"] = g["site_code"].map(wr.site_groups()).eq("on-site CHP")
g = g.sort_values("mean_kw", ascending=False)
g["size"] = 2.0 + 0.9 * np.sqrt(g["mean_kw"].clip(lower=1))

fig, ax = figure(56, 72)
england_map(ax, face="#0D3566", edge="#2B5A94", lw=0.3)
for chp, colour, alpha in ((False, "white", 0.85), (True, GOLD, 0.95)):
    d = g[g["chp"] == chp]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=d["size"],
        color=colour,
        alpha=alpha,
        edgecolor=OXFORD_BLUE,
        linewidth=0.25,
        zorder=3,
    )
fig.get_layout_engine().set(w_pad=0, h_pad=0)
save(
    fig,
    "cover_sites",
    data=g[["site_code", "latitude", "longitude", "mean_kw", "chp"]],
    facecolor=OXFORD_BLUE,
)
print(len(g), "sites,", int(g["chp"].sum()), "with CHP")

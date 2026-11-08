"""Each site's shrunk cold or heat slope on the map (argument: hdd14 or cdd20). One plot (map).

Empirical-Bayes posterior slopes from analysis/weather_heterogeneity.py; diverging colours
centred on zero, limits at the 95th percentile of the absolute posterior.
"""

import sys

import numpy as np
import pandas as pd
from _common import OUT, england_axes, heterogeneity, map_colorbar

from sense_energy.config import GEO_DIR
from sense_energy.visualization import style

DOSE = sys.argv[1] if len(sys.argv) > 1 else "cdd20"
g = pd.read_parquet(GEO_DIR / "sites_geo.parquet")[["site_code", "latitude", "longitude"]]
s = heterogeneity("site_slopes")
s = s[s["dose"] == DOSE].merge(g, on="site_code").dropna(subset=["latitude", "longitude"])
lim = float(np.percentile(np.abs(s["posterior_pct"]), 95))

fig, ax = england_axes()
sc = ax.scatter(
    s["longitude"],
    s["latitude"],
    c=s["posterior_pct"],
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim,
    s=18,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    zorder=3,
)
map_colorbar(
    fig,
    ax,
    sc,
    f"{style.DOSE_NAMES[DOSE][:1].upper() + style.DOSE_NAMES[DOSE][1:]} (% per °C), shrunk",
    extend="both",
)
print(
    f"{DOSE}: {len(s)} sites, posterior q10/50/90 {s['posterior_pct'].quantile([0.1, 0.5, 0.9]).round(2).tolist()}"
)
style.save_figure(
    fig,
    OUT / f"map_site_slopes_{DOSE}",
    data=s[["site_code", "latitude", "longitude", "slope_pct", "se_pct", "posterior_pct"]],
)

"""How much too cool ERA5 reads the hottest days at each site: mean difference between ERA5's and
the nearest station's daily maximum on days the station reached 28 degC. One plot (map).

Diverging colours centred on zero with symmetric limits; negative means ERA5 is cooler than the
station. Sites with fewer than 10 such days are left out.
"""

import pandas as pd
from _common import OUT, england_axes, map_colorbar, site_points

from sense_energy.config import INTERIM_DIR
from sense_energy.visualization import style

HOT_C, MIN_DAYS = 28.0, 10
e = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_max"])
s = pd.read_parquet(
    INTERIM_DIR / "weather_daily_isd.parquet", columns=["site_code", "date", "t_max"]
)
m = e.merge(s, on=["site_code", "date"], suffixes=("_era5", "_station"))
hot = m[m["t_max_station"] >= HOT_C]
bias = (
    hot.assign(d=hot["t_max_era5"] - hot["t_max_station"])
    .groupby("site_code")["d"]
    .agg(bias_c="mean", days="size")
    .reset_index()
)
bias = bias[bias["days"] >= MIN_DAYS]
pts = site_points().merge(bias, on="site_code")
lim = 4.0

fig, ax = england_axes()
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    c=pts["bias_c"],
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim,
    s=18,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    zorder=3,
)
map_colorbar(
    fig, ax, sc, f"ERA5 − station daily maximum on ≥ {HOT_C:g} °C days (°C)", extend="both"
)
print(
    f"{len(pts)} sites; median bias {pts['bias_c'].median():+.2f} °C, 10th–90th {pts['bias_c'].quantile(0.1):+.2f} to {pts['bias_c'].quantile(0.9):+.2f}"
)
style.save_figure(
    fig,
    OUT / "map_hot_day_bias",
    data=pts[["site_code", "site_name", "latitude", "longitude", "bias_c", "days"]],
)

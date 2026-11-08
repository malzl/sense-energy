"""The identifying variation of the heat effect, on the ground: each site's hottest daily
maximum of summer 2025 (June-August), from the station thermometer. One plot (map).

The cross-site spread on the same days is what the date-effect designs use; the map shows how
much of it there is and where. Sequential colours.
"""

import pandas as pd
from _common import OUT, england_axes, map_colorbar, site_points

from sense_energy.config import INTERIM_DIR
from sense_energy.visualization import style

s = pd.read_parquet(
    INTERIM_DIR / "weather_daily_isd.parquet", columns=["site_code", "date", "t_max"]
)
summer = s[(s["date"] >= "2025-06-01") & (s["date"] < "2025-09-01")]
peak = summer.groupby("site_code")["t_max"].max().rename("t_max_2025").reset_index()
days30 = (
    (summer["t_max"] >= 30).groupby(summer["site_code"]).sum().rename("days_ge_30").reset_index()
)
pts = site_points().merge(peak, on="site_code").merge(days30, on="site_code")

fig, ax = england_axes()
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    c=pts["t_max_2025"],
    cmap="magma",
    vmin=26,
    vmax=35,
    s=18,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    zorder=3,
)
map_colorbar(fig, ax, sc, "Hottest daily maximum, June–August 2025 (°C)", extend="both")
print(
    f"{len(pts)} sites; hottest {pts['t_max_2025'].max():.1f} °C, coolest site peak {pts['t_max_2025'].min():.1f} °C; "
    f"days at 30 °C+: median {pts['days_ge_30'].median():.0f}, max {pts['days_ge_30'].max()}"
)
style.save_figure(
    fig,
    OUT / "map_heatwave_exposure",
    data=pts[["site_code", "site_name", "latitude", "longitude", "t_max_2025", "days_ge_30"]],
)

"""Sites and the Met Office stations that serve as their thermometer: each site joined to its
nearest station by a line, stations as crosses. One plot (map).

Station candidates from analysis/thermometers.py (NOAA ISD history); the line length is the
distance that the station reading has to travel to the site.
"""

import pandas as pd
from _common import OUT, england_axes, site_points

from sense_energy.config import INTERIM_DIR
from sense_energy.data import isd
from sense_energy.visualization import style

cand = pd.read_parquet(INTERIM_DIR / "isd_site_stations.parquet")
nearest = cand[cand["rank"] == 0]
stations = isd.station_table().set_index("station_id")
pts = site_points().merge(nearest[["site_code", "station_id", "distance_km"]], on="site_code")
pts["st_lat"] = pts["station_id"].map(stations["lat"])
pts["st_lon"] = pts["station_id"].map(stations["lon"])
used = stations.loc[sorted(set(pts["station_id"]))]

fig, ax = england_axes()
for r in pts.itertuples():
    ax.plot(
        [r.longitude, r.st_lon],
        [r.latitude, r.st_lat],
        color=style.THERMOMETER_COLORS["stations"],
        linewidth=0.5,
        alpha=0.6,
        zorder=2,
    )
ax.scatter(
    pts["longitude"],
    pts["latitude"],
    s=9,
    color=style.THERMOMETER_COLORS["ERA5"],
    edgecolor="white",
    linewidth=0.3,
    zorder=3,
    label=f"NHS sites (n = {len(pts)})",
)
ax.scatter(
    used["lon"],
    used["lat"],
    s=22,
    marker="x",
    color=style.THERMOMETER_COLORS["stations"],
    linewidth=0.9,
    zorder=4,
    label=f"Met Office stations used (n = {len(used)})",
)
style.add_bottom_legend(ax, lowercase=False, ncol=1, anchor_y=-0.02, bottom=0.12)
print(
    f"median site-station distance {pts['distance_km'].median():.1f} km, max {pts['distance_km'].max():.1f} km"
)
style.save_figure(
    fig,
    OUT / "map_thermometer_network",
    data=pts[
        [
            "site_code",
            "site_name",
            "latitude",
            "longitude",
            "station_id",
            "st_lat",
            "st_lon",
            "distance_km",
        ]
    ],
)

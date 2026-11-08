"""Where ERA5's site thermometer is good and where it is not: RMSE of each site's day-to-day
temperature anomaly (within site x month) against its nearest Met Office station. One plot (map).

From analysis/thermometers.py (agreement.csv). Sequential colours, fixed limits so the map is
comparable with the ERA5-Land one (same script, THERMOMETER argument).
"""

import sys

from _common import OUT, england_axes, map_colorbar, site_points, thermometers

from sense_energy.visualization import style

THERMOMETER = sys.argv[1] if len(sys.argv) > 1 else "ERA5"
a = thermometers("agreement")
a = a[a["thermometer"] == THERMOMETER]
pts = site_points().merge(a[["site_code", "anomaly_rmse_c", "station_km"]], on="site_code")

fig, ax = england_axes()
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    c=pts["anomaly_rmse_c"],
    cmap="viridis",
    vmin=0.3,
    vmax=1.0,
    s=18,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    zorder=3,
)
map_colorbar(
    fig,
    ax,
    sc,
    f"{style.THERMOMETER_NAMES[THERMOMETER]}: anomaly RMSE vs station (°C)",
    extend="max",
)
print(
    f"{THERMOMETER}: median {pts['anomaly_rmse_c'].median():.2f} °C, worst {pts['anomaly_rmse_c'].max():.2f} °C"
)
style.save_figure(
    fig,
    OUT / f"map_thermometer_error_{THERMOMETER.lower().replace('-', '_')}",
    data=pts[["site_code", "site_name", "latitude", "longitude", "anomaly_rmse_c", "station_km"]],
)

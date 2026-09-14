"""Per-site CRPS skill of Chronos-2 + IFS ENS against seasonal naive on the England map,
marker area proportional to mean demand. One plot."""

from _common import OUT, per_site
from _maps import base, colorbar, site_points

from sense_energy.visualization import style

MODEL = "chronos2_ifs"

per = per_site()
per = per[per["model"] == MODEL]
pts = site_points().merge(per[["site_code", "crps_skill_vs_naive", "y_mean"]], on="site_code")
pts["mean_kw"] = pts["y_mean"] * 2
area = 8 + 160 * pts["mean_kw"] / pts["mean_kw"].max()

fig, ax = style.figure(style.SQUARE)
base(ax)
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    s=area,
    c=pts["crps_skill_vs_naive"],
    cmap="viridis",
    vmin=0,
    vmax=0.75,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    alpha=0.9,
    zorder=3,
)
colorbar(fig, sc, "CRPS skill vs seasonal naive")
style.save_figure(
    fig,
    OUT / "map_site_skill",
    data=pts[["site_code", "site_name", "latitude", "longitude", "mean_kw", "crps_skill_vs_naive"]],
)

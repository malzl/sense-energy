"""Per-site change in CRPS skill from the known-ahead covariates with forecast weather
(Chronos-2 + IFS ENS minus Chronos-2) on the England map. One plot."""

from _common import OUT, per_site
from _maps import base, colorbar, site_points

from sense_energy.visualization import style

BASE, VARIANT = "chronos2", "chronos2_ifs"

per = per_site().set_index(["model", "site_code"])["crps_skill_vs_naive"]
gain = (per.loc[VARIANT] - per.loc[BASE]).rename("gain_pp") * 100
pts = site_points().merge(gain.reset_index(), on="site_code")
lim = float(min(15, max(abs(pts["gain_pp"].quantile(0.02)), abs(pts["gain_pp"].quantile(0.98)))))

fig, ax = style.figure(style.SQUARE)
base(ax)
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    s=22,
    c=pts["gain_pp"],
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    alpha=0.9,
    zorder=3,
)
colorbar(fig, sc, "Change in CRPS skill (percentage points)")
style.save_figure(
    fig,
    OUT / "map_covariate_gain",
    data=pts[["site_code", "site_name", "latitude", "longitude", "gain_pp"]],
)

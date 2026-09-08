"""Sites with electricity data, coloured by share of half-hours observed within their span. One plot."""

import numpy as np

from sense_energy.eda import common as C
from sense_energy.visualization import style

_, index = C.series_for("elec", "site", flavour="raw")
pts = C.site_points().merge(
    index[["series_id", "coverage_in_span"]], left_on="site_code", right_on="series_id"
)

fig, ax = style.figure(style.SQUARE)
C.england().plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
sc = ax.scatter(
    pts["longitude"],
    pts["latitude"],
    c=pts["coverage_in_span"],
    cmap="viridis",
    vmin=0.5,
    vmax=1.0,
    s=18,
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    zorder=3,
)
minx, miny, maxx, maxy = C.england().total_bounds
ax.set_xlim(minx - 0.2, maxx + 0.2)
ax.set_ylim(miny - 0.1, maxy + 0.1)
ax.set_aspect(1 / np.cos(np.deg2rad(53)))
ax.set_axis_off()
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend="min")
cb.set_label("Share of half-hours observed")
cb.ax.tick_params(
    labelsize=style.FONT_SIZE["tick"], width=style.TICK_WIDTH, length=style.TICK_LENGTH
)
cb.outline.set_linewidth(style.AXIS_LINEWIDTH)
style.save_figure(
    fig,
    C.OUT / "map_site_coverage",
    data=pts[["site_code", "site_name", "latitude", "longitude", "coverage_in_span"]],
)

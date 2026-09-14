"""Shared England base map for the PoC map figures (one plot per script)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.config import GEO_DIR
from sense_energy.eda import common as C
from sense_energy.visualization import style

LAND, COAST, BOUNDARY = "#F1F0EA", "#C3C2B7", "#9A9A9A"


def base(ax, regions: bool = False):
    eng = C.england()
    eng.plot(ax=ax, facecolor=LAND, edgecolor=COAST, linewidth=0.5, zorder=1)
    if regions:
        nhs_regions().boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.5, zorder=2)
    minx, miny, maxx, maxy = eng.total_bounds
    ax.set_xlim(minx - 0.2, maxx + 0.2)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect(1 / np.cos(np.deg2rad(53)))
    ax.set_axis_off()


def nhs_regions():
    import geopandas as gpd

    g = gpd.read_file(GEO_DIR / "nhs_regions_en.geojson")
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(epsg=4326)
    name_col = next(c for c in g.columns if c.upper().endswith("NM"))
    g["region_id"] = g[name_col].map(_region_id)
    return g


def _region_id(name: str) -> str:
    from sense_energy.experiments.hierarchy import region_id

    return region_id(str(name).replace(" Region", "").replace(" region", ""))


def site_points() -> pd.DataFrame:
    return C.site_points()


def colorbar(fig, mappable, label: str):
    cb = fig.colorbar(mappable, ax=fig.axes[0], fraction=0.035, pad=0.02, shrink=0.75)
    cb.set_label(label, fontsize=style.FONT_SIZE["label"])
    cb.ax.tick_params(
        labelsize=style.FONT_SIZE["tick"], width=style.TICK_WIDTH, length=style.TICK_LENGTH
    )
    cb.outline.set_linewidth(style.AXIS_LINEWIDTH)
    return cb

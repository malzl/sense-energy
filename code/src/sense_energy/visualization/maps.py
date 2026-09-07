"""Maps of the NHS site estate.

Colour carries site identity (three collapsed categories) and area carries
magnitude; the two encode different variables rather than doubling up on one.
Three categorical slots is the documented cap for all-pairs forms such as maps
and scatters, so the eight source ``organisation_type`` values are collapsed
rather than given eight hues.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import FIGURES_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Reference palette, unmodified. Slots 1-3 validate on the all-pairs pairlist.
SERIES = {"Acute": "#2a78d6", "Mental health & community": "#eb6834", "Ambulance": "#1baf7a"}
SURFACE = "#fcfcfb"
LAND = "#f1f0ea"
BOUNDARY = "#e1e0d9"
COASTLINE = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

#: Collapse the eight source organisation types onto the three categorical slots.
CATEGORY_MAP = {
    "ACUTE - TEACHING": "Acute",
    "ACUTE - LARGE": "Acute",
    "ACUTE - MEDIUM": "Acute",
    "ACUTE - SMALL": "Acute",
    "ACUTE - SPECIALIST": "Acute",
    "MENTAL HEALTH AND LEARNING DISABILITY": "Mental health & community",
    "COMMUNITY": "Mental health & community",
    "AMBULANCE": "Ambulance",
}


def categorise(organisation_type: pd.Series) -> pd.Series:
    return organisation_type.map(CATEGORY_MAP).fillna("Mental health & community")


def _area_scale(values: np.ndarray, max_points: float = 900.0) -> np.ndarray:
    """Marker *area* proportional to value, so a dot twice the area means twice the load."""
    top = np.nanmax(values)
    if not np.isfinite(top) or top <= 0:
        return np.full_like(values, 30.0)
    return 30.0 + (values / top) * max_points


def plot_site_map(
    sites_geo,
    site_demand: pd.DataFrame,
    countries=None,
    regions=None,
    label_top: int = 6,
    ax=None,
):
    """Plot located sites over an England basemap, sized by mean demand.

    ``site_demand`` needs ``site_code`` and ``mean_kwh``.
    """
    gdf = sites_geo.merge(site_demand, on="site_code", how="inner")
    gdf = gdf[gdf["mean_kwh"].notna() & (gdf["mean_kwh"] > 0)].copy()
    gdf["category"] = categorise(gdf["organisation_type"])

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor(SURFACE)

    if countries is not None:
        # Every site is in England (this is NHS England data), so the basemap is
        # cropped to England - an empty Scotland would be half the frame.
        name_col = next(
            (
                c
                for c in countries.columns
                if c.upper().startswith("CTRY") and c.upper().endswith("NM")
            ),
            None,
        )
        base = countries
        if name_col is not None:
            england = countries[countries[name_col].astype(str).eq("England")]
            if len(england):
                base = england
        base.plot(ax=ax, facecolor=LAND, edgecolor=COASTLINE, linewidth=0.6, zorder=1)
        minx, miny, maxx, maxy = base.total_bounds
        pad_x, pad_y = (maxx - minx) * 0.04, (maxy - miny) * 0.03
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    if regions is not None:
        regions.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.7, zorder=2)

    for category, colour in SERIES.items():
        subset = gdf[gdf["category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=_area_scale(subset["mean_kwh"].to_numpy()),
            facecolor=colour,
            edgecolor=SURFACE,  # 2px surface ring so overlapping sites stay separable
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
            label=category,
        )

    # Direct labels on the largest sites - also the relief for the low-contrast slot.
    # Labels flip to the left of their mark on the eastern half so none runs off frame.
    x_mid = gdf.geometry.x.median()
    for _, row in gdf.nlargest(label_top, "mean_kwh").iterrows():
        east = row.geometry.x > x_mid
        ax.annotate(
            row["site_name"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(-9 if east else 9, 6),
            textcoords="offset points",
            ha="right" if east else "left",
            fontsize=8,
            color=INK,
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)],
        )

    ax.set_axis_off()
    ax.set_aspect("equal")

    colour_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=c,
            markeredgecolor=SURFACE,
            label=k,
        )
        for k, c in SERIES.items()
        if (gdf["category"] == k).any()
    ]
    legend = ax.legend(
        handles=colour_handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        title="Site type",
    )
    legend.get_title().set_color(INK_SECONDARY)
    legend.get_title().set_fontsize(9)
    ax.add_artist(legend)

    top = gdf["mean_kwh"].max()
    size_values = [v for v in (round(top / 10, -1), round(top / 2, -1), round(top, -1)) if v > 0]
    # The size legend scales against the same maximum as the plotted marks.
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=np.sqrt(30.0 + (v / top) * 900.0),
            markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE,
            alpha=0.6,
            label=f"{v:,.0f}",
        )
        for v in size_values
    ]
    size_legend = ax.legend(
        handles=size_handles,
        loc="lower left",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        title="Mean electricity demand\n(kWh per half hour)",
        labelspacing=1.4,
        borderpad=1.0,
        handletextpad=1.4,
    )
    size_legend.get_title().set_color(INK_SECONDARY)
    size_legend.get_title().set_fontsize(9)

    logger.info("Mapped %d sites", len(gdf))
    return ax, gdf


def save_map(fig, name: str = "site_map", dpi: int = 200) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    logger.info("Wrote %s", FIGURES_DIR / f"{name}.png")

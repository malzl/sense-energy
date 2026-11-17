"""Slide 2: one map for each retrofit question. Each site is a circle; its area shows how much
the electricity of the site follows cold (heating), heat (cooling) and daylight (lighting).
Grey rings: no clear reaction. First estimates for the sites with complete meter data and no
on-site generation (analysis/heat_drivers.py: site slopes for cold below 14 degC, heat above
20 degC and daily sunshine, date effects). An illustration of the approach, not a result."""

import numpy as np
import pandas as pd
from _deck import (
    ASH,
    BLUE,
    CHARCOAL,
    MID_GREY,
    ORANGE,
    REPORTS,
    SHELL,
    TEXT_GOLD,
    TEXT_ORANGE,
    YELLOW,
    england_map,
    figure,
    save,
)
from matplotlib.lines import Line2D

from sense_energy.eda import common as C

s = pd.read_csv(REPORTS / "heat_drivers" / "site_features.csv")
s = s.merge(C.site_points()[["site_code", "latitude", "longitude"]], on="site_code")
s = s.dropna(subset=["latitude", "longitude"])
panels = [
    (
        "Heating",
        "more demand\nwhen it is cold",
        100 * s["cold"],
        s["cold"] / s["cold_se"],
        BLUE,
        BLUE,
    ),
    (
        "Cooling",
        "more demand\nwhen it is hot",
        100 * s["heat"],
        s["heat"] / s["heat_se"],
        ORANGE,
        TEXT_ORANGE,
    ),
    (
        "Lighting",
        "less demand\nin daylight",
        -100 * s["sun"],
        -s["sun"] / s["sun_se"],
        YELLOW,
        TEXT_GOLD,
    ),
]
# each column: the name of the retrofit on the left, its map on the right
fig, axes = figure(145.6, 44.5, ncols=6, gridspec_kw={"width_ratios": [0.5, 1] * 3})
out = s[["site_code", "latitude", "longitude"]].copy()
for k, (name, sub, value, z, colour, ink) in enumerate(panels):
    label, ax = axes[2 * k], axes[2 * k + 1]
    label.set_axis_off()
    label.text(0.0, 0.86, name, ha="left", va="top", fontsize=9.2, color=ink, fontweight="bold")
    label.text(0.0, 0.72, sub, ha="left", va="top", fontsize=6.4, color=CHARCOAL, linespacing=1.25)
    england_map(ax, face=SHELL)
    clear = (z > 1.96) & (value > 0)
    scale = np.nanpercentile(value[clear], 90)
    size = 4.0 + 46.0 * np.clip(value / scale, 0, 1.6)
    weak = s[~clear]
    ax.scatter(
        weak["longitude"],
        weak["latitude"],
        s=3.2,
        facecolor="none",
        edgecolor=MID_GREY,
        linewidth=0.35,
        zorder=3,
    )
    d = s[clear].iloc[np.argsort(-size[clear].to_numpy())]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=size[clear].loc[d.index],
        color=colour,
        alpha=0.78,
        edgecolor="white" if colour != YELLOW else TEXT_GOLD,
        linewidth=0.3,
        zorder=4,
    )
    out[name.lower()] = value.to_numpy()
    out[name.lower() + "_clear"] = clear.to_numpy()
    print(name, "clear reaction at", int(clear.sum()), "of", len(s), "sites")
handles = [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=1.9,
        markerfacecolor="none",
        markeredgecolor=MID_GREY,
        markeredgewidth=0.4,
        label="no clear change",
    ),
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=2.6,
        color=ASH,
        markeredgewidth=0,
        label="small change in demand",
    ),
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=6.4,
        color=ASH,
        markeredgewidth=0,
        label="large change in demand",
    ),
]
fig.legend(
    handles=handles,
    loc="outside lower center",
    ncol=3,
    fontsize=6.2,
    columnspacing=2.2,
    handletextpad=0.5,
)
save(fig, "retrofit_maps", data=out)

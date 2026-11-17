"""Slide 2: one map for each retrofit question. Each site is a circle; its area shows how much
the electricity of the site follows cold (heating), heat (cooling) and daylight (lighting).
Grey rings: no clear reaction. First estimates for the sites with complete meter data and no
on-site generation (analysis/heat_drivers.py: site slopes for cold below 14 degC, heat above
20 degC and daily sunshine, date effects). An illustration of the approach, not a result."""

import numpy as np
import pandas as pd
from _deck import (
    BLUE,
    CHARCOAL,
    MID_GREY,
    ORANGE,
    REPORTS,
    SHELL,
    YELLOW,
    england_map,
    figure,
    save,
)

from sense_energy.eda import common as C

s = pd.read_csv(REPORTS / "heat_drivers" / "site_features.csv")
s = s.merge(C.site_points()[["site_code", "latitude", "longitude"]], on="site_code")
s = s.dropna(subset=["latitude", "longitude"])
panels = [
    ("Heating", "more demand when it is cold", 100 * s["cold"], s["cold"] / s["cold_se"], BLUE),
    ("Cooling", "more demand when it is hot", 100 * s["heat"], s["heat"] / s["heat_se"], ORANGE),
    ("Lighting", "less demand in daylight", -100 * s["sun"], -s["sun"] / s["sun_se"], YELLOW),
]
fig, axes = figure(91, 40, ncols=3)
out = s[["site_code", "latitude", "longitude"]].copy()
for ax, (name, sub, value, z, colour) in zip(axes, panels, strict=True):
    england_map(ax, face=SHELL)
    clear = (z > 1.96) & (value > 0)
    scale = np.nanpercentile(value[clear], 90)
    size = 3.5 + 40.0 * np.clip(value / scale, 0, 1.6)
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
        edgecolor="white",
        linewidth=0.3,
        zorder=4,
    )
    ax.text(
        0.5,
        1.13,
        name,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=colour,
        fontweight="bold",
    )
    ax.text(
        0.5,
        1.03,
        sub,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=CHARCOAL,
    )
    ax.set_title(" ", fontsize=13, pad=2)
    out[name.lower()] = value.to_numpy()
    out[name.lower() + "_clear"] = clear.to_numpy()
    print(name, "clear reaction at", int(clear.sum()), "of", len(s), "sites")
save(fig, "retrofit_maps", data=out)

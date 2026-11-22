"""Shared paths for the figures of the forecast range (one plot per script)."""

from __future__ import annotations

import pandas as pd

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

RG = PROJECT_ROOT / "code" / "reports" / "forecast_range"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "forecast_range"
MODEL = "chronos2_ifs"


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(RG / f"{name}.csv")


def misses_by_hour(column: str, label: str, name: str) -> None:
    """Share of the half-hours beyond one bound of the model's range, by hour and season."""
    s = table("season_hour")
    s = s[s["method"] == "model"]
    fig, ax = style.figure(style.SINGLE_COLUMN)
    style.reference_line(ax, 10.0)
    for season in style.SEASON_ORDER:
        d = s[s["season"] == season].sort_values("hour")
        ax.plot(
            d["hour"] + 0.5,
            d[column],
            color=style.SEASON_COLORS[season],
            linestyle=style.SEASON_LINESTYLES[season],
            linewidth=style.LINEWIDTH["primary"],
            label=season,
        )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("Hour of the day")
    ax.set_ylim(0, None)
    ax.set_ylabel(label)
    style.style_axis(ax, grid="y")
    style.add_bottom_legend(ax, ncol=4, anchor_y=-0.26, bottom=0.3)
    style.save_figure(fig, OUT / name, data=s)

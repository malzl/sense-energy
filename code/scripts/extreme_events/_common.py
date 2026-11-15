"""Shared paths and helpers for the extreme-event figures (one plot per script)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

EE = PROJECT_ROOT / "code" / "reports" / "extreme_events"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "extreme_events"
PLAIN = "no on-site generation"


def table(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(EE / f"{name}.csv", **kw)


def kind_arg() -> tuple[str, str]:
    """Event type from the command line (``heatwave`` or ``cold``) and its file tag."""
    arg = sys.argv[1] if len(sys.argv) > 1 else "heatwave"
    return ("heatwave", "heatwave") if arg.startswith("heat") else ("cold spell", "cold_spell")


def england_axes():
    """A square map axis with the England outline, aspect-corrected, axis off."""
    from sense_energy.eda import common as C

    fig, ax = style.figure(style.SQUARE)
    eng = C.england()
    eng.plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
    minx, miny, maxx, maxy = eng.total_bounds
    ax.set_xlim(minx - 0.2, maxx + 0.2)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect(1 / np.cos(np.deg2rad(53)))
    ax.set_axis_off()
    return fig, ax


def band_line(ax, d: pd.DataFrame, x: str, color: str, label: str, marker: str | None = None):
    ax.fill_between(d[x], d["lo"], d["hi"], color=color, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(
        d[x],
        d["reaction_pct"],
        color=color,
        linewidth=style.LINEWIDTH["primary"],
        marker=marker,
        markersize=style.MARKER_SIZE - 1,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=label,
    )

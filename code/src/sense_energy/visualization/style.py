"""Shared figure style - the master figure specification, implemented once.

Every plotting script imports this and never defines its own labels, colours
or sizes. Deviations are allowed only for scientific correctness and must be
recorded in the figure script that makes them.

Summary of the specification this encodes (standing instruction, 2026-08-04,
amended 2026-08-25):

* one independent plot per script; no titles, panel labels, annotations,
  text boxes or decoration - captions and panel letters are added in LaTeX
* exports: ``<base>.pdf`` (vector, transparent, fonts embedded) and
  ``<base>.png`` (white, 600 dpi), plus the plotted data as ``<base>_data.csv``
* sans-serif, never bold; labels 8 pt, ticks 7 pt, legend 7 pt
* axes 0.7 pt, ticks 0.6 pt / 3 pt out, top and right spines removed
* legend outside, centred below the axes, borderless, fixed method order
* colours by conceptual role from an Okabe-Ito palette
* fixed orders for horizons and methods - never by result
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Dimensions (inches). No arbitrary sizes.
# --------------------------------------------------------------------------- #

SINGLE_COLUMN = (3.5, 2.8)  # default
WIDE_SINGLE = (5.2, 3.2)  # many horizons / dates / model names
DOUBLE_COLUMN = (7.2, 3.8)  # only when smaller widths are illegible
SQUARE = (3.5, 3.5)
WIDE_SHORT = (7.2, 2.6)  # distribution / strip / trajectory-construction panels (amendment)

# --------------------------------------------------------------------------- #
# Typography and marks
# --------------------------------------------------------------------------- #

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE = {"label": 8, "tick": 7, "legend": 7, "annotation": 7}
FONT_SIZE_WIDE_SHORT_TICK = 8  # amendment: tick font for the wide/short format

LINEWIDTH = {"primary": 1.5, "secondary": 1.0, "reference": 0.8}
MARKER_SIZE = 4
MARKER_EDGE = 0.5
BAND_ALPHA = 0.18
MAX_LINESTYLES = 4

AXIS_LINEWIDTH = 0.7
TICK_WIDTH = 0.6
TICK_LENGTH = 3

GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.15

REFERENCE_LINE = {"color": "#7A7A7A", "linewidth": 0.8, "linestyle": "--"}

# --------------------------------------------------------------------------- #
# Colours - by conceptual role, never by appearance order.
# --------------------------------------------------------------------------- #

COLORS = {  # Okabe-Ito
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#7A7A7A",
    "light_grey": "#B8B8B8",
}

ROLE_COLORS = {
    "reference": "#000000",
    "simple_baseline": "#7A7A7A",
    "classical_model": "#56B4E9",
    "foundation_model": "#0072B2",
    "specialist_model": "#009E73",
    "exact_bayes": "#E69F00",
    "pfn": "#D55E00",
    "zero_effect": "#B8B8B8",
}

SEQUENTIAL_CMAPS = ("viridis", "cividis", "magma")

#: Entity colours for exploratory figures, by conceptual role. Energy vectors
#: take two categorical slots; day types take the reference/baseline greys so
#: the two pairings never compete inside one figure.
ENERGY_COLORS = {"elec": "#0072B2", "gas": "#E69F00"}
ENERGY_LABELS = {"elec": "electricity", "gas": "gas"}
DAYTYPE_COLORS = {"weekday": "#000000", "weekend": "#7A7A7A"}
SPAGHETTI = "#B8B8B8"  # many background series; the highlighted statistic is black
DIVERGING_CMAPS = ("RdBu_r", "coolwarm")

# --------------------------------------------------------------------------- #
# Nomenclature (binding). Update here, nowhere else.
# --------------------------------------------------------------------------- #

DATASET_NAMES = {
    "nhs": "NHS half-hourly consumption",  # Energy Systems Catapult extract
    "era5": "ERA5",
    "aifs_ens": "AIFS-ENS",
    "agile": "Octopus Agile",
}

#: Display names for this project's candidate methods, in the fixed order used
#: in every figure. Keys match MODEL_REGISTRY in models/train.py.
METHOD_NAMES = {
    "seasonal_naive": "Seasonal naive",
    "profile_mean": "Profile mean",
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
}
METHOD_ORDER = list(METHOD_NAMES.values())

METHOD_ROLES = {
    "Seasonal naive": "simple_baseline",
    "Profile mean": "simple_baseline",
    "Ridge": "classical_model",
    "LightGBM": "specialist_model",
}

#: Generic lead times for half-hourly demand; fixed, never reordered by result.
HORIZON_ORDER_HOURS = [0.5, 1, 3, 6, 12, 24, 48, 168]

#: Axis-label vocabulary. One term per quantity, units included.
QUANTITY_LABELS = {
    "horizon": "Forecast horizon (hours)",
    "demand": "Demand (kWh per half hour)",
    "temperature": "Temperature (\u00b0C)",
    "mae": "MAE (kWh)",
    "rmse": "RMSE (kWh)",
    "mase": "MASE",
    "crps": "CRPS (kWh)",
    "crps_skill": "CRPS skill",
    "coverage": "Coverage (%, target 80)",
    "bias": "Bias (kWh)",
}


def method_color(method: str) -> str:
    """Colour for a method name via its conceptual role."""
    role = METHOD_ROLES.get(method)
    if role is None:
        raise KeyError(f"'{method}' has no role in style.py - add it there, not in the script")
    return ROLE_COLORS[role]


def ordered(methods: Sequence[str]) -> list[str]:
    """Return the given methods in the fixed dissertation order, never by performance."""
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    unknown = [m for m in methods if m not in order]
    if unknown:
        raise KeyError(f"Methods without a fixed order: {unknown}")
    return sorted(methods, key=order.__getitem__)


# --------------------------------------------------------------------------- #
# Matplotlib configuration
# --------------------------------------------------------------------------- #


def configure_matplotlib() -> None:
    """Apply the specification to rcParams. Call once at the top of every script."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "figure.titleweight": "normal",
            "mathtext.default": "regular",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE["label"],
            "axes.labelsize": FONT_SIZE["label"],
            "xtick.labelsize": FONT_SIZE["tick"],
            "ytick.labelsize": FONT_SIZE["tick"],
            "legend.fontsize": FONT_SIZE["legend"],
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.minor.width": TICK_WIDTH,
            "ytick.minor.width": TICK_WIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.alpha": GRID_ALPHA,
            "lines.linewidth": LINEWIDTH["primary"],
            "lines.markersize": MARKER_SIZE,
            "lines.markeredgewidth": MARKER_EDGE,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.2,
            "legend.handletextpad": 0.5,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "image.cmap": "viridis",
        }
    )


def figure(size: tuple[float, float] = SINGLE_COLUMN):
    """A configured (fig, ax) at one of the allowed sizes."""
    if size not in (SINGLE_COLUMN, WIDE_SINGLE, DOUBLE_COLUMN, SQUARE, WIDE_SHORT) and not (
        size[0] == 7.2 and 2.4 <= size[1] <= 2.8
    ):
        raise ValueError(f"{size} is not an allowed figure size")
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=size)
    style_axis(ax, wide_short=(size[0] == 7.2 and size[1] <= 2.8))
    return fig, ax


def style_axis(
    ax, grid: str | None = None, all_spines: bool = False, wide_short: bool = False
) -> None:
    """Spines, ticks and optional grid per the specification.

    ``grid`` is ``None`` (default), ``"y"`` (bar plots, score comparisons) or
    ``"both"``; only use it when it helps numerical comparison.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(all_spines)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    tick_size = FONT_SIZE_WIDE_SHORT_TICK if wide_short else FONT_SIZE["tick"]
    ax.tick_params(width=TICK_WIDTH, length=TICK_LENGTH, direction="out", labelsize=tick_size)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(
            True,
            axis=grid if grid != "both" else "both",
            which="major",
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
    else:
        ax.grid(False)
    ax.set_title("")


def add_bottom_legend(
    ax,
    ncol: int | None = None,
    anchor_y: float = -0.22,
    bottom: float = 0.30,
    fontsize: int | None = None,
    handles=None,
    labels=None,
):
    """Legend outside, centred below the axes, borderless, one row when possible.

    ``anchor_y`` -0.22 is the default; the wide/short amendment uses -0.30 or
    lower with a matching ``bottom`` margin. Entries are lowercased per the
    amendment.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    labels = [lab[:1].lower() + lab[1:] if lab and not lab[:2].isupper() else lab for lab in labels]
    ncol = ncol or len(handles)
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=fontsize or FONT_SIZE["legend"],
        title=None,
    )
    ax.figure.subplots_adjust(bottom=bottom)
    return legend


def reference_line(ax, value: float, axis: str = "y"):
    """A statistical reference line (0 for differences, 1 for ratios, nominal for coverage)."""
    if axis == "y":
        return ax.axhline(value, **REFERENCE_LINE, zorder=1)
    return ax.axvline(value, **REFERENCE_LINE, zorder=1)


def save_figure(fig, base: str | Path, png_dpi: int = 600, data: Any = None) -> dict[str, Path]:
    """Write ``<base>.pdf`` (transparent vector), ``<base>.png`` (white, dpi) and the data.

    ``base`` is a path without extension, relative to the repository root.
    ``data`` may be a DataFrame (-> ``_data.csv``) or any JSON-serialisable
    object (-> ``_data.json``). Prints the written paths and closes the figure.
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    pdf = base.with_suffix(".pdf")
    fig.savefig(pdf, format="pdf", transparent=True, bbox_inches="tight", pad_inches=0.02)
    written["pdf"] = pdf

    png = base.with_suffix(".png")
    fig.savefig(
        png,
        format="png",
        dpi=png_dpi,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    written["png"] = png

    if data is not None:
        if hasattr(data, "to_csv"):
            csv = base.parent / f"{base.name}_data.csv"
            data.to_csv(csv, index=False)
            written["data"] = csv
        else:
            js = base.parent / f"{base.name}_data.json"
            js.write_text(json.dumps(data, indent=1, default=str))
            written["data"] = js

    plt.close(fig)
    for kind, path in written.items():
        print(f"{kind:<5} {path}")
    return written

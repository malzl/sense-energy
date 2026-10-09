"""Shared loading for the estates figures (ERIC 2023/24 and KH03 beds); one plot per script."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.config import INTERIM_DIR, PROJECT_ROOT
from sense_energy.visualization import style

RESULTS = PROJECT_ROOT / "code" / "reports" / "estates"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "estates"
CLUSTERING = PROJECT_ROOT / "code" / "reports" / "clustering"


def flags() -> pd.DataFrame:
    f = pd.read_csv(RESULTS / "coverage_flags.csv")
    f["flag_name"] = f["coverage_flag"].map(style.COVERAGE_FLAG_NAMES)
    return f


def eric() -> pd.DataFrame:
    return pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet")


def headroom_panel(included_only: bool = True) -> pd.DataFrame:
    h = pd.read_csv(RESULTS / "heat_pump_headroom_panel.csv")
    return h[h["headroom_status"] == "included"] if included_only else h


def headroom_england() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "heat_pump_headroom_england.csv")


def headroom_class(util: pd.Series) -> pd.Series:
    return pd.cut(util, [-np.inf, 0.8, 1.0, np.inf], labels=list(style.HEADROOM_CLASSES)).astype(
        str
    )


def short_type(t: str) -> str:
    return (
        str(t)
        .replace(" (including Specialist services)", "")
        .replace(" (with inpatient beds)", "")
        .replace(" (acute only)", "")
        .replace("Mental Health and Learning Disabilities", "Mental health and LD")
        .replace("Mental Health", "Mental health")
        .replace("Support Facility", "Support facility")
        .replace("Non inpatient", "Non-inpatient")
        .replace("Other Reportable Site", "Other reportable site")
    )


def plain_log_ticks(axis, ticks) -> None:
    """Log axis with plain-number tick labels at the given positions."""
    import matplotlib.ticker as mticker

    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def loglog(ax, lo: float, hi: float):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], **style.REFERENCE_LINE, zorder=1)
    ax.set_aspect("equal")

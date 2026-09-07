"""Standard project plots.

Keeping these here means every notebook and report renders the same chart the
same way, and figures regenerate reproducibly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..config import FIGURES_DIR, LOCAL_TZ


def plot_load_profile(df: pd.DataFrame, site_id: str | None = None, ax=None):
    """Average demand by time of day, split by weekday vs. weekend."""
    if site_id is not None:
        df = df[df["site_id"] == site_id]
    local = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame(
        {
            "minute_of_day": local.dt.hour * 60 + local.dt.minute,
            "is_weekend": local.dt.dayofweek >= 5,
            "value": df["value"].to_numpy(),
        }
    )

    ax = ax or plt.subplots(figsize=(10, 4))[1]
    for is_weekend, group in frame.groupby("is_weekend"):
        profile = group.groupby("minute_of_day")["value"].mean()
        ax.plot(
            profile.index / 60, profile.to_numpy(), label="Weekend" if is_weekend else "Weekday"
        )
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("Mean demand")
    ax.set_title(f"Load profile{f' — {site_id}' if site_id else ''}")
    ax.legend()
    return ax


def plot_forecast_vs_actual(timestamps, y_true, y_pred, ax=None):
    """Overlay forecast on actuals for a test window."""
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    ax.plot(timestamps, y_true, label="Actual", linewidth=1.2)
    ax.plot(timestamps, y_pred, label="Forecast", linewidth=1.2, alpha=0.8)
    ax.set_ylabel("Demand")
    ax.set_title("Forecast vs. actual")
    ax.legend()
    return ax


def plot_residuals(y_true, y_pred, ax=None):
    """Residual histogram — should be centred on zero and roughly symmetric."""
    import numpy as np

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    ax.hist(residuals, bins=60)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("Forecast - actual")
    ax.set_title("Residual distribution")
    return ax


def save_figure(fig, name: str, dpi: int = 150) -> None:
    """Write a figure to ``code/reports/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")

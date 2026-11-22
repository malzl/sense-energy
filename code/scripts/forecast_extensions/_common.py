"""Shared paths for the figures of the forecast extensions (one plot per script)."""

from __future__ import annotations

import pandas as pd

from sense_energy.config import PROJECT_ROOT

FE = PROJECT_ROOT / "code" / "reports" / "forecast_extensions"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "forecast_extensions"
PLANT, PLAIN = "on-site generation", "no on-site generation"


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(FE / f"{name}.csv")

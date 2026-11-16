"""Shared paths for the whole-record forecast figures (one plot per script)."""

from __future__ import annotations

import pandas as pd

from sense_energy.config import PROJECT_ROOT

FR = PROJECT_ROOT / "code" / "reports" / "forecast_record"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "forecast_record"
YEARS = ["2023", "2024", "2025", "2026"]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(FR / f"{name}.csv")


def best_model(p: pd.DataFrame) -> str:
    """The run with the weather forecast if it covers the record, else the run without."""
    whole = p[p["window"] == "whole record"].set_index("model")["days"]
    return "chronos2_ifs" if whole.get("chronos2_ifs", 0) > 1000 else "chronos2"

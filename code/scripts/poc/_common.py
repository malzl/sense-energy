"""Shared loading for the PoC figure scripts (one plot per script)."""

from __future__ import annotations

import pandas as pd

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

RESULTS = PROJECT_ROOT / "code" / "reports" / "poc"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "poc"
NAIVE = "seasonal_naive"


def display(model_key: str) -> str:
    return style.METHOD_NAMES.get(model_key, model_key)


def ordered_models(keys) -> list[str]:
    names = [display(k) for k in keys]
    order = {m: i for i, m in enumerate(style.METHOD_ORDER)}
    return [k for _, k in sorted(zip([order.get(n, 99) for n in names], keys, strict=True))]


def per_site() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scores_per_site.csv")


def by_lead() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scores_by_lead.csv")


def forecasts(model_key: str) -> pd.DataFrame:
    name = "baselines" if model_key in ("seasonal_naive", "profile_quantiles") else model_key
    f = pd.read_parquet(RESULTS / f"forecasts_{name}.parquet")
    f = f[f["model"] == model_key]
    f["target"] = pd.to_datetime(f["target"], utc=True)
    return f

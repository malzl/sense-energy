"""Run the PoC models and score everything that has been produced."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ..logging_utils import get_logger
from . import poc

logger = get_logger(__name__)


def weather_of(name: str) -> str:
    """``<model>_era5`` / ``<model>_ifs`` variants carry weather covariates; else none."""
    for suffix in ("era5", "ifs", "ifs_members", "ifs_ctrl"):
        if name.endswith("_" + suffix):
            return "era5" if suffix == "era5" else "ifs"
    return "none"


def run_models(config: dict[str, Any], models: list[str]) -> dict[str, pd.DataFrame]:
    panel = poc.load_panel(config)
    origins = poc.make_origins(
        panel, config, config["test_start"], config["test_end"], int(config["origin_stride_days"])
    )
    out_dir = poc.outputs_dir(config)
    produced = {}
    for name in models:
        t0 = time.time()
        weather = weather_of(name)
        if name in ("seasonal_naive", "profile_quantiles", "baselines"):
            fc = poc.run_baselines(panel, origins, config)
            fname = "baselines"
        elif name == "sarima":
            from .sarima import run_sarima

            fc, fname = run_sarima(panel, origins, config), name
        elif name == "chronos2_ifs_members":
            from .ensemble import run_chronos2_members

            fc, fname = run_chronos2_members(panel, origins, config), name
        elif name.startswith("chronos2"):
            from .zero_shot import run_chronos2

            fc, fname = run_chronos2(panel, origins, config, weather=weather), name
        elif name.startswith("timesfm3"):
            from .zero_shot import run_timesfm3

            fc, fname = run_timesfm3(panel, origins, config, weather=weather), name
        elif name.startswith("tabpfn_ts"):
            from .tabpfn_ts import run_tabpfn_ts

            fc, fname = run_tabpfn_ts(panel, origins, config, weather=weather), name
        elif name.startswith("lightgbm"):
            from .gbm import run_lightgbm

            fc, fname = run_lightgbm(panel, origins, config, variant=name), name
        else:
            raise KeyError(name)
        fc.to_parquet(out_dir / f"forecasts_{fname}.parquet", index=False)
        produced[fname] = fc
        logger.info("%s: %s rows in %.0f s", name, f"{len(fc):,}", time.time() - t0)
    return produced


def score_all(config: dict[str, Any]) -> pd.DataFrame:
    panel = poc.load_panel(config)
    out_dir = poc.outputs_dir(config)
    frames = [
        pd.read_parquet(p)
        for p in sorted(out_dir.glob("forecasts_*.parquet"))
        if not p.name.endswith(".partial.parquet")
    ]
    fc = pd.concat(frames, ignore_index=True)
    per, pooled, by_lead = poc.score(fc, panel, list(config["quantiles"]))
    per.to_csv(out_dir / "scores_per_site.csv", index=False)
    pooled.to_csv(out_dir / "scores_pooled.csv", index=False)
    by_lead.to_csv(out_dir / "scores_by_lead.csv", index=False)
    return pooled

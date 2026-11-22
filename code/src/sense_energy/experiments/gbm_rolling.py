"""A trained baseline that is trained again as the data move on.

The proof of concept trains LightGBM once (June 2023 to May 2025) and tests it on the year
after. Over a longer test a model that is trained once grows old, and a comparison with a
zero-shot model is then unfair to it (Spoek et al. 2026 re-train their baselines when a drift
detector fires and still find the zero-shot models ahead). Here the same LightGBM is run over
2024 to May 2026 in two ways:

* ``lightgbm_once``: trained on 18 January to 31 December 2023, then used for the whole test;
* ``lightgbm_quarterly``: trained again at the start of each quarter on all the data up to
  then (an expanding window, the last 60 days for early stopping).

Re-training on a fixed calendar is used in place of a drift detector: it re-trains at least as
often as a detector would over ten quarters and needs no threshold. Only the median model is
fitted (the comparison is of point forecasts). Forecast weather inputs as in ``lightgbm_ifs``.

Output: ``code/reports/forecast_record/forecasts_lightgbm_rolling.parquet``.
"""

from __future__ import annotations

import time

import pandas as pd

from ..config import PROJECT_ROOT, load_config
from ..logging_utils import get_logger
from . import poc
from .gbm import run_lightgbm

logger = get_logger(__name__)

OUT = PROJECT_ROOT / "code" / "reports" / "forecast_record" / "forecasts_lightgbm_rolling.parquet"
CONFIG = "code/configs/experiments/poc.yaml"
TRAIN_FIRST, TEST_FIRST, TEST_LAST = "2023-01-18", "2024-01-01", "2026-05-08"
VARIANT = "lightgbm_ifs"


def quarters(first: str = TEST_FIRST, last: str = TEST_LAST) -> list[tuple[str, str]]:
    """(first day, last day) of each quarter of the test."""
    starts = pd.date_range(first, last, freq="QS")
    ends = [*(s - pd.Timedelta(days=1) for s in starts[1:]), pd.Timestamp(last)]
    return [(str(a.date()), str(b.date())) for a, b in zip(starts, ends, strict=True)]


def _fit_and_forecast(config, panel, train_end: str, first: str, last: str, threads: int):
    cfg = {
        **config,
        "train_start": TRAIN_FIRST,
        "train_end": train_end,
        "lightgbm": {**config["lightgbm"], "quantiles": [0.5], "n_jobs": threads},
    }
    origins = poc.make_origins(panel, cfg, first, last, 1)
    out = run_lightgbm(panel, origins, cfg, VARIANT)
    return out[["site_code", "target", "point"]]


def run(threads: int = 16) -> pd.DataFrame:
    config = load_config(CONFIG)
    panel = poc.load_panel(config)
    t0 = time.time()
    frames = [
        _fit_and_forecast(config, panel, "2023-12-31", TEST_FIRST, TEST_LAST, threads).assign(
            model="lightgbm_once"
        )
    ]
    print(f"once: {time.time() - t0:.0f} s", flush=True)
    for first, last in quarters():
        train_end = str((pd.Timestamp(first) - pd.Timedelta(days=1)).date())
        frames.append(
            _fit_and_forecast(config, panel, train_end, first, last, threads).assign(
                model="lightgbm_quarterly"
            )
        )
        print(f"quarter from {first}: {time.time() - t0:.0f} s", flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT.with_suffix(".partial.parquet"), index=False
        )
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    OUT.with_suffix(".partial.parquet").unlink(missing_ok=True)
    return out

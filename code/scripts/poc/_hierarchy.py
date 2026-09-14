"""Shared loading for the hierarchy figures: the summary table across levels."""

from __future__ import annotations

import pandas as pd
from _common import RESULTS, display

LEVEL_ORDER = ["meter", "site", "trust", "region", "total"]
LEVEL_LABELS = {
    "meter": "meter",
    "site": "site",
    "trust": "trust",
    "region": "region",
    "total": "national",
}


def summary() -> pd.DataFrame:
    s = pd.read_csv(RESULTS.with_name(RESULTS.name + "_hierarchy") / "summary.csv")
    s["method"] = s["model"].map(display)
    s["level_pos"] = s["level"].map({lv: i for i, lv in enumerate(LEVEL_ORDER)})
    return s.sort_values(["level_pos", "model"])

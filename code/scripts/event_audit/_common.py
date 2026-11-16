"""Shared paths for the figures of the event audit (one plot per script)."""

from __future__ import annotations

import sys

import pandas as pd

from sense_energy.config import PROJECT_ROOT

EA = PROJECT_ROOT / "code" / "reports" / "event_audit"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "event_audit"
PLAIN = "no on-site generation"
#: headline numbers with a specification curve: key -> (metric in the tables, axis label)
METRICS = {
    "reaction": ("reaction, no plant", "Reaction of demand (%)"),
    "build_up": ("third day on minus first day", "Third day on minus first day (points)"),
    "evening": ("evening minus morning", "Evening minus morning (points)"),
    "plant": ("plant minus no plant", "Sites with a plant minus the others (points)"),
    "region": ("region share minus chance", "Explained by region, above chance (points)"),
    "own_events": ("other events r2", "Explained by the site's other events (%)"),
}


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(EA / f"{name}.csv")


def metric_arg() -> tuple[str, str, str]:
    key = sys.argv[1] if len(sys.argv) > 1 else "reaction"
    return key, *METRICS[key]

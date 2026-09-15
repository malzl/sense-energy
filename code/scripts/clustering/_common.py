"""Shared loading for the clustering figures (one plot per script)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sense_energy.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "code" / "reports" / "clustering"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "clustering"
POC = PROJECT_ROOT / "code" / "reports" / "poc"
LEVEL_NAMES = {"meter": "meters", "site": "sites", "trust": "trusts"}
ORG_GROUPS = {
    "ACUTE - TEACHING": "acute",
    "ACUTE - LARGE": "acute",
    "ACUTE - MEDIUM": "acute",
    "ACUTE - SMALL": "acute",
    "ACUTE - SPECIALIST": "acute",
    "MENTAL HEALTH AND LEARNING DISABILITY": "mental health",
    "COMMUNITY": "community",
    "AMBULANCE": "ambulance",
}
GROUP_MARKERS = {"acute": "o", "mental health": "s", "community": "^", "ambulance": "D"}


def args(default_energy: str = "elec", default_level: str = "site") -> tuple[str, str]:
    energy = sys.argv[1] if len(sys.argv) > 1 else default_energy
    level = sys.argv[2] if len(sys.argv) > 2 else default_level
    return energy, level


def clusters(energy: str, level: str) -> pd.DataFrame:
    t = pd.read_csv(RESULTS / f"{energy}_{level}_clusters.csv").set_index("series_id")
    if "organisation_type" in t:
        t["org_group"] = t["organisation_type"].map(ORG_GROUPS).fillna("other")
    return t


def profiles(energy: str, level: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"{energy}_{level}_profiles.parquet")


def centroids(energy: str, level: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{energy}_{level}_centroids.csv").set_index("cluster")


def silhouette(energy: str, level: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{energy}_{level}_silhouette.csv")


def cluster_label(i: int, table: pd.DataFrame) -> str:
    n = int((table["cluster"] == i).sum())
    return f"cluster {i} (n = {n})"


def hours() -> np.ndarray:
    return np.arange(48) / 2

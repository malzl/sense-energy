"""Paths and configuration loading.

All paths are derived from the repository root, so code works regardless of the
working directory it is invoked from. Anything environment-specific (API keys,
alternative data locations) comes from ``.env`` — never from hard-coded values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# .../sense-energy/code/src/sense_energy/config.py -> .../sense-energy
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CODE_DIR = PROJECT_ROOT / "code"
DOC_DIR = PROJECT_ROOT / "doc"

# A relative SENSE_DATA_DIR is taken relative to the repository root, never the
# working directory: cron and other launchers do not start in the repo.
_data_dir_setting = Path(os.getenv("SENSE_DATA_DIR", str(CODE_DIR / "data")))
DATA_DIR = (
    _data_dir_setting if _data_dir_setting.is_absolute() else PROJECT_ROOT / _data_dir_setting
).resolve()
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"

CONFIG_DIR = CODE_DIR / "configs"
MODELS_DIR = CODE_DIR / "models"
REPORTS_DIR = CODE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

#: Everything is stored tz-aware UTC; this is for display and for holiday lookups.
LOCAL_TZ = "Europe/London"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving relative paths against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Secrets:
    """Credentials read from the environment. Empty strings mean 'not configured'."""

    cdsapi_url: str = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    cdsapi_key: str = os.getenv("CDSAPI_KEY", "")
    ecds_api_url: str = os.getenv("ECDS_API_URL", "https://ecds.ecmwf.int/api")
    ecds_api_key: str = os.getenv("ECDS_API_KEY", "")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


SECRETS = Secrets()


def ensure_dirs() -> None:
    """Create the data/artifact directories if they do not exist."""
    for directory in (
        RAW_DIR,
        EXTERNAL_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        GEO_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

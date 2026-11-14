"""Shared paths for the forecast-anatomy figures (one plot per script)."""

from __future__ import annotations

from sense_energy.config import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "code" / "reports"
FA = REPORTS / "forecast_anatomy"
OUT = REPORTS / "figures" / "forecast_anatomy"
BEST = "chronos2_ifs"
GROUP_NAMES = {"inpatient": "inpatient", "outpatient": "outpatient", "on-site CHP": "on-site CHP"}

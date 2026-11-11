"""Shared paths for the flexibility, design-winter, heat-pump and solar-programme figures
(one plot per script)."""

from __future__ import annotations

from sense_energy.config import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "code" / "reports"
FLEX = REPORTS / "flexibility"
DESIGN = REPORTS / "design_winter"
HEAT_PUMPS = REPORTS / "heat_pump_field"
SOLAR = REPORTS / "solar_programme"
OUT = REPORTS / "figures" / "flexibility"
YEAR = "2024/25"

"""Offline tests for the NESO module: tidy transforms and time handling."""

from __future__ import annotations

import pandas as pd
import pytest

from sense_energy.data import neso


def test_scenario_codes_cover_recent_fes_pathways():
    for code in ("HE", "EE", "HT", "CF"):
        assert code in neso.SCENARIO_NAMES


def test_melt_years_turns_wide_year_columns_into_long_rows():
    wide = pd.DataFrame(
        {"pathway": ["A"], "unit": ["GW"], "2024": ["1.5"], "2025": [None], "note": ["x"]}
    )
    out = neso._melt_years(wide, ["pathway", "unit"])
    assert list(out.columns) == ["pathway", "unit", "year", "value"]
    assert out["year"].tolist() == [2024, 2025]
    assert out["value"].iloc[0] == 1.5 and pd.isna(out["value"].iloc[1])


def test_settlement_periods_map_to_utc_including_clock_change():
    # 31 March 2024: clocks go forward at 01:00 GMT, so the day has 46 periods.
    date = pd.Series(pd.to_datetime(["2024-03-31", "2024-03-31", "2024-01-10"]))
    period = pd.Series([1, 46, 3])
    out = neso._settlement_to_utc(date, period)
    assert str(out.iloc[0]) == "2024-03-31 00:00:00+00:00"
    assert str(out.iloc[1]) == "2024-03-31 22:30:00+00:00"  # last period of a 46-period day
    assert str(out.iloc[2]) == "2024-01-10 01:00:00+00:00"


def test_latest_forecast_before_respects_lead_time():
    target = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    archive = pd.DataFrame(
        {
            "datetime": [target] * 3,
            "issued_at": [target - pd.Timedelta(hours=h) for h in (30, 20, 2)],
            "embedded_wind_forecast": [100, 110, 120],
        }
    )
    archive["lead_hours"] = (archive["datetime"] - archive["issued_at"]) / pd.Timedelta(hours=1)
    day_ahead = neso.latest_forecast_before(archive, min_lead_hours=12)
    assert (
        len(day_ahead) == 1 and day_ahead["embedded_wind_forecast"].iloc[0] == 110
    )  # the 2 h one is future information


def test_planned_files_expand_years_from_config():
    cfg = {"fes_years": [2025], "history_years": [2024], "forecast_archive_years": []}
    names = [p[0] for p in neso.planned_files(cfg)]
    assert "fes_2025_ed1_demand_summary.csv" in names and "historic_demand_2024.csv" in names
    assert not any(n.startswith("embedded_forecast_archive") for n in names)
    assert "gsp_regions_20260209.zip" in names


def test_manifest_templates_belong_to_exactly_one_family():
    yearly = [t for t in neso.MANIFEST if "{y}" in t]
    for t in yearly:
        assert sum(t in members for members in neso.YEARLY_FAMILIES.values()) == 1, t


@pytest.mark.parametrize("two_digit, full", [(23, 2023), (50, 2050), (2030, 2030)])
def test_two_digit_years_are_expanded(two_digit, full):
    assert (two_digit + 2000 if two_digit < 100 else two_digit) == full


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("01-JAN-2024", "2024-01-01"),
        ("12-AUG-2024", "2024-08-12"),
        ("2026-08-12", "2026-08-12"),
        ('"2025-12-31"', "2025-12-31"),
    ],
)
def test_settlement_dates_never_swap_month_and_day(raw, expected):
    out = neso._parse_settlement_date(pd.Series([raw]))
    assert str(out.iloc[0].date()) == expected

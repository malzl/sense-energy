"""Daily degree days and humidity terms from hourly ERA5 site weather."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.data import weather_daily as dd


def _hourly(start: str, end: str, t_c, td_c=None, ssrd=0.0, site="S1") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    t = np.broadcast_to(np.asarray(t_c, dtype=float), idx.shape)
    td = t - 5 if td_c is None else np.broadcast_to(np.asarray(td_c, dtype=float), idx.shape)
    return pd.DataFrame(
        {
            "site_code": site,
            "datetime": idx,
            "t2m": t + 273.15,
            "d2m": td + 273.15,
            "sp": 101325.0,
            "ssrd": ssrd,
        }
    )


def test_constant_day_degree_days():
    # 2025-07-01 is BST: the local day is 23:00 UTC the day before to 23:00 UTC.
    w = _hourly("2025-06-30 23:00", "2025-07-01 23:00", 20.0)
    out = dd.daily(w).set_index("date").loc["2025-07-01"]
    assert out["n_hours"] == 24
    assert out["cdd_15_5"] == pytest.approx(4.5)
    assert out["cdd_18"] == pytest.approx(2.0)
    assert out["cdd_22"] == 0
    assert out["hdd_15_5"] == 0
    assert out["hours_ge_25"] == 0
    assert out["cdd_eurostat"] == 0
    assert out["t_eve"] == pytest.approx(20.0)


def test_integration_method_counts_a_hot_afternoon():
    # 12 h at 14 degC and 12 h at 26 degC: mean 20, so the daily-mean CDD(22) is
    # zero but four degrees above the base for half the day gives 2 degC day.
    t = np.r_[np.full(12, 14.0), np.full(12, 26.0)]
    w = _hourly("2025-07-01 23:00", "2025-07-02 23:00", t)
    out = dd.daily(w).set_index("date").loc["2025-07-02"]
    assert out["cdd_22"] == pytest.approx(2.0)
    assert out["cdd_dm_22"] == 0
    assert out["hdd_15_5"] == pytest.approx(0.75)
    assert out["hours_ge_25"] == 12
    assert out["t_max"] == pytest.approx(26.0)


def test_eurostat_threshold():
    w = _hourly("2025-07-01 23:00", "2025-07-03 23:00", np.r_[np.full(24, 24.0), np.full(24, 25.0)])
    out = dd.daily(w).set_index("date")["cdd_eurostat"]
    assert out.loc["2025-07-02"] == 0  # mean 24 is not above the threshold
    assert out.loc["2025-07-03"] == pytest.approx(4.0)


def test_bst_transition_days_have_23_and_25_hours():
    w = _hourly("2025-03-29 00:00", "2025-11-01 00:00", 10.0)
    n = dd.daily(w).set_index("date")["n_hours"]
    assert n.loc["2025-03-30"] == 23
    assert n.loc["2025-10-26"] == 25
    assert n.loc["2025-07-01"] == 24


def test_psychrometrics_match_reference_values():
    # Stull (2011): T = 20 degC, RH = 50 % gives Tw = 13.7 degC.
    assert float(dd.wet_bulb_stull(20.0, 50.0)) == pytest.approx(13.7, abs=0.1)
    # Saturated air: dew point equals the temperature.
    assert float(dd.relative_humidity(18.0, 18.0)) == pytest.approx(100.0)
    # 20 degC, dew point 9.3 degC (about 50 % RH) at 1013 hPa: about 38.5 kJ/kg.
    assert float(dd.moist_air_enthalpy(20.0, 9.3, 101325.0)) == pytest.approx(38.5, abs=0.5)


def test_solar_is_dated_by_the_hour_it_accumulates_over():
    # One hour of 3.6 MJ/m2 (1 kWh/m2) stamped 23:00 UTC on 1 July covers 22:00-23:00
    # UTC = 23:00-00:00 BST, so it belongs to 1 July local, not 2 July.
    w = _hourly("2025-06-30 23:00", "2025-07-02 23:00", 15.0)
    w.loc[w["datetime"] == pd.Timestamp("2025-07-01 23:00", tz="UTC"), "ssrd"] = 3.6e6
    out = dd.daily(w).set_index("date")["solar_kwh_m2"]
    assert out.loc["2025-07-01"] == pytest.approx(1.0)
    assert out.loc["2025-07-02"] == 0


def test_wind_rain_and_snow_daily_terms():
    w = _hourly("2025-01-01 00:00", "2025-01-03 00:00", 1.0)  # GMT: local day = UTC day
    w["u10"], w["v10"], w["i10fg"] = 3.0, 4.0, 10.0
    w["tp"], w["sf"] = 0.001, 0.0  # 1 mm per hour
    w.loc[w["datetime"] == pd.Timestamp("2025-01-01 12:00", tz="UTC"), "i10fg"] = 30.0
    out = dd.daily(w).set_index("date")
    assert out.loc["2025-01-01", "wind_mean"] == pytest.approx(5.0)
    assert out.loc["2025-01-01", "gust_max"] == pytest.approx(30.0)
    assert out.loc["2025-01-02", "gust_max"] == pytest.approx(10.0)
    # The 00:00 UTC value on 2 Jan accumulates over 23:00-00:00 on 1 Jan.
    assert out.loc["2025-01-01", "precip_mm"] == pytest.approx(24.0)
    assert out.loc["2025-01-01", "snow_mm_we"] == 0


def test_evening_temperature_covers_18_to_midnight_local():
    # GMT day: hours 0-17 at 10 degC, 18-23 at 16 degC -> evening mean 16, daily mean 11.5.
    t = np.r_[np.full(18, 10.0), np.full(6, 16.0)]
    w = _hourly("2025-01-10 00:00", "2025-01-11 00:00", t)
    out = dd.daily(w).set_index("date").loc["2025-01-10"]
    assert out["t_eve"] == pytest.approx(16.0)
    assert out["t_mean"] == pytest.approx(11.5)


def test_thermometer_only_product_has_no_flux_columns():
    w = _hourly("2025-01-10 00:00", "2025-01-12 00:00", 7.0).drop(columns="ssrd")
    out = dd.daily(w)
    assert len(out) == 2
    assert out["t_mean"].tolist() == pytest.approx([7.0, 7.0])
    assert not {"solar_kwh_m2", "precip_mm", "snow_mm_we", "wind_mean"} & set(out.columns)

"""Flexibility, design winters, heat pumps in the field, the solar programme and the long ERA5
record: the pure parts, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import design_winter as dw
from sense_energy.analysis import flexibility as fx
from sense_energy.analysis import heat_pump_field as hpf
from sense_energy.analysis import solar_programme as sp
from sense_energy.data import era5_timeseries as ets
from sense_energy.data import prices


def test_load_mid_keeps_one_traded_row_per_half_hour(tmp_path):
    t = pd.date_range("2025-01-01", periods=4, freq="30min", tz="UTC")
    raw = pd.DataFrame(
        {
            "datetime": list(t) * 2,
            "provider": ["APXMIDP"] * 4 + ["N2EXMIDP"] * 4,
            "price_gbp_mwh": [80.0, 90.0, 100.0, 0.0] + [0.0] * 4,
            "volume_mwh": [10.0, 10.0, 10.0, 0.0] + [0.0] * 4,
        }
    )
    path = tmp_path / "mid.parquet"
    raw.to_parquet(path)
    out = prices.load_mid(path)
    assert out["datetime"].is_unique and len(out) == 3  # the untraded period is dropped
    assert out["price_gbp_mwh"].mean() == pytest.approx(90.0)  # not halved by the empty provider


def test_calendar_marks_the_red_band_on_working_days_only():
    idx = pd.date_range("2025-12-24", "2025-12-27", freq="30min", tz="Europe/London").tz_convert(
        "UTC"
    )
    cal = fx.calendar(idx)
    red_days = cal[cal["red"]]["date"].dt.date.astype(str).unique().tolist()
    assert red_days == ["2025-12-24"]  # Christmas Day and Boxing Day are holidays
    assert set(cal[cal["red"]]["hour"]) == {16, 17, 18}
    assert (cal["fy"] == "2025/26").all()


def test_battery_day_buys_low_and_sells_high():
    price = np.r_[np.full(24, 5.0), np.full(24, 25.0)]  # p/kWh: cheap morning, dear afternoon
    value, net = fx.battery_day(price)
    # 2 kWh sold at 25p; bought with the round-trip loss at 5p
    expected = (2.0 * 25.0 - 2.0 / fx.ROUND_TRIP * 5.0) / 100.0
    assert value == pytest.approx(expected, rel=1e-6)
    assert net[:24].sum() < 0 < net[24:].sum()
    flat, _ = fx.battery_day(np.full(48, 10.0))
    assert flat == pytest.approx(0.0, abs=1e-9)  # no spread, no value


def test_threshold_plan_and_real_time_control():
    load = np.full((1, 48), 10.0)
    load[0, 20:24] = 20.0  # a two-hour peak of 10 kW above the base
    p, e = np.array([5.0]), np.array([10.0])
    theta = fx.plan_threshold(load, p, e)
    assert theta[0] == pytest.approx(15.0, abs=1e-6)  # power-limited: 5 kW for 2 h = 10 kWh
    assert fx.realised_peak(theta, load, p, e)[0] == pytest.approx(15.0, abs=1e-6)
    # a forecast that misses the peak sets the ceiling too low: the store empties and the
    # real peak comes through
    low = fx.plan_threshold(
        np.full((1, 48), 10.0) + np.r_[np.zeros(20), 2.0 * np.ones(4), np.zeros(24)], p, e
    )
    assert fx.realised_peak(low, load, p, e)[0] == pytest.approx(20.0)


def test_bill_anatomy_shape_premium():
    idx = pd.date_range("2024-04-01", "2025-03-31 23:30", freq="30min", tz="Europe/London")
    idx = idx.tz_convert("UTC")
    cal = fx.calendar(idx)
    price = pd.Series(np.where(cal["hour"].between(8, 19), 12.0, 6.0), index=idx)
    w = pd.DataFrame(
        {"flat": 10.0, "day": np.where(cal["hour"].between(8, 19), 20.0, 0.0)}, index=idx
    )
    b = fx.bill_anatomy(w, cal, price).set_index("site_code")
    assert b.loc["flat", "shape_premium_pct"] == pytest.approx(0.0, abs=1e-9)
    assert b.loc["day", "paid_p_kwh"] == pytest.approx(12.0)
    assert b.loc["day", "shape_premium_pct"] > 30
    assert b.loc["flat", "red_share_energy"] == pytest.approx(b.loc["flat", "red_share_time"])


def test_storage_need_counts_the_run_of_excess():
    load = np.array([[5.0, 12.0, 12.0, 5.0, 5.0, 12.0, 5.0, 5.0]])
    p, e, hours = dw.storage_need(load, np.array([10.0]))
    assert p[0] == pytest.approx(2.0) and hours[0] == pytest.approx(1.5)
    assert e[0] == pytest.approx(2.0)  # two half-hours of 2 kW; fully recharged before the third
    assert dw.store_class(0.0) == "within capacity"
    assert dw.store_class(1.0).startswith("battery") and dw.store_class(100.0).startswith("beyond")
    assert dw.year_label(1962) == "1962/63"


def test_implied_kwp_inverts_the_slope_change():
    # 100 kWp at PR 0.8 lower a 12-hour mean by 100 x 0.8 / 12 kW per kWh/m2
    day_kw = 200.0
    change_pct = -100 * (100 * fx.PERFORMANCE_RATIO / 12.0) / day_kw
    k = sp.implied_kwp(pd.Series([change_pct]), pd.Series([day_kw]))
    assert k.iloc[0] == pytest.approx(100.0)
    d = pd.Series(pd.to_datetime(["2026-02-28", "2026-03-01", "2026-05-08", "2026-05-09"]))
    assert sp.in_window(d).tolist() == [False, True, True, False]


def test_funded_postcodes_splits_and_normalises(tmp_path):
    path = tmp_path / "gbe.csv"
    path.write_text(
        "ODS Trust Code,Trust name,Region,Postcode(s), No. , Funding (\xa3) , Savings \n"
        'RXX,SOME TRUST,LONDON,"AB1 2CD, ef3  4gh",2," 300,000 ",x\n',
        encoding="latin-1",
    )
    f = sp.funded_postcodes(path)
    assert f["postcode_norm"].tolist() == ["AB12CD", "EF34GH"]
    assert f["trust_funding_gbp"].iloc[0] == 300000.0 and f["trust_sites_funded"].iloc[0] == 2


def test_heat_pump_status_and_switchers(tmp_path, monkeypatch):
    monkeypatch.setattr(hpf, "INTERIM_DIR", tmp_path)
    base = {
        "gia_m2": 1000.0,
        "chp_units": 0.0,
        "site_name": "s",
        "site_type": "t",
        "trust_name": "x",
    }
    a = pd.DataFrame(
        [
            {"site_code": "new", "elec_kwh": 100e3, "gas_kwh": 400e3, "heat_pumps": 0.0, **base},
            {"site_code": "never", "elec_kwh": 100e3, "gas_kwh": 400e3, "heat_pumps": 0.0, **base},
            {"site_code": "scope", "elec_kwh": 100e3, "gas_kwh": 400e3, "heat_pumps": 0.0, **base},
        ]
    )
    b = a.copy()
    b.loc[0, ["elec_kwh", "gas_kwh", "heat_pumps"]] = [200e3, 100e3, 2.0]  # 255 MWh heat for 100
    b.loc[2, "elec_kwh"] = 900e3  # a ninefold jump is a reporting change
    a.to_parquet(tmp_path / "eric_2023_24_site.parquet")
    b.to_parquet(tmp_path / "eric_2024_25_site.parquet")
    m = hpf.changes("2023_24", "2024_25").set_index("site_code")
    assert m.loc["new", "status"] == "new" and m.loc["never", "status"] == "never"
    assert m.loc["new", "usable"] and not m.loc["scope", "plausible"]
    s = hpf.switchers(m.reset_index()).set_index("site_code")
    assert list(s.index) == ["new"]
    assert s.loc["new", "implied_cop"] == pytest.approx(0.85 * 300e3 / 100e3)


def test_era5_cells_and_daily_stats():
    assert ets.cell_id(51.5, -0.25) == "+051.50_-000.25"
    t = pd.date_range("2000-01-01", periods=72, freq="h", tz="UTC")
    h = pd.DataFrame({"datetime": t, "t2m": np.r_[np.zeros(24), np.arange(24.0), np.full(24, 5.0)]})
    d = ets.daily_stats(h).set_index("date")
    assert d.loc["2000-01-02", "t_max"] == 23.0 and d.loc["2000-01-02", "t_mean"] == 11.5
    assert len(d) == 3

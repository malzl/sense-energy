"""NOAA ISD parsing, station candidates and the daily site table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.data import isd


def test_field_parses_tenths_and_quality():
    s = pd.Series(["+0123,1", "-0045,5", "+9999,9", "+0200,3", None])
    v = isd._field(s)
    assert v.iloc[0] == pytest.approx(12.3)
    assert v.iloc[1] == pytest.approx(-4.5)
    assert np.isnan(v.iloc[2])  # missing marker
    assert np.isnan(v.iloc[3])  # quality code 3 = erroneous
    assert np.isnan(v.iloc[4])


def test_parse_file_hourly_means_and_skips_summaries(tmp_path):
    rows = [
        ("03772099999", "2025-07-01T12:20:00", "FM-15", "+0200,1", "+0100,1"),
        ("03772099999", "2025-07-01T12:50:00", "FM-15", "+0220,1", "+0120,1"),
        ("03772099999", "2025-07-01T13:50:00", "FM-12", "+0250,1", "+9999,9"),
        ("03772099999", "2025-07-01T23:59:00", "SOD", "+9999,9", "+9999,9"),
    ]
    f = tmp_path / "03772099999.csv"
    pd.DataFrame(rows, columns=["STATION", "DATE", "REPORT_TYPE", "TMP", "DEW"]).to_csv(
        f, index=False
    )
    out = isd.parse_file(f)
    assert len(out) == 2
    assert out["t_c"].tolist() == pytest.approx([21.0, 25.0])
    assert out["datetime"].iloc[0] == pd.Timestamp("2025-07-01 12:00", tz="UTC")
    assert np.isnan(out["td_c"].iloc[1])


def test_candidates_rank_by_distance_within_the_limit():
    sites = pd.DataFrame({"site_code": ["A"], "latitude": [51.5], "longitude": [-0.1]})
    stations = pd.DataFrame(
        {
            "station_id": ["near", "mid", "far"],
            "lat": [51.48, 51.15, 53.35],
            "lon": [-0.46, -0.19, -2.28],
        }
    )
    c = isd.candidates(sites, stations)
    assert c["station_id"].tolist() == ["near", "mid"]  # Manchester is beyond 60 km
    assert c["rank"].tolist() == [0, 1]
    assert c["distance_km"].iloc[0] < c["distance_km"].iloc[1]


def test_site_hourly_falls_back_to_the_next_station_and_daily_needs_18_hours():
    idx = pd.date_range("2025-01-10", periods=48, freq="h", tz="UTC")
    hourly = pd.concat(
        [
            pd.DataFrame({"station_id": "near", "datetime": idx[:30], "t_c": 5.0, "td_c": 1.0}),
            pd.DataFrame({"station_id": "mid", "datetime": idx, "t_c": 8.0, "td_c": 1.0}),
        ]
    )
    cand = pd.DataFrame(
        {
            "site_code": "A",
            "station_id": ["near", "mid"],
            "rank": [0, 1],
            "distance_km": [5.0, 30.0],
        }
    )
    sh = isd.site_hourly(hourly, cand)
    assert len(sh) == 48
    assert (sh["station_id"].iloc[:30] == "near").all() and (
        sh["station_id"].iloc[30:] == "mid"
    ).all()
    d = isd.daily(sh).set_index("date")
    assert d.loc["2025-01-10", "t_mean"] == pytest.approx(5.0)
    assert d.loc["2025-01-11", "t_mean"] == pytest.approx((6 * 5.0 + 18 * 8.0) / 24)
    assert d.loc["2025-01-11", "t_eve"] == pytest.approx(8.0)
    assert d.loc["2025-01-10", "n_hours"] == 24

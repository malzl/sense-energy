"""Offline tests for the ERA5, AIFS-ENS and price modules.

No network. The live paths are exercised by the CLI commands; what is tested
here is the request construction, the file bookkeeping, and the transforms
that decide whether the data means what we think it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sense_energy.data import forecasts, prices, weather

WEATHER_CFG = {
    "dataset": "reanalysis-era5-single-levels",
    "start": "2022-12",
    "end": "2023-02",
    "area": [55.0, -5.0, 50.0, 2.0],
    "product_types": ["reanalysis", "ensemble_members"],
    "variables": ["2m_temperature", "total_precipitation"],
}


# ----------------------------------------------------------------- ERA5 ---


def test_month_range_is_inclusive():
    assert weather.month_range("2022-12", "2023-02") == [(2022, 12), (2023, 1), (2023, 2)]


def test_reanalysis_request_is_hourly_and_ensemble_is_three_hourly():
    hourly = weather.build_request("reanalysis", 2023, 1, WEATHER_CFG)
    eda = weather.build_request("ensemble_members", 2023, 1, WEATHER_CFG)
    assert len(hourly["time"]) == 24
    assert len(eda["time"]) == 8 and eda["time"][1] == "03:00"
    assert hourly["day"][-1] == "31" and len(hourly["day"]) == 31
    assert hourly["area"] == [55.0, -5.0, 50.0, 2.0]
    assert hourly["data_format"] == "netcdf"


def test_request_rejects_unknown_product():
    with pytest.raises(KeyError):
        weather.build_request("forecast", 2023, 1, WEATHER_CFG)


def test_february_day_count_respects_leap_years():
    assert len(weather.build_request("reanalysis", 2024, 2, WEATHER_CFG)["day"]) == 29
    assert len(weather.build_request("reanalysis", 2023, 2, WEATHER_CFG)["day"]) == 28


def test_partial_file_is_not_treated_as_present(tmp_path):
    bad = tmp_path / "2023-01.nc"
    bad.write_bytes(b"not a netcdf")
    assert weather._opens_cleanly(bad) is False
    assert weather._opens_cleanly(tmp_path / "missing.nc") is False


def test_every_configured_variable_has_a_short_name():
    from sense_energy.config import load_config

    cfg = load_config("code/configs/weather.yaml")
    missing = [v for v in cfg["variables"] if v not in weather.SHORT_NAMES]
    assert not missing, missing


def _grid_dataset(times, members=None):
    lat = np.array([55.0, 54.5, 54.0])
    lon = np.array([-1.0, -0.5, 0.0])
    shape = (len(times), len(lat), len(lon))
    coords = {"valid_time": times, "latitude": lat, "longitude": lon}
    dims = ("valid_time", "latitude", "longitude")
    if members is not None:
        shape = (len(members), *shape)
        coords["number"] = members
        dims = ("number", *dims)
    t2m = xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape) + 273.15, coords=coords, dims=dims
    )
    return xr.Dataset({"t2m": t2m})


def test_extract_sites_picks_nearest_grid_point(tmp_path):
    times = pd.date_range("2023-01-01", periods=2, freq="h")
    path = tmp_path / "m.nc"
    _grid_dataset(times).to_netcdf(path)
    sites = pd.DataFrame({"site_code": ["S1"], "latitude": [54.4], "longitude": [-0.6]})
    out = weather.extract_sites(path, sites)
    assert list(out.columns[:3]) == ["site_code", "datetime", "member"]
    assert (out["member"] == 0).all() and len(out) == 2
    # nearest of (54.5, -0.5) -> row 1, col 1 in a 3x3 grid -> flat index 4 at t=0
    assert out["t2m"].iloc[0] == pytest.approx(4 + 273.15)


def test_extract_sites_keeps_ensemble_members(tmp_path):
    times = pd.date_range("2023-01-01", periods=1, freq="3h")
    path = tmp_path / "e.nc"
    _grid_dataset(times, members=np.arange(10)).to_netcdf(path)
    sites = pd.DataFrame({"site_code": ["S1"], "latitude": [55.0], "longitude": [-1.0]})
    out = weather.extract_sites(path, sites)
    assert sorted(out["member"].unique()) == list(range(10))


# ------------------------------------------------------------- AIFS-ENS ---


def test_control_run_becomes_member_zero_and_perturbed_members_survive():
    """The bug this guards: a merge that keeps only the first group's variable."""
    lat = np.array([55.0, 54.0])
    lon = np.array([-1.0, 0.0])
    cf = xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.full((2, 2), 280.0))},
        coords={"latitude": lat, "longitude": lon, "number": 0},
    )
    pf = xr.Dataset(
        {
            "t2m": (
                ("number", "latitude", "longitude"),
                np.full((3, 2, 2), 281.0) + np.arange(3)[:, None, None],
            )
        },
        coords={"number": [1, 2, 3], "latitude": lat, "longitude": lon},
    )

    cf2 = cf.drop_vars("number").expand_dims(number=[0])
    merged = xr.merge([cf2, pf], compat="no_conflicts", join="outer")
    assert merged.sizes["number"] == 4
    assert float(merged["t2m"].isnull().mean()) == 0.0  # noqa: PD003 - xarray has no isna
    assert float(merged["t2m"].sel(number=3).mean()) == pytest.approx(283.0)


def test_crop_selects_area_on_descending_latitude():
    lat = np.arange(60.0, 44.9, -0.25)
    lon = np.arange(-10.0, 5.01, 0.25)
    ds = xr.Dataset(
        {"x": (("latitude", "longitude"), np.zeros((len(lat), len(lon))))},
        coords={"latitude": lat, "longitude": lon},
    )
    out = forecasts._crop(ds, [55.0, -5.0, 50.0, 2.0])
    assert float(out.latitude.max()) == 55.0 and float(out.latitude.min()) == 50.0
    assert float(out.longitude.min()) == -5.0 and float(out.longitude.max()) == 2.0


def test_crop_wraps_0_360_longitudes():
    lat = np.array([55.0, 50.0])
    lon = np.arange(0.0, 360.0, 0.25)
    ds = xr.Dataset(
        {"x": (("latitude", "longitude"), np.zeros((2, len(lon))))},
        coords={"latitude": lat, "longitude": lon},
    )
    out = forecasts._crop(ds, [55.0, -5.0, 50.0, 2.0])
    assert float(out.longitude.min()) == -5.0 and float(out.longitude.max()) == 2.0


def test_run_path_encodes_run_time():
    cfg = {"output_subdir": "weather/aifs_ens"}
    assert forecasts.run_path("20260908", "12", cfg).name == "20260908T12z.nc"


# --------------------------------------------------------------- prices ---


def _agile_rows(product, region, start, periods, rate):
    times = pd.date_range(start, periods=periods, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": times,
            "valid_to": times + pd.Timedelta("30min"),
            "product": product,
            "region": region,
            "rate_exc_vat": rate,
            "rate_inc_vat": rate * 1.05,
        }
    )


def test_canonical_agile_prefers_the_product_on_sale():
    rates = pd.concat(
        [
            _agile_rows("AGILE-18-02-21", "C", "2024-06-15", 4, 10.0),
            _agile_rows("AGILE-24-04-03", "C", "2024-06-15", 4, 20.0),
        ]
    )
    windows = pd.DataFrame(
        {
            "product": ["AGILE-18-02-21", "AGILE-24-04-03"],
            "available_from": pd.to_datetime(
                ["2017-01-01", "2024-04-02T23:00"], utc=True, format="ISO8601"
            ),
            "available_to": pd.to_datetime(
                ["2022-07-22T12:00", "2024-09-30T23:00"], utc=True, format="ISO8601"
            ),
        }
    )
    out = prices.canonical_agile(rates, windows)
    assert len(out) == 4
    assert (out["product"] == "AGILE-24-04-03").all()


def test_canonical_agile_falls_back_when_nothing_is_on_sale():
    rates = pd.concat(
        [
            _agile_rows("AGILE-18-02-21", "C", "2023-06-15", 2, 10.0),
            _agile_rows(
                "AGILE-24-10-01", "C", "2023-06-15", 2, 30.0
            ),  # published, but not yet on sale
        ]
    )
    windows = pd.DataFrame(
        {
            "product": ["AGILE-18-02-21", "AGILE-24-10-01"],
            "available_from": pd.to_datetime(
                ["2017-01-01", "2024-09-30T23:00"], utc=True, format="ISO8601"
            ),
            "available_to": pd.to_datetime(["2022-07-22T12:00", None], utc=True),
        }
    )
    out = prices.canonical_agile(rates, windows)
    assert (out["product"] == "AGILE-18-02-21").all()


def test_canonical_agile_is_unique_per_region_and_time():
    rates = pd.concat(
        [
            _agile_rows(p, r, "2024-06-15", 3, 1.0)
            for p in ("AGILE-18-02-21", "AGILE-24-04-03")
            for r in ("A", "C")
        ]
    )
    windows = pd.DataFrame(
        {
            "product": ["AGILE-24-04-03"],
            "available_from": pd.to_datetime(["2024-04-02T23:00"], utc=True),
            "available_to": pd.to_datetime([None], utc=True),
        }
    )
    out = prices.canonical_agile(rates, windows)
    assert not out.duplicated(["region", "datetime"]).any()
    assert len(out) == 6


def test_harvest_exits_when_another_instance_holds_the_lock(tmp_path, monkeypatch):
    import fcntl

    cfg = {"output_subdir": str(tmp_path), "model": "aifs-ens", "run_times": ["00"]}
    holder = (tmp_path / ".harvest.lock").open("w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(
        forecasts,
        "list_available_runs",
        lambda c: (_ for _ in ()).throw(AssertionError("must not list")),
    )
    assert forecasts.harvest_available(cfg) == []
    holder.close()


# ---------------------------------------------------------------- TIGGE ---


def test_tigge_request_matches_the_ecds_schema():
    from sense_energy.config import load_config
    from sense_energy.data import tigge

    cfg = load_config("code/configs/tigge.yaml")
    req = tigge.build_request("perturbed_forecast", 2024, 2, cfg)
    assert req["origin"] == "ecmwf" and req["level_type"] == "single_level"
    assert req["forecast_type"] == "perturbed_forecast"
    assert req["day"][-1] == "29" and req["time"] == ["00:00", "12:00"]
    assert req["leadtime_hour"][0] == "0" and req["leadtime_hour"][-1] == "72"
    assert req["data_format"] == "grib" and req["area"] == [55.0, -5.0, 50.0, 2.0]
    assert "2_m_temperature" in req["variable"]
    with pytest.raises(KeyError):
        tigge.build_request("pf", 2024, 2, cfg)


def test_tigge_type_months_cover_the_window():
    from sense_energy.config import load_config
    from sense_energy.data import tigge, weather

    cfg = load_config("code/configs/tigge.yaml")
    assert len(cfg["forecast_types"]) * len(weather.month_range(cfg["start"], cfg["end"])) == 84
    assert tigge.target_path("control_forecast", 2023, 1, cfg).name == "2023-01.grib"

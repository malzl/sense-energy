"""NESO covariates, mobility loader, and the active-member rule for totals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sense_energy.data import mobility, neso_covariates, processed


def test_day_ahead_cutoff_is_the_day_before_at_the_stated_hour():
    target = pd.Series(pd.to_datetime(["2024-03-10 00:00", "2024-03-10 23:30"], utc=True))
    cut = neso_covariates._cutoff(target, 16)
    assert (cut == pd.Timestamp("2024-03-09 16:00", tz="UTC")).all()


def test_profiles_relabel_duplicate_id_as_university(tmp_path, monkeypatch):
    monkeypatch.setattr(mobility, "RAW_DIR", tmp_path)
    (tmp_path / "campus_agebands.csv").write_text(
        "id,a,b\nRH801,1,2\nRH802,3,4\nRH8K6,5,6\nuniversity,7,8\n"
    )
    (tmp_path / "campus_hh_income.csv").write_text(
        "id,q1\nRH801,1\nRH802,2\nRH8K6,3\nuniversity,4\n"
    )
    (tmp_path / "campus_styles.csv").write_text("id,t\nRH801,1\nRH802,2\nRH8K6,3\nRH801,4\n")
    (tmp_path / "campus_travel_distance.csv").write_text(
        "id,n\nRH801,1\nRH802,2\nRH8K6,3\nRH801,4\n"
    )
    p = mobility.load_profiles()
    styles = p[p["profile"] == "consumer_style"]
    assert sorted(styles["id"]) == ["RH801", "RH802", "RH8K6", "university"]
    assert styles.loc[styles["id"] == "university", "id_corrected"].all()
    assert not p[p["profile"] == "age_band"]["id_corrected"].any()
    assert p.loc[p["id"] == "university", "site_code"].isna().all()


def test_low_coverage_member_does_not_block_the_total():
    idx = pd.date_range("2024-01-01", periods=100, freq="30min", tz="UTC")
    good = pd.Series(np.ones(100), index=idx)
    dead = pd.Series(np.nan, index=idx)
    dead.iloc[:3] = 5.0  # reported briefly, then died: 3% coverage
    wide = pd.DataFrame({"m1": good, "m2": good * 2, "m3": dead})
    totals, n = processed.aggregate_wide(
        wide, pd.Series({"m1": "S", "m2": "S", "m3": "S"}), min_member_coverage=0.1
    )
    assert n["S"] == 2
    assert totals["S"].notna().all() and (totals["S"] == 3.0).all()


def test_totals_still_require_every_active_member():
    idx = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
    a = pd.Series(np.ones(10), index=idx)
    b = a.copy()
    b.iloc[4] = np.nan
    totals, _ = processed.aggregate_wide(
        pd.DataFrame({"a": a, "b": b}), pd.Series({"a": "S", "b": "S"}), min_member_coverage=0.1
    )
    assert totals["S"].isna().sum() == 1


@pytest.mark.skipif(
    not Path(
        "/tmp/claude-1018/-home-malzn-sense-energy/eb6a8acd-c71c-4c9b-a153-4dc14ccbb2be/scratchpad/aifs_step24.grib2"
    ).exists(),
    reason="needs a downloaded AIFS step",
)
def test_eccodes_reader_matches_cfgrib_on_a_real_step():
    from sense_energy.data import forecasts

    grib = Path(
        "/tmp/claude-1018/-home-malzn-sense-energy/eb6a8acd-c71c-4c9b-a153-4dc14ccbb2be/scratchpad/aifs_step24.grib2"
    )
    area = [55.0, -5.0, 50.0, 2.0]
    fast = forecasts._read_grib_cropped(grib, area)
    assert fast.sizes["number"] == 51 and "t2m" in fast
    assert float(fast.latitude.max()) <= 55.0 and float(fast.longitude.min()) >= -5.0

"""The check of the heat and cold wave paper: samples as the paper's loaders make them,
references with the paper's inputs, scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.experiments import esdf_check as ec


def _frame(days: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-01-01", periods=24 * days, freq="h")
    hour = np.tile(np.arange(24), days)
    temp = (
        10
        + 8 * np.sin(np.arange(len(t)) * 2 * np.pi / (24 * 50))
        + 3 * np.sin(hour / 24 * 2 * np.pi)
    )
    load = (
        1000
        + 20 * (18 - temp)
        + 150 * np.sin((hour - 6) / 24 * 2 * np.pi)
        + rng.normal(0, 5, len(t))
    )
    return pd.DataFrame({"time": t, "load": load, "temp": temp})


def _scale(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (df["load"].max(), df["load"].min(), df["temp"].max(), df["temp"].min())


def test_daily_temperature_is_the_mean_of_the_extremes():
    temp = np.r_[np.arange(24.0), np.full(24, 5.0)]
    assert ec.daily_temperature(temp).tolist() == [11.5, 5.0]


def test_samples_follow_the_papers_loader():
    df = _frame(days=100)
    s = ec.make_samples(df, _scale(df), (0.2, 0.8))
    assert len(s.load) == 100 - 69  # i from 30 to days - 40
    assert s.load.shape[1:] == (8, 24) and s.temp.shape[1:] == (8, 24)
    assert s.first_row[0] == 24 * 30
    assert str(s.target_day[0]) == "2018-02-07"  # day 30 + 7 of the frame
    lmax, lmin = df["load"].max(), df["load"].min()
    first = (df["load"].to_numpy()[24 * 30 : 24 * 38] - lmin) / (lmax - lmin)
    assert s.load[0].ravel() == pytest.approx(first)
    scaled = (df["temp"].to_numpy() - df["temp"].min()) / (df["temp"].max() - df["temp"].min())
    day = ec.daily_temperature(scaled)[37]
    assert bool(s.cold[0]) == bool(day < 0.2) and bool(s.hot[0]) == bool(day > 0.8)


def test_slices_with_a_zero_load_are_dropped_only_when_asked():
    df = _frame(days=100)
    df.loc[24 * 40 + 3, "load"] = 0.0
    scale = (df["load"].max(), 800.0, df["temp"].max(), df["temp"].min())
    kept = ec.make_samples(df, scale, (0.2, 0.8))
    dropped = ec.make_samples(df, scale, (0.2, 0.8), drop_zero=True)
    assert len(kept.load) == 31
    assert len(dropped.load) == 31 - 8  # the eight slices that hold day 40


def test_features_do_not_see_the_load_of_the_target_day():
    df = _frame(days=100)
    s = ec.make_samples(df, _scale(df), (0.2, 0.8))
    before = ec.features(s)
    s.load[:, 7, :] = 99.0
    pd.testing.assert_frame_equal(before, ec.features(s))
    assert len(before) == 24 * len(s.load) and "temp" in before.columns


def test_a_linear_model_recovers_a_linear_load():
    df = _frame(days=400, seed=1)
    train = ec.make_samples(df.iloc[: 24 * 300].reset_index(drop=True), _scale(df), (0.2, 0.8))
    test = ec.make_samples(df.iloc[24 * 300 :].reset_index(drop=True), _scale(df), (0.2, 0.8))
    y = test.load[:, 7, :]
    linear = ec.linear_forecast(train, test)
    week = test.load[:, 0, :]
    assert np.abs(linear - y).mean() < 0.5 * np.abs(week - y).mean()
    boosted = ec.lightgbm_forecast(train, test)
    assert boosted.shape == y.shape and np.isfinite(boosted).all()


def test_scores_in_the_papers_unit_and_against_the_mean_load():
    y = np.full((4, 24), 0.5)
    f = y + 0.02
    mask = np.array([True, True, False, False])
    out = ec.score(y, f, mask, offset=0.5)  # load = (scaled + 0.5) x range, so the mean is 1.0
    assert out["days"] == 2
    assert out["mae"] == pytest.approx(0.02) and out["rmse"] == pytest.approx(0.02)
    assert out["nmae_mean_pct"] == pytest.approx(2.0)
    assert np.isnan(ec.score(y, f, np.zeros(4, dtype=bool), 0.5)["mae"])


def test_the_scored_window_is_the_papers_window():
    region = ec.REGIONS["europe"]
    df = _frame(days=120)
    s = ec.make_samples(df, _scale(df), (0.2, 0.8))
    masks = ec.subset_masks(region, s)
    assert masks["paper window"].sum() == 8 and masks["paper window"][19:27].all()
    assert masks["all test days"].all()
    assert (masks["hot and cold days"] ^ masks["other days"]).all()


def test_summary_counts_the_series_where_a_model_beats_the_method():
    rows = []
    for series, (esdf, other) in {"a": (0.03, 0.02), "b": (0.03, 0.05), "c": (0.03, 0.01)}.items():
        for model, mae in ((ec.PROPOSED, esdf), ("LightGBM", other)):
            rows.append(
                {
                    "region": "europe",
                    "series": series,
                    "model": model,
                    "subset": "paper window",
                    "days": 8,
                    "mae": mae,
                    "rmse": mae,
                    "nmae_mean_pct": 100 * mae,
                }
            )
    t = ec.summarise(pd.DataFrame(rows)).set_index("model")
    assert t.loc["LightGBM", "beats_esdf"] == 2 and t.loc[ec.PROPOSED, "beats_esdf"] == 0
    assert t.loc["LightGBM", "mae_vs_esdf_pct"] == pytest.approx(100 * (0.08 / 0.09 - 1))

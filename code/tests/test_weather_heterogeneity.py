"""Heterogeneity of the weather effects: random-effects decomposition, shrinkage, design helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import weather_heterogeneity as wh


def test_random_effects_recovers_the_true_spread():
    rng = np.random.default_rng(0)
    true = rng.normal(0.012, 0.004, 200)
    se = np.full(200, 0.002)
    obs = true + rng.normal(0, se)
    re = wh.random_effects(pd.Series(obs), pd.Series(se))
    assert re["pooled"] == pytest.approx(0.012, abs=0.001)
    assert re["tau"] == pytest.approx(0.004, abs=0.001)
    assert re["i2"] > 0.7


def test_random_effects_with_no_true_spread_gives_small_tau():
    rng = np.random.default_rng(1)
    se = np.full(200, 0.003)
    obs = 0.01 + rng.normal(0, se)
    re = wh.random_effects(pd.Series(obs), pd.Series(se))
    assert re["tau"] < 0.0015
    assert re["i2"] < 0.3


def test_shrinkage_pulls_noisy_sites_towards_the_pool():
    sl = pd.DataFrame(
        {
            "site_code": ["precise", "noisy"],
            "dose": "hdd14",
            "slope": [0.010, 0.040],
            "se": [0.001, 0.030],
        }
    )
    # Add a few more precise sites so tau is estimable.
    extra = pd.DataFrame(
        {
            "site_code": [f"p{i}" for i in range(8)],
            "dose": "hdd14",
            "slope": 0.010 + np.linspace(-0.002, 0.002, 8),
            "se": 0.001,
        }
    )
    out = wh.shrink(pd.concat([sl, extra], ignore_index=True)).set_index("site_code")
    assert out.loc["noisy", "shrinkage"] > out.loc["precise", "shrinkage"]
    assert abs(out.loc["noisy", "posterior"] - out.loc["noisy", "pooled"]) < abs(
        0.040 - out.loc["noisy", "pooled"]
    )
    assert out.loc["precise", "posterior"] == pytest.approx(0.010, abs=0.001)


def test_categorical_design_keeps_the_index_and_drops_the_reference():
    idx = pd.Index([10, 20, 30, 40])
    d = wh._categorical_design(pd.Series(["a", "b", "a", "c"], index=idx), ["a", "b", "c"])
    assert d.index.tolist() == [10, 20, 30, 40]
    assert d.columns.tolist() == ["b", "c"]
    assert d["b"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_interactions_multiply_each_dose_by_each_column():
    sub = pd.DataFrame({"hdd14": [1.0, 2.0], "cdd20": [0.0, 3.0]}, index=[5, 6])
    design = pd.DataFrame({"m": [1.0, 0.5]}, index=[5, 6])
    inter, names = wh._interactions(sub, design)
    assert names == ["hdd14_x_m", "cdd20_x_m"]
    assert inter["hdd14_x_m"].tolist() == [1.0, 1.0]
    assert inter["cdd20_x_m"].tolist() == [0.0, 1.5]
    assert inter.index.tolist() == [5, 6]

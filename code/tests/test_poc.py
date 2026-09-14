"""Origins, baselines and scoring of the day-ahead PoC on a synthetic panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.experiments import poc

CFG = {
    "issue_time_local": "16:30",
    "timezone": "Europe/London",
    "data_lag_periods": 0,
    "quantiles": [0.1, 0.5, 0.9],
    "profile_weeks": 4,
}


def _panel(days: int = 70, sites=("S1", "S2")) -> poc.Panel:
    index = pd.date_range("2025-01-01", periods=days * 48, freq="30min", tz="UTC")
    local = index.tz_convert("Europe/London")
    tod = (local.hour * 2 + local.minute // 30).to_numpy()
    rng = np.random.default_rng(0)
    data = {
        s: 100 * (1 + i) + 30 * np.sin(2 * np.pi * tod / 48) + rng.normal(0, 1, len(index))
        for i, s in enumerate(sites)
    }
    y = pd.DataFrame(data, index=index)
    panel = poc.Panel(
        y_raw=y,
        y_imp=y,
        index=index,
        local=local,
        local_date=local.tz_localize(None).normalize().to_numpy().astype("datetime64[D]"),
        tod=tod,
        dow=local.dayofweek.to_numpy(),
        sites=list(sites),
    )
    return panel


def test_origins_sit_before_targets_with_day_ahead_leads():
    panel = _panel()
    origins = poc.make_origins(panel, CFG, "2025-02-10", "2025-02-20", 5)
    assert len(origins) == 3
    o = origins[0]
    assert o.origin_time == pd.Timestamp("2025-02-09 16:30", tz="Europe/London").tz_convert("UTC")
    assert len(o.target_pos) == 48 and o.target_pos[0] > o.origin_pos
    assert o.lead_hours.min() == pytest.approx(7.5) and o.lead_hours.max() == pytest.approx(31.0)


def test_data_lag_moves_the_origin_back():
    panel = _panel()
    a = poc.make_origins(panel, CFG, "2025-02-10", "2025-02-10", 1)[0]
    b = poc.make_origins(panel, {**CFG, "data_lag_periods": 48}, "2025-02-10", "2025-02-10", 1)[0]
    assert a.origin_pos - b.origin_pos == 48


def test_seasonal_naive_is_the_value_one_week_earlier():
    panel = _panel()
    o = poc.make_origins(panel, CFG, "2025-02-10", "2025-02-10", 1)[0]
    f = poc.seasonal_naive(panel, o, "S1")
    assert np.allclose(f, panel.y_imp["S1"].to_numpy()[o.target_pos - 336])


def test_profile_quantiles_are_monotone_and_bracket_the_signal():
    panel = _panel()
    o = poc.make_origins(panel, CFG, "2025-02-10", "2025-02-10", 1)[0]
    q = poc.profile_quantiles(panel, o, "S1", 4, [0.1, 0.5, 0.9])
    assert q.shape == (48, 3)
    assert (np.diff(q, axis=1) >= 0).all()
    truth = panel.y_raw["S1"].to_numpy()[o.target_pos]
    assert np.abs(q[:, 1] - truth).mean() < 3


def test_score_recovers_perfect_and_biased_forecasts():
    panel = _panel()
    origins = poc.make_origins(panel, CFG, "2025-02-10", "2025-02-20", 5)
    frames = []
    for o in origins:
        for s in panel.sites:
            truth = panel.y_raw[s].to_numpy()[o.target_pos]
            frames.append(
                poc.forecast_frame(
                    "seasonal_naive",
                    s,
                    o,
                    panel,
                    CFG["quantiles"],
                    np.stack([truth - 5, truth, truth + 5], axis=1),
                )
            )
            frames.append(
                poc.forecast_frame(
                    "biased",
                    s,
                    o,
                    panel,
                    CFG["quantiles"],
                    np.stack([truth + 5, truth + 10, truth + 15], axis=1),
                )
            )
    per, pooled, by_lead = poc.score(pd.concat(frames), panel, CFG["quantiles"])
    p = pooled.set_index("model")
    assert p.loc["seasonal_naive", "mae"] == pytest.approx(0.0, abs=1e-3) and p.loc[
        "seasonal_naive", "coverage_80"
    ] == pytest.approx(1.0)
    assert p.loc["biased", "mae"] == pytest.approx(10.0, abs=1e-3) and p.loc[
        "biased", "coverage_80"
    ] == pytest.approx(0.0)
    assert p.loc["biased", "mae_skill_vs_naive_median"] < 0
    assert set(by_lead["lead_bucket"]) <= {6, 12, 18, 24, 30}

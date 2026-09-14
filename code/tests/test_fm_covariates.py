"""Past/future covariate windows for the foundation models."""

from __future__ import annotations

import numpy as np

from sense_energy.experiments import fm_covariates as fmc
from sense_energy.experiments import poc
from test_poc import CFG, _panel


def test_clean_interpolates_gaps_and_zeros_all_missing():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(fmc.clean(x), [1.0, 1.0, 2.0, 3.0, 3.0])
    assert np.array_equal(fmc.clean(np.full(4, np.nan)), np.zeros(4))
    assert fmc.clean(x).dtype == np.float32


def test_standardise_uses_the_past_statistics_for_the_future():
    past = {"a": np.array([1.0, 2.0, 3.0]), "b": np.ones(3)}
    future = {"a": np.array([4.0]), "b": np.array([1.0])}
    p, f = fmc.standardise(past, future)
    assert np.isclose(p["a"].mean(), 0) and np.isclose(p["a"].std(), 1)
    assert np.isclose(f["a"][0], (4 - 2) / np.std([1, 2, 3]))
    assert np.array_equal(p["b"], np.zeros(3)) and f["b"][0] == 0  # constant: spread 1


class _Cov:
    """Stand-in for Covariates with an IFS block that is missing entirely."""

    def __init__(self, T: int):
        self.T = T

    def ifs_block(self, site, o, targets_ns):
        return {k: np.full(len(targets_ns), np.nan) for k in fmc.WEATHER_TO_IFS.values()}


def _windows(panel: poc.Panel, weather: str) -> fmc.CovariateWindows:
    cw = fmc.CovariateWindows.__new__(fmc.CovariateWindows)
    T, S = len(panel.index), len(panel.sites)
    cw.panel, cw.weather = panel, weather
    cw.cov = _Cov(T)
    cw.weather_vars = ["t2m"]
    cw.shared = {"hod_sin": np.sin(2 * np.pi * panel.tod / 48)}
    cw.per_site = {"t2m": np.tile(np.arange(T, dtype=float)[:, None], (1, S))}  # = position
    cw.names = ["hod_sin", "t2m"]
    cw.n_ifs_missing = cw.n_windows = 0
    return cw


def test_windows_align_to_the_origin_and_fall_back_to_persistence():
    panel = _panel(days=20)
    o = poc.make_origins(panel, CFG, "2025-01-15", "2025-01-15", 1)[0]
    L, H = 96, int(o.target_pos[-1] - o.origin_pos)
    past, future = _windows(panel, "era5").window(o, 1, "S2", L, H)
    assert past["t2m"].shape == (L,) and future["t2m"].shape == (H,)
    assert past["t2m"][-1] == o.origin_pos and future["t2m"][0] == o.origin_pos + 1
    assert future["t2m"][-1] == o.target_pos[-1]
    cw = _windows(panel, "ifs")
    _, future_ifs = cw.window(o, 1, "S2", L, H)
    assert cw.n_ifs_missing == 1
    # persistence: the same half-hour one day earlier for leads <= 24 h, two days for longer
    lead = np.arange(1, H + 1)
    expect = (o.origin_pos + lead) - 48 * np.ceil(lead / 48)
    assert np.array_equal(future_ifs["t2m"], expect.astype("float32"))
    assert (expect <= o.origin_pos).all()


def test_windows_clip_at_the_panel_edges():
    panel = _panel(days=20)
    o = poc.make_origins(panel, CFG, "2025-01-02", "2025-01-02", 1)[0]
    past, _ = _windows(panel, "era5").window(o, 0, "S1", 48 * 5, 63)
    assert past["t2m"].shape == (240,) and past["t2m"][0] == 0  # left edge held

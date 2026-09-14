"""Reconciliation algebra on a tiny hierarchy: coherence, bottom-up, top-down, OLS/WLS/MinT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.experiments import reconciliation as rc


def _tiny() -> rc.Hierarchy:
    # national <- T1 (m1, m2), T2 (m3); regions and sites coincide with trusts / meters here
    nodes = ["national", "T1", "T2", "m1", "m2", "m3"]
    bottom = ["m1", "m2", "m3"]
    S = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    idx = pd.date_range("2025-01-01", periods=48 * 40, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    b = pd.DataFrame(
        {
            "m1": 10 + rng.normal(0, 1, len(idx)),
            "m2": 30 + rng.normal(0, 1, len(idx)),
            "m3": 60 + rng.normal(0, 1, len(idx)),
        },
        index=idx,
    )
    y = pd.DataFrame({"national": b.sum(axis=1), "T1": b["m1"] + b["m2"], "T2": b["m3"], **b})[
        nodes
    ]
    level_of = {
        "national": "total",
        "T1": "trust",
        "T2": "trust",
        "m1": "meter",
        "m2": "meter",
        "m3": "meter",
    }
    return rc.Hierarchy(
        nodes, level_of, bottom, S, y, y.copy(), {"m1": "T1", "m2": "T1", "m3": "T2"}
    )


def _coherent(R: np.ndarray, h: rc.Hierarchy) -> bool:
    bottom_idx = [h.nodes.index(b) for b in h.bottom]
    return np.allclose(R, R[:, bottom_idx] @ h.S.T, atol=1e-8)


def test_projections_are_coherent_and_ols_equals_mint_with_identity_covariance():
    h = _tiny()
    rng = np.random.default_rng(1)
    Yhat = np.array([[100, 40, 60, 10, 30, 60]], dtype=float) + rng.normal(0, 3, (5, 6))
    shares = rc.recent_shares(h, 48 * 30)
    assert np.isclose(shares.sum(), 1.0) and shares[2] > shares[1] > shares[0]
    past = rng.normal(0, 1, (40, 6))
    out = {m: rc.reconcile_origin(m, Yhat, h, shares, past) for m in rc.METHODS}
    for m in rc.METHODS[1:]:
        assert _coherent(out[m], h), m
    assert not _coherent(out["base"], h)
    assert np.allclose(out["bottom_up"][:, 0], Yhat[:, 3:].sum(axis=1))
    assert np.allclose(out["top_down"][:, 0], Yhat[:, 0])  # the national forecast is kept
    assert np.allclose(out["middle_out"][:, 1:3], Yhat[:, 1:3])  # trust forecasts are kept
    G_ols = rc.projection(h.S)
    G_id = rc.projection(h.S, np.eye(6))
    assert np.allclose(G_ols, G_id)
    assert np.allclose(h.S @ G_ols @ h.S, h.S)  # S G is a projection onto the coherent space


def test_variance_methods_need_enough_past_errors_and_shrinkage_is_bounded():
    h = _tiny()
    Yhat = np.ones((2, 6))
    assert np.isnan(rc.reconcile_origin("wls_var", Yhat, h, rc.recent_shares(h, 100), None)).all()
    R = np.random.default_rng(2).normal(0, 1, (30, 6)) * np.array([10, 4, 6, 1, 3, 6])
    W, lam = rc._shrink_cov(R)
    assert 0.0 <= lam <= 1.0 and np.allclose(W, W.T)
    assert np.all(np.linalg.eigvalsh(W) > 0)

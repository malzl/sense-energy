"""JEPA on the demand panel: day tokens, masks, windows, collapse term, a short CPU training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.experiments import jepa


def _wide(n_days=120, sites=("A", "B"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days * 48, freq="30min", tz="UTC")
    local_hour = idx.tz_convert("Europe/London").hour.to_numpy()
    base = 10 + 5 * np.sin(2 * np.pi * (local_hour - 6) / 24)
    w = pd.DataFrame(
        {s: base * (1 + 0.1 * k) + rng.normal(0, 0.3, len(idx)) for k, s in enumerate(sites)},
        index=idx,
    )
    return w


def test_day_tokens_scale_by_trailing_mean_and_flag_incomplete_days():
    w = _wide()
    w.iloc[48 * 50 : 48 * 50 + 10, 0] = np.nan  # ten missing half-hours on day 50 of site A
    w.iloc[48 * 60 + 5, 1] = np.nan  # one missing half-hour on day 60 of site B: still valid
    tok = jepa.day_tokens(w)
    assert tok["x"].shape == (2, len(tok["dates"]), 48)
    a, b = tok["sites"].index("A"), tok["sites"].index("B")
    assert not tok["valid"][a, 50] and tok["valid"][b, 60]
    assert not tok["valid"][:, :7].any()  # no trailing mean yet
    valid_tokens = tok["x"][tok["valid"]]
    assert valid_tokens.mean() == pytest.approx(1.0, abs=0.1)  # scaled by the site's recent mean
    assert np.isfinite(tok["x"]).all() and tok["x"].min() >= 0


def test_sample_mask_has_a_block_and_the_right_count():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = jepa.sample_mask(rng)
        assert m.sum() == jepa.N_TARGET
        runs = np.diff(np.where(np.diff(np.r_[0, m.astype(int), 0]) != 0)[0])[::2]
        assert runs.max() >= 4


def test_windows_require_every_day_valid():
    tok = {"valid": np.array([[True] * 30 + [False] + [True] * 40])}
    ws = jepa.windows(tok, window=28)
    starts = [d for _, d in ws]
    assert 0 in starts and 2 in starts and 3 not in starts  # a window over day 30 is invalid
    assert max(starts) == 71 - 28


def test_variance_term_penalises_collapse():
    import torch

    collapsed = torch.zeros(64, 8)
    spread = torch.randn(64, 8) * 2
    assert jepa.variance_term(collapsed).item() > 0.9
    assert jepa.variance_term(spread).item() < 0.1


def test_short_training_runs_and_exports_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(jepa, "OUT_DIR", tmp_path)
    tok = jepa.day_tokens(_wide(n_days=150))
    res = jepa.train(
        {
            "steps": 2,
            "batch_size": 8,
            "warmup": 1,
            "d_model": 16,
            "n_heads": 2,
            "enc_layers": 1,
            "pred_layers": 1,
        },
        device="cpu",
        tokens=tok,
    )
    assert np.isfinite(res["log"]["loss"]).all()
    days = jepa.embed_days(res["model"], tok, "cpu", batch=32)
    assert set(days["site_code"]) == {"A", "B"} and days.shape[1] == 2 + 16
    sites = jepa.site_embeddings(days)
    assert list(sites.index) == ["A", "B"] and "days" in sites
    sp = jepa.surprise(res["model"], tok, "cpu", batch=32)
    assert (sp["surprise"] >= 0).all() and len(sp) > 0
    assert (tmp_path / "jepa.pt").exists()

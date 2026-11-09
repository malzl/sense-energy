"""Latent PFN: task sampling, model shapes, a two-step training on synthetic latent tokens."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.experiments import jepa
from sense_energy.experiments import latent_pfn as lp


def _tokens(n_sites=3, n_days=400, d=8, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days)
    k = len(jepa.WEATHER_VARS)
    wx = rng.normal(size=(n_sites, n_days, k)).astype(np.float32)
    e = np.zeros((n_sites, n_days, d), dtype=np.float32)
    e[:, :, :k] = 0.5 * wx  # the embedding carries the weather linearly
    e += rng.normal(0, 0.05, e.shape).astype(np.float32)
    x = np.clip(
        1 + 0.2 * e[:, :, :1] + rng.normal(0, 0.05, (n_sites, n_days, jepa.SLOTS)), 0, None
    ).astype(np.float32)
    has = np.ones((n_sites, n_days), dtype=bool)
    has[0, 100:120] = False
    return {
        "sites": [f"S{i}" for i in range(n_sites)],
        "dates": dates,
        "e": e,
        "has": has,
        "valid": has.copy(),
        "x": x,
        "wx": wx,
        "wx_mean": np.zeros(k, np.float32),
        "wx_std": np.ones(k, np.float32),
        "cal": jepa.calendar_features(dates.dayofweek.to_numpy(), dates.dayofyear.to_numpy()),
        "scale": np.full((n_sites, n_days), 10.0, dtype=np.float32),
    }


def test_sample_task_respects_order_and_gaps():
    tok = _tokens()
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = lp.sample_task(tok, rng, 0, train_end_idx=399)
        assert t is not None
        c, q = t
        assert len(c) == lp.CONTEXT_DAYS and len(q) == lp.N_QUERY
        assert c.max() < q.min()  # context strictly precedes the queries
        assert (
            q.min() - c.min() <= lp.LOOKBACK_DAYS + lp.N_QUERY + 20
        )  # 20 missing days may stretch it
        assert not set(c) & set(range(100, 120))  # missing days are never context


def test_models_have_the_right_shapes():
    import torch

    cfg = {**lp.DEFAULT, "d_model": 32, "layers": 1}
    pfn = lp.build_pfn(cfg, d_latent=8, n_side=16)
    out = pfn(torch.zeros(2, 10, 8), torch.zeros(2, 10, 16), torch.zeros(2, 3, 16))
    assert out.shape == (2, 3, 8)
    dec = lp.build_decoder(8, 16)
    assert dec(torch.zeros(5, 8)).shape == (5, jepa.SLOTS)


def test_short_training_learns_a_linear_latent_map(tmp_path):
    import torch

    tok = _tokens(n_days=300)
    res = lp.train(
        {
            "steps": 60,
            "decoder_steps": 50,
            "batch_tasks": 16,
            "d_model": 32,
            "layers": 1,
            "warmup": 5,
        },
        device="cpu",
        tokens=tok,
        out_dir=tmp_path,
    )
    assert (tmp_path / "pfn.pt").exists() and (tmp_path / "decoder.pt").exists()
    rng = np.random.default_rng(2)
    c, q = lp.sample_task(tok, rng, 1, 299)
    ce = torch.tensor(tok["e"][1, c])[None]
    cs = torch.tensor(lp._side(tok, 1, c, int(q[0])))[None]
    qs = torch.tensor(lp._side(tok, 1, q, int(q[0])))[None]
    with torch.no_grad():
        pred = res["pfn"](ce, cs, qs)[0].numpy()
    truth = tok["e"][1, q]
    assert (
        np.corrcoef(pred[:, 0], truth[:, 0])[0, 1] > 0.5
        or np.abs(pred - truth).mean() < np.abs(truth).mean()
    )


def test_poc_site_days_are_local_dates():
    try:
        d = lp.poc_site_days()
    except FileNotFoundError:
        pytest.skip("no proof-of-concept forecasts on disk")
    assert {"site_code", "date"} <= set(d.columns) and d["date"].dt.hour.eq(0).all()

"""Shape features and k-means clustering on a synthetic half-hourly panel."""

import numpy as np
import pandas as pd

from sense_energy.analysis import clustering as cl


def _wide(days: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=days * 48, freq="30min", tz="UTC")
    tod = (
        idx.tz_convert("Europe/London").hour * 2 + idx.tz_convert("Europe/London").minute // 30
    ).to_numpy()
    rng = np.random.default_rng(0)
    flat = 100 + 5 * np.sin(2 * np.pi * tod / 48)
    office = 40 + 60 * (np.abs(tod - 24) < 8)
    cols = {f"F{i}": flat * (1 + 0.1 * i) + rng.normal(0, 1, len(idx)) for i in range(4)}
    cols |= {f"O{i}": office * (1 + 0.1 * i) + rng.normal(0, 1, len(idx)) for i in range(4)}
    return pd.DataFrame(cols, index=idx)


def test_shape_features_are_normalised_profiles():
    feats = cl.shape_features(_wide(), min_days=100)
    assert feats.shape == (8, 97) and list(feats.columns[:2]) == ["wd_00", "wd_01"]
    wd = feats[[f"wd_{h:02d}" for h in range(48)]]
    assert np.allclose(wd.mean(axis=1), 1.0, atol=0.05)  # series mean = 1
    assert feats.loc["O0", "wd_24"] > 1.5 and feats.loc["F0", "wd_24"] < 1.1


def test_kmeans_separates_the_two_shapes():
    feats = cl.shape_features(_wide(), min_days=100)
    labels, sil, _ = cl.cluster(feats[cl.PROFILE_COLS], range(2, 4))
    best = labels["k2"]
    flat = best[[c for c in best.index if c.startswith("F")]]
    office = best[[c for c in best.index if c.startswith("O")]]
    assert (flat == flat.iloc[0]).all() and (office == office.iloc[0]).all()
    assert best["F0"] != best["O0"]
    assert sil.loc[sil["k"] == 2, "silhouette"].iloc[0] > 0.8


def test_activity_name_normalisation():
    from sense_energy.data.nhs_activity import _norm

    assert (
        _norm("Birmingham Women's and Children's NHS Foundation Trust")
        == "BIRMINGHAM WOMENS AND CHILDRENS NHS FOUNDATION TRUST"
    )

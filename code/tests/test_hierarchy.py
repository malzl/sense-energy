"""Aggregate levels of the hierarchy and bottom-up sums on a synthetic site panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.experiments import hierarchy, poc
from test_poc import CFG, _panel

SEL = {
    **CFG,
    "train_start": "2025-01-01",
    "train_end": "2025-02-09",
    "test_start": "2025-02-10",
    "test_end": "2025-03-01",
    "min_coverage_train": 0.5,
    "min_coverage_test": 0.5,
    "min_history_days": 10,
}


def _site_panel() -> poc.Panel:
    panel = _panel(days=70, sites=("S1", "S2", "S3"))
    panel.y_imp = panel.y_raw.copy()  # _panel shares one frame; the raw gap must not leak
    panel.site_meta = pd.DataFrame(
        {
            "trust_id": ["T1", "T1", "T2"],
            "commissioning_region": ["EAST OF ENGLAND COMMISSIONING REGION"] * 2
            + ["LONDON COMMISSIONING REGION"],
            "organisation_name": ["Trust one", "Trust one", "Trust two"],
            "organisation_type": ["COMMUNITY", "COMMUNITY", "ACUTE - LARGE"],
            "n_meters": [1, 2, 1],
        },
        index=pd.Index(["S1", "S2", "S3"], name="series_id"),
    )
    panel.members = {s: {s: 1.0} for s in panel.sites}
    return panel


def test_region_id_strips_the_suffix():
    assert hierarchy.region_id("EAST OF ENGLAND COMMISSIONING REGION") == "east_of_england"
    assert hierarchy.region_id("North West Commissioning Region") == "north_west"


def test_aggregate_panel_sums_members_and_needs_all_present():
    site = _site_panel()
    site.y_raw.iloc[10, 0] = np.nan  # S1 unobserved at one period
    trust = hierarchy.aggregate_panel(site, "trust", SEL)
    assert trust.level == "trust" and trust.sites == ["T1", "T2"]
    expect = site.y_imp["S1"] + site.y_imp["S2"]
    assert np.allclose(trust.y_imp["T1"], expect)
    assert np.isnan(trust.y_raw["T1"].iloc[10]) and not np.isnan(trust.y_imp["T1"].iloc[10])
    assert set(trust.members["T1"]) == {"S1", "S2"}
    assert abs(sum(trust.members["T1"].values()) - 1.0) < 1e-9
    assert trust.members["T1"]["S2"] > trust.members["T1"]["S1"]  # S2 has the larger mean
    assert trust.site_meta.loc["T1", "n_sites"] == 2 and trust.site_meta.loc["T1", "n_meters"] == 3
    region = hierarchy.aggregate_panel(site, "region", SEL)
    assert region.sites == ["east_of_england", "london"]
    total = hierarchy.aggregate_panel(site, "total", SEL)
    assert total.sites == ["national"] and total.site_meta.loc["national", "n_sites"] == 3


def test_bottom_up_sums_points_only_for_complete_groups(tmp_path, monkeypatch):
    site = _site_panel()
    origins = poc.make_origins(site, SEL, "2025-02-10", "2025-02-15", 5)
    frames = []
    for o in origins:
        for s in site.sites:
            truth = site.y_raw[s].to_numpy()[o.target_pos]
            frames.append(
                poc.forecast_frame("m", s, o, site, SEL["quantiles"], np.stack([truth] * 3, axis=1))
            )
    lower = pd.concat(frames, ignore_index=True)
    lower = lower[~((lower["site_code"] == "S3") & (lower["origin"] == origins[1].origin_time))]
    cfg = {**SEL, "energy": "elec", "outputs_dir": str(tmp_path / "poc")}
    (tmp_path / "poc").mkdir()
    lower.to_parquet(tmp_path / "poc" / "forecasts_m.parquet", index=False)
    monkeypatch.setattr(poc, "PROJECT_ROOT", tmp_path.parent)
    monkeypatch.setattr(
        hierarchy, "load_panel", lambda c: hierarchy.aggregate_panel(site, c["level"], c)
    )
    out = hierarchy.bottom_up(cfg, "trust")
    assert set(out["model"]) == {"bu_m"}
    t1 = out[(out["site_code"] == "T1")]
    assert len(t1) == 2 * 48  # both origins complete for T1
    t2 = out[(out["site_code"] == "T2")]
    assert (
        len(t2) == 48 and (t2["origin"] == origins[0].origin_time).all()
    )  # second origin incomplete
    o = origins[0]
    got = t1[t1["origin"] == o.origin_time].sort_values("target")["point"].to_numpy()
    expect = (site.y_raw["S1"] + site.y_raw["S2"]).to_numpy()[o.target_pos]
    assert np.allclose(got, expect, atol=1e-3)
    assert (t1["q10"] == t1["point"]).all()

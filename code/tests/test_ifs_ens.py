"""Pure parts of the IFS ENS mirror harvester."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sense_energy.data import ifs_ens


def test_index_key_replaces_the_extension():
    assert (
        ifs_ens.index_key("a/b/20240615000000-24h-enfo-ef.grib2")
        == "a/b/20240615000000-24h-enfo-ef.index"
    )


def test_select_messages_keeps_only_wanted_params_and_maps_control_to_member_zero():
    lines = [
        {"param": "2t", "number": "3", "_offset": 100, "_length": 50},
        {"param": "2t", "_offset": 0, "_length": 100},  # control run: no number
        {"param": "gh", "number": "3", "_offset": 150, "_length": 50},
        {"param": "tp", "number": "50", "_offset": 200, "_length": 60},
    ]
    text = "\n".join(json.dumps(x) for x in lines)
    out = ifs_ens.select_messages(text, ["2t", "tp"])
    assert out == [(100, 50, "2t", 3), (0, 100, "2t", 0), (200, 60, "tp", 50)]


def test_all_runs_and_missing_runs(tmp_path):
    cfg = {
        "start": "2024-01-01",
        "end": "2024-01-02",
        "run_times": ["00", "12"],
        "output_subdir": str(tmp_path),
    }
    runs = ifs_ens.all_runs(cfg)
    assert runs == [("20240101", "00"), ("20240101", "12"), ("20240102", "00"), ("20240102", "12")]
    (tmp_path / "sites").mkdir()
    (tmp_path / "sites" / "20240101T12z.parquet").write_bytes(b"x")
    assert ifs_ens.missing_runs(cfg) == [("20240101", "00"), ("20240102", "00"), ("20240102", "12")]


def test_extract_takes_nearest_grid_point_per_member_and_step():
    lat = np.array([55.0, 54.5, 54.0])
    lon = np.array([-1.0, -0.5, 0.0])
    members = np.array([0, 1])
    field = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3) + 273.0  # (member, lat, lon)
    payloads = {24: {"number": members, "latitude": lat, "longitude": lon, "vars": {"t2m": field}}}
    sites = pd.DataFrame({"site_code": ["S1"], "latitude": [54.4], "longitude": [-0.6]})
    out = ifs_ens._extract(payloads, sites, pd.Timestamp("2024-01-01", tz="UTC"), "0p25")
    assert len(out) == 2 and set(out["member"]) == {0, 1}
    # nearest of (54.5, -0.5) is row 1, col 1 -> flat index 4 within each member block
    assert out.loc[out["member"] == 0, "t2m"].item() == 273.0 + 4
    assert out.loc[out["member"] == 1, "t2m"].item() == 273.0 + 9 + 4
    assert (out["valid_time"] == pd.Timestamp("2024-01-02", tz="UTC")).all()

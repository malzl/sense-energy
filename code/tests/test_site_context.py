"""Site context: postcode normalisation, rural-urban classes, REPD matching, density join."""

from __future__ import annotations

import pandas as pd

from sense_energy.data import site_context as sc


def test_normalise_postcode_strips_spaces_and_rejects_junk():
    s = pd.Series(["e9 6sr", " M8  5RB ", "NAN", None, "SW1A1AA", "TOOLONGPOSTCODE"])
    out = sc.normalise_postcode(s)
    assert out.tolist()[:2] == ["E96SR", "M85RB"]
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])
    assert out.iloc[4] == "SW1A1AA"
    assert pd.isna(out.iloc[5])


def test_urban_class_collapses_the_ruc11_codes():
    out = sc.urban_class(pd.Series(["A1", "B1", "C1", "C2", "D1", "E1", "F2", None, "ZZ"]))
    assert out.tolist()[:7] == [
        "major conurbation",
        "minor conurbation",
        "city and town",
        "city and town",
        "rural",
        "rural",
        "rural",
    ]
    assert out.isna().tolist()[7:] == [True, True]
    assert set(sc.RUC11_CLASS.values()) == set(sc.URBAN_ORDER)


def test_parse_result_handles_missing_postcodes():
    found = sc.parse_result(
        "E96SR", {"lsoa": "Hackney 019A", "codes": {"lsoa21": "E01001729", "ruc11": "A1"}}
    )
    assert found["lsoa21"] == "E01001729" and found["ruc11"] == "A1"
    assert found["lsoa_name"] == "Hackney 019A"
    missing = sc.parse_result("XX11XX", None)
    assert missing["postcode_norm"] == "XX11XX" and missing["lsoa21"] is None


def test_repd_at_sites_separates_operational_from_planned():
    repd = pd.DataFrame(
        {
            "ref": [1, 2, 3, 4, 5],
            "scheme": ["a", "b", "c", "d", "e"],
            "technology": [sc.REPD_SOLAR, sc.REPD_SOLAR, sc.REPD_BATTERY, sc.REPD_SOLAR, "Wind"],
            "capacity_mw": [0.5, 0.3, 1.0, 2.0, 5.0],
            "status": [
                "Operational",
                "Awaiting Construction",
                "Under Construction",
                "Operational",
                "Operational",
            ],
            "operational": pd.to_datetime(["2024-06-01", None, None, "2026-09-01", "2020-01-01"]),
            "postcode_norm": ["E96SR", "E96SR", "E96SR", "M85RB", "M85RB"],
            "country": "England",
        }
    )
    postcodes = pd.Series(["E96SR", "M85RB", "NOTHERE", pd.NA], index=["S1", "S2", "S3", "S4"])
    out = sc.repd_at_sites(postcodes, repd, sample_end=pd.Timestamp("2026-05-08"))
    assert out.loc["S1", "repd_solar_schemes"] == 2
    assert out.loc["S1", "repd_solar_mw"] == 0.8
    assert out.loc["S1", "repd_solar_operational_mw"] == 0.5
    assert out.loc["S1", "repd_solar_status"] == "Operational"
    assert out.loc["S1", "repd_battery_mw"] == 1.0
    # Operational after the sample end does not count as operational within it.
    assert out.loc["S2", "repd_solar_schemes"] == 1
    assert pd.isna(out.loc["S2", "repd_solar_operational_mw"])
    assert out.loc["S3", "repd_solar_schemes"] == 0 and pd.isna(out.loc["S3", "repd_solar_mw"])
    assert out.loc["S4", "repd_solar_schemes"] == 0


def test_best_status_prefers_the_most_advanced():
    assert sc._best_status(pd.Series(["Application Refused", "Awaiting Construction"])) == (
        "Awaiting Construction"
    )
    assert sc._best_status(pd.Series(["Operational", "Under Construction"])) == "Operational"
    assert sc._best_status(pd.Series([None])) is None


def test_density_table_reads_the_nomis_layout(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(
        '"GEOGRAPHY_CODE","GEOGRAPHY_NAME","CELL_NAME","OBS_VALUE"\n'
        '"E01000001","City 001A","Usual residents per square kilometre",12345.6\n'
        '"E01000002","City 001B","Usual residents per square kilometre",100.0\n'
    )
    d = sc.density_table(p)
    assert d.loc["E01000001"] == 12345.6 and d.name == "density_km2"

"""Display Energy Certificates: flattening, latest-per-building, site aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.data import dec


def _record(number, address, env, ac, kw, fuel, area, nominated, uprn=None, elec=100.0):
    return {
        "certificate_number": number,
        "uprn": uprn,
        "postcode": "E9 6SR",
        "address_line_1": address,
        "property_type": "Clinic",
        "schema_type": "CEPC-8.0.0",
        "registration_date": nominated,
        "technical_information": {
            "floor_area": area,
            "main_heating_fuel": fuel,
            "building_environment": env,
        },
        "ac_questionnaire": {
            "ac_present": ac,
            "ac_rated_output": ({"ac_kw_rating": kw} if kw else {"ac_rating_unknown_flag": 1}),
        },
        "this_assessment": {"energy_rating": 110, "nominated_date": nominated},
        "or_energy_consumption": {
            "electricity": {"consumption": 1000, "estimate": 0},
            "gas": {"consumption": 2000, "estimate": 0},
            "oil": {"consumption": 500, "estimate": 0},
        },
        "dec_annual_energy_summary": {
            "annual_energy_use_electrical": elec,
            "annual_energy_use_fuel_thermal": 150,
        },
    }


def test_flatten_reads_nested_fields_and_sums_thermal_fuels():
    r = dec.flatten(
        _record(
            "1",
            "Ward A",
            "Air Conditioning",
            "Yes",
            250,
            "Natural Gas",
            3000,
            "2025-10-01",
            uprn=123,
        )
    )
    assert r["floor_area_m2"] == 3000 and r["building_environment"] == "Air Conditioning"
    assert r["ac_present"] == "Yes" and r["ac_rated_kw"] == 250
    assert (
        r["electricity_kwh"] == 1000
        and r["thermal_kwh"] == 2500
        and r["thermal_fuels"] == "gas;oil"
    )
    assert r["uprn"] == "123" and r["nominated_date"] == "2025-10-01"
    unknown = dec.flatten(
        _record(
            "2",
            "Ward B",
            "Heating and Natural Ventilation",
            "No",
            None,
            "Grid Supplied Electricity",
            500,
            "2024-10-01",
        )
    )
    assert np.isnan(unknown["ac_rated_kw"]) and unknown["uprn"] is None


def test_latest_per_building_keeps_one_certificate_per_building():
    certs = pd.DataFrame(
        [
            dec.flatten(
                _record(
                    "1",
                    "Ward A",
                    "Air Conditioning",
                    "Yes",
                    250,
                    "Natural Gas",
                    3000,
                    "2024-10-01",
                    uprn=123,
                )
            ),
            dec.flatten(
                _record(
                    "2",
                    "Ward A",
                    "Air Conditioning",
                    "Yes",
                    300,
                    "Natural Gas",
                    3000,
                    "2025-10-01",
                    uprn=123,
                )
            ),
            dec.flatten(
                _record(
                    "3",
                    "Ward A",
                    "Air Conditioning",
                    "Yes",
                    300,
                    "Natural Gas",
                    3000,
                    "2027-10-01",
                    uprn=123,
                )
            ),
            dec.flatten(
                _record(
                    "4",
                    "Clinic",
                    "Heating and Natural Ventilation",
                    "No",
                    None,
                    "Grid Supplied Electricity",
                    1000,
                    "2023-10-01",
                )
            ),
            dec.flatten(
                _record(
                    "5",
                    "clinic ",
                    "Heating and Natural Ventilation",
                    "No",
                    None,
                    "Grid Supplied Electricity",
                    1000,
                    "2025-10-01",
                )
            ),
        ]
    )
    for c in ("registration_date", "nominated_date"):
        certs[c] = pd.to_datetime(certs[c])
    certs["postcode_norm"] = "E96SR"
    latest = dec.latest_per_building(certs, until=pd.Timestamp("2026-05-08"))
    assert sorted(latest["certificate_number"]) == ["2", "5"]  # the 2027 one is after `until`
    stale = dec.latest_per_building(
        certs, until=pd.Timestamp("2024-12-31"), since=pd.Timestamp("2024-01-01")
    )
    assert sorted(stale["certificate_number"]) == ["1"]  # the clinic's latest (2023) is too old


def test_aggregate_sites_weights_by_floor_area():
    certs = pd.DataFrame(
        [
            dec.flatten(
                _record(
                    "1",
                    "Ward A",
                    "Air Conditioning",
                    "Yes",
                    300,
                    "Natural Gas",
                    3000,
                    "2025-10-01",
                    uprn=1,
                    elec=200,
                )
            ),
            dec.flatten(
                _record(
                    "2",
                    "Clinic",
                    "Heating and Natural Ventilation",
                    "No",
                    None,
                    "Grid Supplied Electricity",
                    1000,
                    "2025-10-01",
                    uprn=2,
                    elec=80,
                )
            ),
        ]
    )
    for c in ("registration_date", "nominated_date"):
        certs[c] = pd.to_datetime(certs[c])
    certs["postcode_norm"] = "E96SR"
    certs["environment"] = certs["building_environment"].map(dec.ENVIRONMENT_CLASS)
    certs["electric_heating"] = certs["main_heating_fuel"].isin(dec.ELECTRIC_FUELS)
    certs["ac_yes"] = certs["ac_present"].eq("Yes")
    latest = dec.latest_per_building(certs)
    sites = pd.DataFrame({"site_code": ["S1", "S2"], "postcode_norm": ["E96SR", "ZZ99ZZ"]})
    out = dec.aggregate_sites(latest, sites).set_index("site_code")
    assert out.loc["S1", "dec_buildings"] == 2 and out.loc["S1", "dec_floor_area_m2"] == 4000
    assert out.loc["S1", "dec_ac_share"] == 0.75 and out.loc["S1", "dec_cooled_share"] == 0.75
    assert out.loc["S1", "dec_electric_heating_share"] == 0.25
    assert out.loc["S1", "dec_ac_kw_m2"] == 0.1  # 300 kW over the 3,000 m2 that have it
    assert out.loc["S1", "dec_elec_kwh_m2"] == 170  # (200*3000 + 80*1000) / 4000
    assert out.loc["S1", "dec_main_environment"] == "air conditioning"
    assert out.loc["S2", "dec_buildings"] == 0 and np.isnan(out.loc["S2", "dec_ac_share"])


def test_themes_and_recommendation_aggregation():
    assert dec.themes_of("Review the air conditioning energy performance report") == ["cooling"]
    assert set(dec.themes_of("Consider installing building mounted photovoltaic panels")) == {
        "renewables"
    }
    assert dec.themes_of(
        "Seek to minimise simultaneous operation of heating and cooling systems"
    ) == ["cooling", "heating"]
    recs = pd.DataFrame(
        {
            "advisory_number": ["R1", "R1", "R2", "R9"],
            "dec_number": ["1", "1", "2", "9"],
            "postcode_norm": "E96SR",
            "payback": ["short", "long", "short", "short"],
            "code": ["X26", "AE4", "X11", "X1"],
            "recommendation": [
                "air conditioning report",
                "photovoltaic panels",
                "lighting maintenance",
                "unlinked",
            ],
            "co2_impact": "LOW",
            "themes": ["cooling", "renewables", "lighting", ""],
        }
    )
    latest = pd.DataFrame(
        {
            "postcode_norm": ["E96SR", "E96SR", "E96SR"],
            "building": ["b1", "b2", "b3"],
            "related_certificate_number": ["R1", "R2", None],
        }
    )
    out = dec.aggregate_recommendations(recs, latest).set_index("postcode_norm")
    assert out.loc["E96SR", "rec_buildings"] == 2 and out.loc["E96SR", "rec_total"] == 3
    assert out.loc["E96SR", "rec_short_payback"] == 2
    assert (
        out.loc["E96SR", "rec_cooling"] == 1
        and out.loc["E96SR", "rec_cooling_building_share"] == 0.5
    )
    assert out.loc["E96SR", "rec_lighting"] == 1 and out.loc["E96SR", "rec_fabric"] == 0

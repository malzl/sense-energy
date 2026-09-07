"""Offline tests for the geography module.

Nothing here touches the network: the download and geocoding paths are exercised
by ``sense-energy fetch-geo``, not by the test suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sense_energy.data import geo
from sense_energy.visualization import maps

ORGANISATION_TYPES = [
    "ACUTE - TEACHING",
    "ACUTE - LARGE",
    "ACUTE - MEDIUM",
    "ACUTE - SMALL",
    "ACUTE - SPECIALIST",
    "COMMUNITY",
    "MENTAL HEALTH AND LEARNING DISABILITY",
    "AMBULANCE",
]


def test_every_boundary_layer_is_fully_specified():
    for key, spec in geo.BOUNDARY_LAYERS.items():
        assert {"service", "description", "name_field"} <= set(spec), key


def test_boundary_services_are_not_no_coordinate_tables():
    """ONS '_NC' services are lookup tables with no geometry - useless for maps."""
    for key, spec in geo.BOUNDARY_LAYERS.items():
        assert not spec["service"].endswith("_NC"), key


def test_bulk_chunk_respects_the_api_limit():
    assert geo.BULK_CHUNK <= 100


def test_every_organisation_type_maps_to_a_category():
    """An unmapped type would silently fall into the community bucket."""
    for org_type in ORGANISATION_TYPES:
        assert org_type in maps.CATEGORY_MAP


def test_categories_stay_within_the_three_slot_cap():
    """Maps use the all-pairs pairlist, where only three categorical slots validate."""
    assert len(set(maps.CATEGORY_MAP.values())) == 3
    assert set(maps.CATEGORY_MAP.values()) == set(maps.SERIES)


def test_categorise_handles_unknown_types():
    out = maps.categorise(pd.Series(["ACUTE - TEACHING", "SOMETHING NEW", None]))
    assert out.iloc[0] == "Acute"
    assert out.iloc[1] == "Mental health & community"
    assert out.iloc[2] == "Mental health & community"


def test_area_scale_is_proportional_to_value():
    """Area, not radius, carries magnitude - otherwise big sites look far bigger."""
    import numpy as np

    scaled = maps._area_scale(np.array([0.0, 50.0, 100.0]))
    assert scaled[0] == pytest.approx(30.0)
    assert (scaled[2] - 30.0) == pytest.approx(2 * (scaled[1] - 30.0))

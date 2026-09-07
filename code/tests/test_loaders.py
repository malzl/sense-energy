"""Tests for the reading-type taxonomy that separates energy from reactive power."""

from __future__ import annotations

from sense_energy.data import loaders


def test_active_types_are_energy_in_kwh():
    for rt in loaders.ACTIVE_HH_READING_TYPES:
        _, quantity, _ = loaders.READING_TYPES[rt]
        assert quantity == "active_energy"


def test_reactive_power_is_excluded_from_active_types():
    """Types 8-11 are kVArh, not kWh - roughly 9.6M rows of the source."""
    for rt in ("8", "9", "10", "11"):
        assert loaders.READING_TYPES[rt][1] == "reactive"
        assert rt not in loaders.ACTIVE_HH_READING_TYPES


def test_non_halfhourly_and_export_types_are_excluded():
    for rt in ("4", "5", "14", "6"):
        assert rt not in loaders.ACTIVE_HH_READING_TYPES


def test_estimated_types_are_a_subset_of_active():
    assert set(loaders.ESTIMATED_READING_TYPES) < set(loaders.ACTIVE_HH_READING_TYPES)
    for rt in loaders.ESTIMATED_READING_TYPES:
        assert loaders.READING_TYPES[rt][2] is True

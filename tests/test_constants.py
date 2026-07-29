"""Tests for dart_mlci.constants module."""

from pathlib import Path

from dart_mlci.chip import load_chip_config
from dart_mlci.constants import (
    ARTIFACTS_DIR,
    CHAMBER_TYPE_NUMBERS,
    CHAMBER_TYPE_ORDER,
    DEFAULT_CHIP_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_STRUCTURE_LIBRARY_PATH,
)


def test_artifacts_dir_is_path():
    assert isinstance(ARTIFACTS_DIR, Path)


def test_default_model_path_is_path():
    assert isinstance(DEFAULT_MODEL_PATH, Path)
    assert DEFAULT_MODEL_PATH.suffix == ".pt"


def test_default_chip_config_path_is_path():
    assert isinstance(DEFAULT_CHIP_CONFIG_PATH, Path)
    assert DEFAULT_CHIP_CONFIG_PATH.suffix == ".json"


def test_default_structure_library_path_is_path():
    assert isinstance(DEFAULT_STRUCTURE_LIBRARY_PATH, Path)
    assert DEFAULT_STRUCTURE_LIBRARY_PATH.suffix == ".json"


def test_paths_are_under_artifacts_dir():
    assert str(DEFAULT_MODEL_PATH).startswith(str(ARTIFACTS_DIR))
    assert str(DEFAULT_CHIP_CONFIG_PATH).startswith(str(ARTIFACTS_DIR))
    assert str(DEFAULT_STRUCTURE_LIBRARY_PATH).startswith(str(ARTIFACTS_DIR))


def test_chamber_type_numbers_are_1_to_8_no_gaps():
    """Matches the 'SAK RoI dimensions' reference table — all 8 slots filled, no skips."""
    assert list(CHAMBER_TYPE_NUMBERS.values()) == list(range(1, 9))


def test_chamber_type_order_matches_sak_chip_config():
    """Names must exactly match dart_mlci's chip config chamber-type keys."""
    config = load_chip_config(DEFAULT_CHIP_CONFIG_PATH)
    assert set(CHAMBER_TYPE_ORDER) == set(config.chamber_types)


def test_mothermachine_types_are_4_and_8():
    """Regression guard: both mother-machine variants must be numbered, not skipped."""
    assert CHAMBER_TYPE_NUMBERS["Mothermachine-inner"] == 4
    assert CHAMBER_TYPE_NUMBERS["Mothermachine-2x-inner"] == 8

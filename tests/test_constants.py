"""Tests for dart_mlci.constants module."""

from pathlib import Path

from dart_mlci.constants import (
    ARTIFACTS_DIR,
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

"""Tests for create_structure_library factory function."""

import pytest

from dart_mlci.chip import ChipStructureLibrary, create_structure_library
from dart_mlci.constants import DEFAULT_CHIP_CONFIG_PATH, DEFAULT_STRUCTURE_LIBRARY_PATH


class TestCreateStructureLibrary:
    def test_with_chip_config(self):
        lib = create_structure_library(
            pixel_size=0.065789, chip_config_path=DEFAULT_CHIP_CONFIG_PATH
        )
        assert isinstance(lib, ChipStructureLibrary)

    def test_with_legacy_path_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="legacy"):
            create_structure_library(
                pixel_size=0.065789,
                structure_library_path=DEFAULT_STRUCTURE_LIBRARY_PATH,
            )

    def test_with_no_args_uses_default_legacy(self):
        with pytest.warns(DeprecationWarning):
            lib = create_structure_library(pixel_size=0.065789)
        assert hasattr(lib, "polygon_library")

    def test_chip_config_takes_precedence(self):
        lib = create_structure_library(
            pixel_size=0.065789,
            chip_config_path=DEFAULT_CHIP_CONFIG_PATH,
            structure_library_path=DEFAULT_STRUCTURE_LIBRARY_PATH,
        )
        assert isinstance(lib, ChipStructureLibrary)

    def test_custom_pixel_size(self):
        lib = create_structure_library(
            chip_config_path=DEFAULT_CHIP_CONFIG_PATH,
            pixel_size=0.1,
        )
        assert lib.pixel_size == 0.1

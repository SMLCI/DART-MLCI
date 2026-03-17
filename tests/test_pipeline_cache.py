"""Tests for ChamberPipelineCache in dart_mlci.pipeline module."""

import warnings

import pytest

from dart_mlci.constants import DEFAULT_STRUCTURE_LIBRARY_PATH
from dart_mlci.pipeline import ChamberPipelineCache


@pytest.fixture
def structure_library():
    """Load the legacy SAK structure library for testing."""
    from dart_mlci.mask import SAKRoIStructureLibrary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SAKRoIStructureLibrary(
            lookup_path=DEFAULT_STRUCTURE_LIBRARY_PATH,
            pixel_size=0.065789,
        )


class TestChamberPipelineCache:
    def test_get_returns_components(self, structure_library):
        cache = ChamberPipelineCache(structure_library)
        components = cache.get("NormaleBox-inner")
        assert "roi_polygon" in components
        assert "marker_group" in components
        assert "matching_step" in components
        assert "rotation_step" in components
        assert "masking_step" in components

    def test_caching_returns_same_instance(self, structure_library):
        cache = ChamberPipelineCache(structure_library)
        c1 = cache.get("NormaleBox-inner")
        c2 = cache.get("NormaleBox-inner")
        assert c1 is c2

    def test_invalid_name_raises_key_error(self, structure_library):
        cache = ChamberPipelineCache(structure_library)
        with pytest.raises(KeyError, match="Unknown structure name"):
            cache.get("NonExistentChamber")

    def test_different_types_cached_separately(self, structure_library):
        cache = ChamberPipelineCache(structure_library)
        c1 = cache.get("NormaleBox-inner")
        c2 = cache.get("BigBox-inner")
        assert c1 is not c2

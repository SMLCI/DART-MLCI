"""Tests for the unified chip configuration module."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np

import dmc_masking
from dmc_masking.chip import ChipConfig, ChipStructureLibrary, load_chip_config
from dmc_masking.map import Map
from dmc_masking.mask import SAKRoIStructureLibrary

# Paths to artifacts
ARTIFACTS_DIR = Path(dmc_masking.__file__).parent.parent / "artifacts"
SAK_CONFIG_PATH = ARTIFACTS_DIR / "chips" / "sak.json"
CHAMBER_STRUCTURE_PATH = ARTIFACTS_DIR / "chamber_structure.json"
BLUEPRINT_MAP_PATH = ARTIFACTS_DIR / "sak_blueprint_map.csv"


class TestLoadChipConfig(unittest.TestCase):
    """Tests for load_chip_config function."""

    def test_load_sak_config(self):
        """Test loading the SAK chip config."""
        config = load_chip_config(SAK_CONFIG_PATH)

        self.assertEqual(config.chip_name, "SAK")
        self.assertEqual(config.version, "2.0")
        self.assertEqual(config.pixel_size, 0.065789)
        self.assertEqual(len(config.chamber_types), 8)
        self.assertEqual(len(config.blueprint_map), 1164)

    def test_all_chamber_types_present(self):
        """Test that all expected chamber types are in the config."""
        config = load_chip_config(SAK_CONFIG_PATH)

        expected_types = [
            "NormaleBox-inner",
            "BigBox-inner",
            "OpenBox-inner",
            "Mothermachine-inner",
            "NormaleBox-pillar-inner",
            "BigBox-pillar-inner",
            "OpenBox-collector-inner",
            "Mothermachine-2x-inner",
        ]
        for ct in expected_types:
            self.assertIn(ct, config.chamber_types)

    def test_chamber_type_has_required_fields(self):
        """Test that each chamber type has polygon and markers."""
        config = load_chip_config(SAK_CONFIG_PATH)

        for name, ct in config.chamber_types.items():
            self.assertIsInstance(ct.polygon, dict, f"{name} polygon should be dict")
            self.assertIn("type", ct.polygon, f"{name} polygon missing 'type'")
            self.assertIn("coordinates", ct.polygon, f"{name} polygon missing 'coordinates'")
            self.assertIsInstance(ct.markers, dict, f"{name} markers should be dict")
            self.assertIn("cross", ct.markers, f"{name} missing 'cross' marker")
            self.assertIn("circle", ct.markers, f"{name} missing 'circle' marker")

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_chip_config("/nonexistent/path.json")

    def test_missing_required_fields(self):
        """Test validation rejects config missing required fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"chip_name": "test"}, f)
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                load_chip_config(f.name)
            self.assertIn("missing required fields", str(ctx.exception))

    def test_missing_structure_type_in_blueprint(self):
        """Test validation rejects blueprint entry without structure_type."""
        config_data = {
            "chip_name": "test",
            "version": "2.0",
            "pixel_size": 0.065789,
            "chamber_types": {
                "Box": {
                    "polygon": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "markers": {"cross": [0, 0], "circle": [1, 0]},
                }
            },
            "blueprint_map": [
                {"roi_id": "0000", "x": 0, "y": 0},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                load_chip_config(f.name)
            self.assertIn("missing 'structure_type'", str(ctx.exception))

    def test_invalid_structure_type_in_blueprint(self):
        """Test validation rejects blueprint entry referencing unknown structure_type."""
        config_data = {
            "chip_name": "test",
            "version": "2.0",
            "pixel_size": 0.065789,
            "chamber_types": {
                "Box": {
                    "polygon": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "markers": {"cross": [0, 0], "circle": [1, 0]},
                }
            },
            "blueprint_map": [
                {"roi_id": "0000", "x": 0, "y": 0, "structure_type": "NonExistent"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                load_chip_config(f.name)
            self.assertIn("unknown structure_type", str(ctx.exception))


class TestChipStructureLibrary(unittest.TestCase):
    """Tests for ChipStructureLibrary class."""

    def setUp(self):
        """Load both old and new libraries for comparison."""
        self.new_lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH, pixel_size=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.old_lib = SAKRoIStructureLibrary(
                lookup_path=CHAMBER_STRUCTURE_PATH,
                pixel_size=1.0,
            )

    def test_from_file(self):
        """Test from_file classmethod."""
        lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH)
        self.assertIsInstance(lib, ChipStructureLibrary)
        self.assertEqual(lib.pixel_size, 0.065789)

    def test_from_file_with_pixel_size_override(self):
        """Test from_file with pixel_size override."""
        lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH, pixel_size=0.1)
        self.assertEqual(lib.pixel_size, 0.1)

    def test_call_returns_correct_types(self):
        """Test __call__ returns (str, RoIPolygon, dict)."""
        from dmc_masking.mask import RoIPolygon

        name, polygon, markers = self.new_lib("0000")
        self.assertIsInstance(name, str)
        self.assertIsInstance(polygon, RoIPolygon)
        self.assertIsInstance(markers, dict)
        self.assertIn("cross", markers)
        self.assertIn("circle", markers)

    def test_unknown_roi_id_raises(self):
        """Test that an unknown ROI ID raises ValueError."""
        with self.assertRaises(ValueError):
            self.new_lib("9999")

    def test_roi_to_structure_dict(self):
        """Test _roi_to_structure dict is populated from blueprint_map."""
        self.assertEqual(len(self.new_lib._roi_to_structure), 1164)
        for roi_id, structure_type in self.new_lib._roi_to_structure.items():
            self.assertIsInstance(roi_id, str)
            self.assertIn(structure_type, self.new_lib.polygon_library)

    def test_call_with_all_blueprint_rois(self):
        """Test that every ROI in blueprint_map resolves via __call__."""
        for entry in self.new_lib.chip_config.blueprint_map:
            roi_id = entry["roi_id"]
            name, _polygon, _markers = self.new_lib(roi_id)
            self.assertEqual(name, entry["structure_type"])

    def test_unknown_roi_id_not_in_blueprint(self):
        """Test that a ROI ID not in blueprint_map raises ValueError."""
        with self.assertRaises(ValueError):
            self.new_lib("9999")


class TestEquivalence(unittest.TestCase):
    """Equivalence tests: ChipStructureLibrary must produce identical results
    to the old SAKRoIStructureLibrary for all 8 chamber types."""

    def setUp(self):
        """Load both libraries with pixel_size=1 for comparison."""
        self.new_lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH, pixel_size=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.old_lib = SAKRoIStructureLibrary(
                lookup_path=CHAMBER_STRUCTURE_PATH,
                pixel_size=1.0,
            )

    # Representative ROI IDs for each of the 8 chamber types
    CHAMBER_TEST_CASES: ClassVar[list] = [
        ("0000", "NormaleBox-inner"),
        ("0100", "BigBox-inner"),
        ("0200", "OpenBox-inner"),
        ("0300", "Mothermachine-inner"),
        ("1000", "NormaleBox-pillar-inner"),
        ("1100", "BigBox-pillar-inner"),
        ("1200", "OpenBox-collector-inner"),
        ("1300", "Mothermachine-2x-inner"),
    ]

    def test_structure_names_match(self):
        """Test that structure names match for all 8 chamber types."""
        for roi_id, expected_name in self.CHAMBER_TEST_CASES:
            new_name, _, _ = self.new_lib(roi_id)
            old_name, _, _ = self.old_lib(roi_id)
            self.assertEqual(new_name, old_name, f"Name mismatch for {roi_id}")
            self.assertEqual(new_name, expected_name, f"Unexpected name for {roi_id}")

    def test_polygon_areas_match(self):
        """Test that polygon areas match for all 8 chamber types."""
        for roi_id, _ in self.CHAMBER_TEST_CASES:
            _, new_poly, _ = self.new_lib(roi_id)
            _, old_poly, _ = self.old_lib(roi_id)
            np.testing.assert_almost_equal(
                new_poly.area,
                old_poly.area,
                decimal=1,
                err_msg=f"Area mismatch for {roi_id}",
            )

    def test_polygon_bounds_match(self):
        """Test that polygon bounding boxes match for all 8 chamber types."""
        for roi_id, _ in self.CHAMBER_TEST_CASES:
            _, new_poly, _ = self.new_lib(roi_id)
            _, old_poly, _ = self.old_lib(roi_id)
            np.testing.assert_array_almost_equal(
                new_poly.roi_polygon.bounds,
                old_poly.roi_polygon.bounds,
                decimal=3,
                err_msg=f"Bounds mismatch for {roi_id}",
            )

    def test_marker_positions_match(self):
        """Test that marker positions match for all 8 chamber types."""
        for roi_id, _ in self.CHAMBER_TEST_CASES:
            _, _, new_markers = self.new_lib(roi_id)
            _, _, old_markers = self.old_lib(roi_id)

            for marker_name in ["cross", "circle"]:
                np.testing.assert_array_almost_equal(
                    new_markers[marker_name],
                    old_markers[marker_name],
                    decimal=6,
                    err_msg=f"Marker '{marker_name}' mismatch for {roi_id}",
                )

    def test_equivalence_with_default_pixel_size(self):
        """Test equivalence with the default pixel size (0.065789)."""
        new_lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH, pixel_size=0.065789)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old_lib = SAKRoIStructureLibrary(
                lookup_path=CHAMBER_STRUCTURE_PATH,
                pixel_size=0.065789,
            )

        for roi_id, _ in self.CHAMBER_TEST_CASES:
            _, new_poly, new_markers = new_lib(roi_id)
            _, old_poly, old_markers = old_lib(roi_id)

            np.testing.assert_almost_equal(
                new_poly.area,
                old_poly.area,
                decimal=0,
                err_msg=f"Area mismatch for {roi_id} at default pixel_size",
            )

            for marker_name in ["cross", "circle"]:
                np.testing.assert_array_almost_equal(
                    new_markers[marker_name],
                    old_markers[marker_name],
                    decimal=3,
                    err_msg=f"Marker '{marker_name}' mismatch for {roi_id} at default pixel_size",
                )


class TestBlueprintMap(unittest.TestCase):
    """Tests for blueprint map loading from chip config."""

    def test_get_blueprint_map(self):
        """Test get_blueprint_map returns a valid Map."""
        lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH)
        blueprint_map = lib.get_blueprint_map()

        self.assertIsInstance(blueprint_map, Map)
        self.assertEqual(len(blueprint_map.roi_positions), 1164)

    def test_blueprint_map_matches_csv(self):
        """Test that blueprint map from chip config matches the CSV."""
        lib = ChipStructureLibrary.from_file(SAK_CONFIG_PATH)
        new_map = lib.get_blueprint_map()
        old_map = Map.from_csv(BLUEPRINT_MAP_PATH)

        # Same number of ROIs
        self.assertEqual(
            len(new_map.roi_positions),
            len(old_map.roi_positions),
        )

        # Same ROI IDs
        self.assertEqual(
            set(new_map.roi_positions.keys()),
            set(old_map.roi_positions.keys()),
        )

        # Same positions
        for roi_id in old_map.roi_positions:
            np.testing.assert_array_almost_equal(
                new_map.roi_positions[roi_id].position,
                old_map.roi_positions[roi_id].position,
                decimal=1,
                err_msg=f"Blueprint position mismatch for {roi_id}",
            )

    def test_empty_blueprint_map_raises(self):
        """Test that get_blueprint_map raises on empty blueprint."""
        config = ChipConfig(
            chip_name="test",
            version="1.0",
            description="",
            pixel_size=0.065789,
            chamber_types={},
            blueprint_map=[],
        )
        lib = ChipStructureLibrary(config)
        with self.assertRaises(ValueError):
            lib.get_blueprint_map()


class TestMapFromDictList(unittest.TestCase):
    """Tests for Map.from_dict_list classmethod."""

    def test_basic(self):
        """Test basic creation from dict list."""
        entries = [
            {"roi_id": "0", "x": 100.0, "y": 200.0},
            {"roi_id": "1", "x": 300.0, "y": 400.0},
        ]
        m = Map.from_dict_list(entries)
        self.assertEqual(len(m.roi_positions), 2)
        self.assertIn("0000", m.roi_positions)
        self.assertIn("0001", m.roi_positions)

    def test_positions_correct(self):
        """Test that positions are stored correctly."""
        entries = [{"roi_id": "50", "x": 1234.5, "y": -6789.0}]
        m = Map.from_dict_list(entries)
        np.testing.assert_array_almost_equal(
            m.roi_positions["0050"].position,
            [1234.5, -6789.0],
        )


class TestDeprecationWarnings(unittest.TestCase):
    """Test that deprecated classes emit DeprecationWarning."""

    def test_sak_roi_structure_library_warns(self):
        """Test SAKRoIStructureLibrary emits deprecation warning."""
        with self.assertWarns(DeprecationWarning) as ctx:
            SAKRoIStructureLibrary(
                lookup_path=CHAMBER_STRUCTURE_PATH,
                pixel_size=1.0,
            )
        self.assertIn("deprecated", str(ctx.warning).lower())


if __name__ == "__main__":
    unittest.main()

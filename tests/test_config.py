"""Tests for the configuration system."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from dmc_masking.config import (
    AxisDirection,
    CalibrationConfig,
    CoordinatesConfig,
    CoordinateSystemConfig,
    DetectionConfig,
    DMCConfig,
    PathConfig,
    get_default_config,
    set_default_config,
)


class TestDetectionConfig(unittest.TestCase):
    """Tests for DetectionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DetectionConfig()
        self.assertEqual(config.tolerance, 60)
        self.assertEqual(config.confidence, 0.6)

    def test_custom_values(self):
        """Test setting custom values."""
        config = DetectionConfig(tolerance=80, confidence=0.7)
        self.assertEqual(config.tolerance, 80)
        self.assertEqual(config.confidence, 0.7)


class TestCalibrationConfig(unittest.TestCase):
    """Tests for CalibrationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        self.assertAlmostEqual(config.pixel_size, 0.065789)
        self.assertEqual(config.min_calibration_points, 3)

    def test_custom_pixel_size(self):
        """Test setting custom pixel size."""
        config = CalibrationConfig(pixel_size=0.1)
        self.assertEqual(config.pixel_size, 0.1)


class TestPathConfig(unittest.TestCase):
    """Tests for PathConfig."""

    def test_default_paths(self):
        """Test default path values."""
        config = PathConfig()
        self.assertEqual(config.model_path, Path("artifacts/models/v26_detect_s_imgsz1280.pt"))
        self.assertEqual(config.structure_library_path, Path("artifacts/chamber_structure.json"))
        self.assertIsNone(config.blueprint_map_path)

    def test_string_to_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = PathConfig(
            model_path="custom/model.pt",
            structure_library_path="custom/structures.json",
            blueprint_map_path="custom/blueprint.csv",
        )
        self.assertIsInstance(config.model_path, Path)
        self.assertIsInstance(config.structure_library_path, Path)
        self.assertIsInstance(config.blueprint_map_path, Path)


class TestCoordinateSystemConfig(unittest.TestCase):
    """Tests for CoordinateSystemConfig."""

    def test_default_values(self):
        """Test default coordinate system values."""
        config = CoordinateSystemConfig()
        self.assertEqual(config.x_direction, AxisDirection.POSITIVE)
        self.assertEqual(config.y_direction, AxisDirection.POSITIVE)
        self.assertFalse(config.flip_x)
        self.assertFalse(config.flip_y)


class TestCoordinatesConfig(unittest.TestCase):
    """Tests for CoordinatesConfig."""

    def test_default_blueprint_y_is_negative(self):
        """Blueprint uses Y-up (Cartesian), so y_direction should be NEGATIVE."""
        config = CoordinatesConfig()
        self.assertEqual(config.blueprint.y_direction, AxisDirection.NEGATIVE)

    def test_default_image_y_is_positive(self):
        """Image uses Y-down (standard), so y_direction should be POSITIVE."""
        config = CoordinatesConfig()
        self.assertEqual(config.image.y_direction, AxisDirection.POSITIVE)

    def test_blueprint_to_image_invert_y_default_true(self):
        """Y-inversion from blueprint to image should be True by default."""
        config = CoordinatesConfig()
        self.assertTrue(config.blueprint_to_image_invert_y)


class TestDMCConfig(unittest.TestCase):
    """Tests for DMCConfig."""

    def test_default_values(self):
        """Test that DMCConfig has sensible defaults."""
        config = DMCConfig()

        # Detection defaults
        self.assertEqual(config.detection.tolerance, 60)
        self.assertEqual(config.detection.confidence, 0.6)

        # Calibration defaults
        self.assertAlmostEqual(config.calibration.pixel_size, 0.065789)
        self.assertEqual(config.calibration.min_calibration_points, 3)

        # Coordinate defaults
        self.assertTrue(config.coordinates.blueprint_to_image_invert_y)

    def test_from_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "detection": {"tolerance": 80, "confidence": 0.75},
            "calibration": {"pixel_size": 0.1, "min_calibration_points": 5},
            "paths": {
                "model_path": "custom/model.pt",
                "structure_library_path": "custom/structures.json",
            },
            "coordinates": {"blueprint_to_image_invert_y": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = DMCConfig.from_json(temp_path)

            self.assertEqual(config.detection.tolerance, 80)
            self.assertEqual(config.detection.confidence, 0.75)
            self.assertEqual(config.calibration.pixel_size, 0.1)
            self.assertEqual(config.calibration.min_calibration_points, 5)
            self.assertEqual(config.paths.model_path, Path("custom/model.pt"))
            self.assertFalse(config.coordinates.blueprint_to_image_invert_y)
        finally:
            os.unlink(temp_path)

    def test_from_json_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with self.assertRaises(FileNotFoundError):
            DMCConfig.from_json("nonexistent_config.json")

    def test_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["DMC_TOLERANCE"] = "100"
        os.environ["DMC_CONFIDENCE"] = "0.8"
        os.environ["DMC_PIXEL_SIZE"] = "0.05"

        try:
            config = DMCConfig.from_env()

            self.assertEqual(config.detection.tolerance, 100)
            self.assertEqual(config.detection.confidence, 0.8)
            self.assertEqual(config.calibration.pixel_size, 0.05)
        finally:
            # Clean up
            del os.environ["DMC_TOLERANCE"]
            del os.environ["DMC_CONFIDENCE"]
            del os.environ["DMC_PIXEL_SIZE"]

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = DMCConfig()
        data = config.to_dict()

        self.assertIn("detection", data)
        self.assertIn("calibration", data)
        self.assertIn("paths", data)
        self.assertIn("coordinates", data)

        self.assertEqual(data["detection"]["tolerance"], 60)
        self.assertEqual(data["calibration"]["pixel_size"], 0.065789)

    def test_to_json(self):
        """Test saving configuration to JSON file."""
        config = DMCConfig()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.to_json(temp_path)

            # Load and verify
            with open(temp_path) as f:
                data = json.load(f)

            self.assertEqual(data["detection"]["tolerance"], 60)
            self.assertEqual(data["calibration"]["pixel_size"], 0.065789)
        finally:
            os.unlink(temp_path)

    def test_roundtrip_json(self):
        """Test that save/load roundtrip preserves values."""
        original = DMCConfig()
        original.detection.tolerance = 75
        original.calibration.pixel_size = 0.08

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            original.to_json(temp_path)
            loaded = DMCConfig.from_json(temp_path)

            self.assertEqual(loaded.detection.tolerance, original.detection.tolerance)
            self.assertEqual(loaded.calibration.pixel_size, original.calibration.pixel_size)
        finally:
            os.unlink(temp_path)


class TestGlobalConfig(unittest.TestCase):
    """Tests for global configuration functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        self.assertIsInstance(config, DMCConfig)

    def test_set_default_config(self):
        """Test setting default configuration."""
        custom = DMCConfig()
        custom.detection.tolerance = 999

        set_default_config(custom)
        retrieved = get_default_config()

        self.assertEqual(retrieved.detection.tolerance, 999)

        # Reset to default for other tests
        set_default_config(DMCConfig())


class TestCoordinatesConfigParsing(unittest.TestCase):
    """Tests for parsing coordinate configuration from JSON."""

    def test_parse_axis_directions(self):
        """Test parsing axis directions from JSON."""
        config_data = {
            "coordinates": {
                "blueprint": {
                    "x_direction": "positive",
                    "y_direction": "negative",
                },
                "image": {
                    "x_direction": "positive",
                    "y_direction": "positive",
                },
                "stage": {
                    "x_direction": "negative",
                    "y_direction": "positive",
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = DMCConfig.from_json(temp_path)

            self.assertEqual(config.coordinates.blueprint.x_direction, AxisDirection.POSITIVE)
            self.assertEqual(config.coordinates.blueprint.y_direction, AxisDirection.NEGATIVE)
            self.assertEqual(config.coordinates.stage.x_direction, AxisDirection.NEGATIVE)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()

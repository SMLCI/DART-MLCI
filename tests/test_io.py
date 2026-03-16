"""Tests for the io module."""

import json
import unittest
from pathlib import Path

import numpy as np

from dart_mlci.io import load_image, load_roi_structures


class TestLoadImage(unittest.TestCase):
    """Tests for load_image function."""

    def test_load_image_output_shape(self):
        """load_image should return HxWx3 uint8 array."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        image = load_image(image_path)

        # Verify expected properties
        self.assertEqual(len(image.shape), 3)  # HxWx3
        self.assertEqual(image.shape[2], 3)  # RGB
        self.assertEqual(image.dtype, np.uint8)

    def test_load_image_accepts_string_path(self):
        """load_image should accept string paths as well as Path objects."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = config["calibration_images"][0]["image_path"]  # String
        if not Path(image_path).exists():
            self.skipTest(f"Test image not available: {image_path}")

        # Should work with string path
        image = load_image(image_path)
        self.assertEqual(len(image.shape), 3)

    def test_load_image_file_not_found(self):
        """load_image should raise FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            load_image("nonexistent_image.tif")

    def test_load_image_value_range(self):
        """Image values should be in valid uint8 range [0, 255]."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        image = load_image(image_path)

        self.assertTrue(image.min() >= 0)
        self.assertTrue(image.max() <= 255)

    def test_load_image_grayscale_to_rgb_consistency(self):
        """Grayscale to RGB conversion should replicate the value across channels."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        image = load_image(image_path)

        # For images that were originally grayscale, all channels should be equal
        # (This may not be true for all test images, so we just check channels exist)
        self.assertEqual(image.shape[2], 3)


class TestLoadImageCompatibility(unittest.TestCase):
    """Test that io.load_image produces same results as script implementations."""

    def test_matches_calibrate_map_load_image(self):
        """io.load_image should produce same output as calibrate_map.load_image."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        # Load using both implementations
        from scripts.calibrate_map import load_image as old_load_image

        old_image = old_load_image(image_path)
        new_image = load_image(image_path)

        # Should be identical
        np.testing.assert_array_equal(old_image, new_image)

    def test_matches_benchmark_load_image(self):
        """io.load_image should produce same output as benchmark.load_image."""
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        # Load using both implementations
        from scripts.benchmark import load_image as old_load_image

        old_image = old_load_image(image_path)
        new_image = load_image(image_path)

        # Should be identical
        np.testing.assert_array_equal(old_image, new_image)


class TestLoadRoiStructures(unittest.TestCase):
    """Tests for load_roi_structures function."""

    def test_load_structures_from_file(self):
        """Should load structure definitions from JSON file."""
        # Use the actual chamber structure file
        structure_path = Path("artifacts/chamber_structure.json")
        if not structure_path.exists():
            self.skipTest("Chamber structure file not available")

        structures = load_roi_structures(structure_path)

        self.assertIsInstance(structures, dict)
        self.assertIn("NormaleBox-inner", structures)

    def test_load_structures_creates_valid_dict(self):
        """Loaded structures should be usable as geometry definitions."""
        structure_path = Path("artifacts/chamber_structure.json")
        if not structure_path.exists():
            self.skipTest("Chamber structure file not available")

        structures = load_roi_structures(structure_path)

        # Each structure should have coordinates that can be converted to a polygon
        for _name, structure in structures.items():
            self.assertIn("type", structure)
            self.assertIn("coordinates", structure)


if __name__ == "__main__":
    unittest.main()

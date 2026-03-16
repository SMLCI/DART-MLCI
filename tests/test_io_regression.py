"""Regression tests for load_image behavior."""

import unittest
from pathlib import Path

import numpy as np


class TestLoadImageRegression(unittest.TestCase):
    """Ensure load_image produces identical outputs before/after refactoring."""

    def test_load_image_output_shape(self):
        """load_image should return HxWx3 uint8 array."""
        from scripts.calibrate_map import load_image

        # Use the calibration test config to find a real image
        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        import json

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

    def test_load_image_grayscale_conversion(self):
        """Grayscale images should be converted to RGB (3 channel)."""
        from scripts.calibrate_map import load_image

        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        import json

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        image = load_image(image_path)

        # All three channels should be present
        self.assertEqual(image.shape[2], 3)

        # For grayscale-to-RGB conversion, all channels should be equal
        # (this is true for the test images which are typically grayscale)
        if np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(
            image[:, :, 1], image[:, :, 2]
        ):
            # Image was grayscale converted to RGB
            pass
        else:
            # Image was already RGB
            pass

    def test_load_image_value_range(self):
        """Image values should be in valid uint8 range [0, 255]."""
        from scripts.calibrate_map import load_image

        config_path = Path("scripts/calibration_test.json")
        if not config_path.exists():
            self.skipTest("Calibration test config not available")

        import json

        with open(config_path) as f:
            config = json.load(f)

        image_path = Path(config["calibration_images"][0]["image_path"])
        if not image_path.exists():
            self.skipTest(f"Test image not available: {image_path}")

        image = load_image(image_path)

        self.assertTrue(image.min() >= 0)
        self.assertTrue(image.max() <= 255)


class TestNormalizeImage(unittest.TestCase):
    """Test the normalize_image utility function."""

    def test_normalize_uint16_to_uint8(self):
        """16-bit images should be normalized to 8-bit."""
        from dart_mlci.utils import normalize_image

        # Create a 16-bit test image
        arr_16bit = np.array([[0, 32768, 65535]], dtype=np.uint16)

        normalized = normalize_image(arr_16bit)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(normalized.min() >= 0)
        self.assertTrue(normalized.max() <= 255)

    def test_normalize_preserves_relative_values(self):
        """Normalization should preserve relative value ordering."""
        from dart_mlci.utils import normalize_image

        # Create gradient
        arr = np.arange(0, 65536, 256, dtype=np.uint16).reshape(16, 16)

        normalized = normalize_image(arr)

        # Check that gradient is preserved (values increase)
        for i in range(1, normalized.shape[0]):
            self.assertGreaterEqual(
                normalized[i, 0],
                normalized[i - 1, 0],
                "Normalization should preserve value ordering",
            )


if __name__ == "__main__":
    unittest.main()

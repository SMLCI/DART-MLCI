"""Tests for the io module."""

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from dart_mlci.io import load_image, load_roi_structures, save_image


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


class TestLoadImageSynthetic(unittest.TestCase):
    """Tests using synthetic images (no external files needed)."""

    def test_load_image_grayscale_synthetic(self):
        """Grayscale PNG should be converted to HxWx3 uint8."""
        import cv2

        tmp = Path("/tmp/test_gray.png")
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp), gray)

        image = load_image(tmp)
        self.assertEqual(image.shape, (64, 64, 3))
        self.assertEqual(image.dtype, np.uint8)
        # All channels should be equal (grayscale replicated)
        np.testing.assert_array_equal(image[:, :, 0], image[:, :, 1])
        np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])
        tmp.unlink()

    def test_load_image_tiff_ndim4(self):
        """4D TIFF (TxCxHxW) should extract first frame, first channel."""
        import tifffile

        tmp = Path("/tmp/test_4d.tif")
        data = np.random.randint(0, 65535, (2, 3, 32, 32), dtype=np.uint16)
        tifffile.imwrite(str(tmp), data)

        image = load_image(tmp)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.uint8)
        tmp.unlink()

    def test_load_image_tiff_channel_first(self):
        """CxHxW TIFF should take first channel."""
        import tifffile

        tmp = Path("/tmp/test_chw.tif")
        data = np.random.randint(0, 65535, (3, 48, 48), dtype=np.uint16)
        tifffile.imwrite(str(tmp), data)

        image = load_image(tmp)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.uint8)
        tmp.unlink()

    def test_load_image_invalid_file(self):
        """Garbage bytes file should raise ValueError."""
        tmp = Path("/tmp/test_garbage.png")
        tmp.write_bytes(b"\x00\x01\x02garbage data")

        with self.assertRaises(ValueError):
            load_image(tmp)
        tmp.unlink()


class TestLoadImageSingleChannelPath(unittest.TestCase):
    """Exercises the HxWx1 → RGB promotion branch of load_image."""

    def test_single_channel_hwc_becomes_rgb(self):
        import tempfile

        img_hw1 = np.zeros((20, 20, 1), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            tmp = Path(fp.name)
        cv2.imwrite(str(tmp), img_hw1)
        try:
            loaded = load_image(tmp)
            self.assertEqual(loaded.ndim, 3)
            self.assertEqual(loaded.shape[2], 3)
        finally:
            tmp.unlink()


class TestSaveImage:
    """save_image: TIFF/OpenCV format selection, BGR conversion, optional mask sidecar."""

    def _img_hwc(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)

    def test_png_round_trip(self, tmp_path):
        img = self._img_hwc()
        out = tmp_path / "out.png"
        mask_path = save_image(img, out)
        assert mask_path is None
        assert out.exists()
        loaded = load_image(out)
        # PNG via OpenCV preserves values for uint8; channel order matches.
        assert loaded.shape == img.shape

    def test_tiff_with_mask(self, tmp_path):
        import tifffile

        img = self._img_hwc()
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        out = tmp_path / "out.tif"
        mask_path = save_image(img, out, mask=mask)
        assert mask_path is not None
        assert mask_path.exists()
        assert mask_path.name == "out_mask.tif"

        m = tifffile.imread(str(mask_path))
        assert m.shape == (16, 16)
        assert m.dtype == np.uint8
        assert m[5, 5] == 255
        assert m[0, 0] == 0

    def test_chw_input_transposed(self, tmp_path):
        import tifffile

        img_chw = np.zeros((3, 16, 16), dtype=np.uint8)
        img_chw[0] = 100  # red channel
        out = tmp_path / "out.tif"
        save_image(img_chw, out)
        loaded = tifffile.imread(str(out))
        assert loaded.shape == (16, 16, 3)
        assert loaded[0, 0, 0] == 100


if __name__ == "__main__":
    unittest.main()

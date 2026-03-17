"""Tests for the calibration core module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from shapely.geometry import box

from dart_mlci.calibration import (
    CalibrationError,
    ImageCalibrationResult,
    ImageDebugData,
    compute_chamber_center,
    compute_microscope_position,
    filter_matched_pairs_by_bounds,
    process_calibration_image,
    run_calibration,
)
from dart_mlci.mask import RoIPolygon

MODEL_PATH = Path("artifacts/models/v26_detect_s_imgsz1280.pt")
CHIP_CONFIG_PATH = Path("artifacts/chips/sak.json")
CAL_SAMPLE_DIR = Path("scripts/calibration_sample")


class TestImageDebugData(unittest.TestCase):
    """Tests for ImageDebugData dataclass."""

    def test_default_values(self):
        """Test that defaults are all None."""
        data = ImageDebugData()

        self.assertIsNone(data.image)
        self.assertIsNone(data.markers)
        self.assertIsNone(data.matched_indices)
        self.assertIsNone(data.chamber_center_pixels)

    def test_with_values(self):
        """Test setting values."""
        data = ImageDebugData(
            stage_position={"x": 100.0, "y": 200.0},
            pixel_size=0.065789,
        )

        self.assertEqual(data.stage_position["x"], 100.0)
        self.assertEqual(data.pixel_size, 0.065789)


class TestImageCalibrationResult(unittest.TestCase):
    """Tests for ImageCalibrationResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ImageCalibrationResult(
            roi_id="0050",
            success=True,
            microscope_position=np.array([6800.0, -4200.0]),
            z_position=2900.0,
        )

        self.assertEqual(result.roi_id, "0050")
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)

    def test_failure_result(self):
        """Test creating a failed result."""
        result = ImageCalibrationResult(
            roi_id="0050",
            success=False,
            microscope_position=None,
            z_position=None,
            error_message="No markers found",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No markers found")


class TestComputeChamberCenter(unittest.TestCase):
    """Tests for compute_chamber_center function."""

    def test_basic_computation(self):
        """Test basic chamber center computation."""
        # Create a 100x100 polygon
        polygon = RoIPolygon(box(0, 0, 100, 100))

        # Marker positions
        marker_group = {
            "cross": np.array([14.0, 8.0]),
            "circle": np.array([66.0, 8.0]),
        }

        # Detected markers
        markers = [
            {"bbox_center": np.array([500.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([552.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        center = compute_chamber_center(
            markers, matched_indices, marker_group, polygon, rotation_angle=0.0
        )

        # Expected:
        # polygon_center = (50, 50)
        # offset_x = 50 - 14 = 36
        # offset_y = 50 + 8 = 58 (note: + for Y inversion)
        # center = (500 + 36, 300 + 58) = (536, 358)
        expected = np.array([536.0, 358.0])

        np.testing.assert_array_almost_equal(center, expected, decimal=1)

    def test_y_offset_uses_addition(self):
        """Verify Y offset uses + (critical for coordinate inversion)."""
        polygon = RoIPolygon(box(0, 0, 100, 100))

        marker_group = {
            "cross": np.array([0.0, 10.0]),  # 10 units in Y
            "circle": np.array([50.0, 10.0]),
        }

        markers = [
            {"bbox_center": np.array([100.0, 100.0]), "label": "cross"},
            {"bbox_center": np.array([150.0, 100.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        center = compute_chamber_center(
            markers, matched_indices, marker_group, polygon, rotation_angle=0.0
        )

        # offset_x = 50 - 0 = 50
        # offset_y = 50 + 10 = 60 (+ not -)
        # center = (100 + 50, 100 + 60) = (150, 160)

        # If Y used subtraction, it would be (150, 140)
        self.assertAlmostEqual(center[1], 160.0, places=1)

    def test_with_rotation(self):
        """Test chamber center computation with rotation."""
        polygon = RoIPolygon(box(0, 0, 100, 100))

        marker_group = {
            "cross": np.array([50.0, 50.0]),
            "circle": np.array([100.0, 50.0]),
        }

        markers = [
            {"bbox_center": np.array([500.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([550.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        # With 90 degree rotation, offset should be rotated
        center = compute_chamber_center(
            markers, matched_indices, marker_group, polygon, rotation_angle=90.0
        )

        self.assertIsNotNone(center)
        self.assertEqual(len(center), 2)

    def test_no_matched_pairs_raises(self):
        """Test that empty matched_indices raises ValueError."""
        polygon = RoIPolygon(box(0, 0, 100, 100))

        with self.assertRaises(ValueError):
            compute_chamber_center(
                markers=[],
                matched_indices=[],
                marker_group_pixels={"cross": np.array([0, 0]), "circle": np.array([50, 0])},
                roi_polygon=polygon,
            )


class TestFilterMatchedPairsByBounds(unittest.TestCase):
    """Tests for filter_matched_pairs_by_bounds function."""

    def test_filters_out_of_bounds_pairs(self):
        """Pairs that would place RoI outside image should be filtered."""
        polygon = RoIPolygon(box(0, 0, 100, 100))

        marker_group = {
            "cross": np.array([14.0, 8.0]),
            "circle": np.array([66.0, 8.0]),
        }

        image_shape = (600, 800)

        # One pair near edge (out of bounds), one in center (valid)
        markers = [
            # Pair 0: near left edge
            {"bbox_center": np.array([10.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([62.0, 300.0]), "label": "circle"},
            # Pair 1: in center
            {"bbox_center": np.array([400.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([452.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1), (2, 3)]

        filtered = filter_matched_pairs_by_bounds(
            markers, matched_indices, marker_group, polygon, image_shape, rotation_angle=0.0
        )

        # Only the center pair should remain
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], (2, 3))

    def test_sorts_by_margin(self):
        """Valid pairs should be sorted by margin (largest first)."""
        polygon = RoIPolygon(box(0, 0, 50, 50))

        marker_group = {
            "cross": np.array([10.0, 5.0]),
            "circle": np.array([40.0, 5.0]),
        }

        image_shape = (600, 800)

        # Two valid pairs with different margins
        markers = [
            # Pair 0: closer to edge
            {"bbox_center": np.array([100.0, 100.0]), "label": "cross"},
            {"bbox_center": np.array([130.0, 100.0]), "label": "circle"},
            # Pair 1: more centered (larger margin)
            {"bbox_center": np.array([400.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([430.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1), (2, 3)]

        filtered = filter_matched_pairs_by_bounds(
            markers, matched_indices, marker_group, polygon, image_shape, rotation_angle=0.0
        )

        # Both should be valid, but pair 1 should come first (larger margin)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], (2, 3))


class TestComputeMicroscopePosition(unittest.TestCase):
    """Tests for compute_microscope_position function."""

    def test_basic_computation(self):
        """Test basic microscope position computation."""
        chamber_center_pixels = np.array([500.0, 300.0])
        stage_position = {"x": 6802.4, "y": -4272.9, "z": 2942.5}
        pixel_size = 0.065789

        pos_xy, z = compute_microscope_position(chamber_center_pixels, stage_position, pixel_size)

        # Expected:
        # center_microns = (500 * 0.065789, 300 * 0.065789) = (32.89, 19.74)
        # pos = (6802.4 + 32.89, -4272.9 + 19.74) = (6835.29, -4253.16)
        expected_x = 6802.4 + 500.0 * 0.065789
        expected_y = -4272.9 + 300.0 * 0.065789

        self.assertAlmostEqual(pos_xy[0], expected_x, places=2)
        self.assertAlmostEqual(pos_xy[1], expected_y, places=2)
        self.assertAlmostEqual(z, 2942.5)

    def test_without_z(self):
        """Test when z is not provided."""
        chamber_center_pixels = np.array([100.0, 100.0])
        stage_position = {"x": 0.0, "y": 0.0}  # No z
        pixel_size = 1.0

        _pos_xy, z = compute_microscope_position(chamber_center_pixels, stage_position, pixel_size)

        self.assertIsNone(z)


class TestProcessCalibrationImage(unittest.TestCase):
    """Tests for process_calibration_image function."""

    def test_returns_detection_error_on_blank_image(self):
        """Blank image should return success=False with DETECTION error."""
        # Create a mock detection step that returns no markers
        detection_step = MagicMock()
        detection_step.return_value = {"markers": []}

        # Create a mock structure library
        structure_library = MagicMock()
        roi_polygon = RoIPolygon(box(0, 0, 100, 100))
        structure_library.return_value = (
            "test_structure",
            roi_polygon,
            {"cross": np.array([14.0, 8.0]), "circle": np.array([66.0, 8.0])},
        )

        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = process_calibration_image(
            image=blank_image,
            roi_id="0050",
            stage_position={"x": 100.0, "y": 200.0, "z": 50.0},
            detection_step=detection_step,
            structure_library=structure_library,
            pixel_size=0.065789,
        )

        self.assertFalse(result.success)
        self.assertIn("DETECTION", result.error_message)
        self.assertIsNone(result.microscope_position)

    def test_returns_matching_error_when_no_pairs(self):
        """When detection finds markers but matching fails, error contains MATCHING."""
        # Detection returns markers but matching produces no pairs
        detection_step = MagicMock()
        detection_step.return_value = {
            "markers": [
                {"bbox_center": np.array([100.0, 100.0]), "label": "cross", "conf": 0.9},
                {"bbox_center": np.array([500.0, 500.0]), "label": "circle", "conf": 0.9},
            ]
        }

        structure_library = MagicMock()
        roi_polygon = RoIPolygon(box(0, 0, 100, 100))
        structure_library.return_value = (
            "test_structure",
            roi_polygon,
            {"cross": np.array([14.0, 8.0]), "circle": np.array([66.0, 8.0])},
        )

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = process_calibration_image(
            image=image,
            roi_id="0050",
            stage_position={"x": 100.0, "y": 200.0},
            detection_step=detection_step,
            structure_library=structure_library,
            pixel_size=0.065789,
        )

        self.assertFalse(result.success)
        self.assertIn("MATCHING", result.error_message)

    @unittest.skipUnless(
        MODEL_PATH.exists() and CHIP_CONFIG_PATH.exists(),
        "Model or chip config not found",
    )
    def test_success_with_real_image(self):
        """Real calibration image should produce a successful result."""
        from dart_mlci import MarkerDetectionStep
        from dart_mlci.chip import ChipStructureLibrary
        from dart_mlci.io import load_image

        test_image_path = Path("tests/fixtures/calibration_image_0000.tif")
        if not test_image_path.exists():
            self.skipTest("Test image not found")

        image = load_image(test_image_path)
        detection_step = MarkerDetectionStep(str(MODEL_PATH), verbose=False)
        structure_library = ChipStructureLibrary.from_file(CHIP_CONFIG_PATH, pixel_size=0.065789)

        result = process_calibration_image(
            image=image,
            roi_id="0000",
            stage_position={"x": 6802.4, "y": -4272.9, "z": 2942.5},
            detection_step=detection_step,
            structure_library=structure_library,
            pixel_size=0.065789,
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.microscope_position)
        self.assertEqual(len(result.microscope_position), 2)


class TestRunCalibration(unittest.TestCase):
    """Tests for run_calibration function."""

    def test_raises_on_insufficient_successful_images(self):
        """3 blank images should raise CalibrationError with image_results."""
        detection_step = MagicMock()
        detection_step.return_value = {"markers": []}

        structure_library = MagicMock()
        roi_polygon = RoIPolygon(box(0, 0, 100, 100))
        structure_library.return_value = (
            "test_structure",
            roi_polygon,
            {"cross": np.array([14.0, 8.0]), "circle": np.array([66.0, 8.0])},
        )

        from dart_mlci.map import Map, RoIPosition

        blueprint_map = Map(
            [
                RoIPosition(roi_id="0000", position=np.array([0.0, 0.0])),
                RoIPosition(roi_id="0001", position=np.array([100.0, 0.0])),
                RoIPosition(roi_id="0002", position=np.array([0.0, 100.0])),
            ]
        )

        blank_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

        with self.assertRaises(CalibrationError) as ctx:
            run_calibration(
                images=blank_images,
                roi_ids=["0000", "0001", "0002"],
                stage_positions=[
                    {"x": 0.0, "y": 0.0},
                    {"x": 100.0, "y": 0.0},
                    {"x": 0.0, "y": 100.0},
                ],
                detection_step=detection_step,
                structure_library=structure_library,
                blueprint_map=blueprint_map,
                pixel_size=0.065789,
            )

        err = ctx.exception
        self.assertIn("got 0", str(err))

    def test_image_results_attached_to_exception(self):
        """CalibrationError should have image_results with per-image errors."""
        detection_step = MagicMock()
        detection_step.return_value = {"markers": []}

        structure_library = MagicMock()
        roi_polygon = RoIPolygon(box(0, 0, 100, 100))
        structure_library.return_value = (
            "test_structure",
            roi_polygon,
            {"cross": np.array([14.0, 8.0]), "circle": np.array([66.0, 8.0])},
        )

        from dart_mlci.map import Map, RoIPosition

        blueprint_map = Map(
            [
                RoIPosition(roi_id="0000", position=np.array([0.0, 0.0])),
                RoIPosition(roi_id="0001", position=np.array([100.0, 0.0])),
                RoIPosition(roi_id="0002", position=np.array([0.0, 100.0])),
            ]
        )

        blank_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

        with self.assertRaises(CalibrationError) as ctx:
            run_calibration(
                images=blank_images,
                roi_ids=["0000", "0001", "0002"],
                stage_positions=[
                    {"x": 0.0, "y": 0.0},
                    {"x": 100.0, "y": 0.0},
                    {"x": 0.0, "y": 100.0},
                ],
                detection_step=detection_step,
                structure_library=structure_library,
                blueprint_map=blueprint_map,
                pixel_size=0.065789,
            )

        err = ctx.exception
        self.assertEqual(len(err.image_results), 3)
        for img_result in err.image_results:
            self.assertFalse(img_result.success)
            self.assertIsNotNone(img_result.error_message)
            self.assertIn("DETECTION", img_result.error_message)

    @unittest.skipUnless(
        MODEL_PATH.exists() and CHIP_CONFIG_PATH.exists() and CAL_SAMPLE_DIR.exists(),
        "Artifacts not found",
    )
    def test_success_with_real_images(self):
        """Full calibration with real images should succeed."""
        import json

        from dart_mlci import MarkerDetectionStep
        from dart_mlci.chip import ChipStructureLibrary
        from dart_mlci.io import load_image

        config_path = CAL_SAMPLE_DIR / "calibration_config.json"
        if not config_path.exists():
            self.skipTest("Calibration config not found")

        with open(config_path) as f:
            config = json.load(f)

        detection_step = MarkerDetectionStep(str(MODEL_PATH), verbose=False)
        structure_library = ChipStructureLibrary.from_file(
            CHIP_CONFIG_PATH, pixel_size=config["pixel_size"]
        )
        blueprint_map = structure_library.get_blueprint_map()

        images = []
        roi_ids = []
        stage_positions = []
        for img_config in config["calibration_images"][:3]:
            img_path = CAL_SAMPLE_DIR / Path(img_config["image_path"]).name
            if not img_path.exists():
                self.skipTest(f"Image not found: {img_path}")
            images.append(load_image(img_path))
            roi_ids.append(str(img_config["roi_id"]))
            stage_positions.append(img_config["stage_position"])

        result = run_calibration(
            images=images,
            roi_ids=roi_ids,
            stage_positions=stage_positions,
            detection_step=detection_step,
            structure_library=structure_library,
            blueprint_map=blueprint_map,
            pixel_size=config["pixel_size"],
        )

        self.assertIsNotNone(result.calibrated_map)
        self.assertIsNotNone(result.measured_map)
        self.assertGreater(len(result.measured_map.roi_positions), 0)


if __name__ == "__main__":
    unittest.main()

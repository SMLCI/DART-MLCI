"""Tests for the calibration core module."""

import unittest

import numpy as np
from shapely.geometry import box

from dmc_masking.calibration import (
    ImageCalibrationResult,
    ImageDebugData,
    compute_chamber_center,
    compute_microscope_position,
    filter_matched_pairs_by_bounds,
)
from dmc_masking.mask import RoIPolygon


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

        pos_xy, z = compute_microscope_position(chamber_center_pixels, stage_position, pixel_size)

        self.assertIsNone(z)


if __name__ == "__main__":
    unittest.main()

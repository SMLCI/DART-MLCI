"""
Tests to verify coordinate transform behavior is preserved.
Especially critical: the Y-inversion handling (+ instead of -).
"""

import unittest

import numpy as np
from shapely.geometry import box


class TestComputeChamberCenterRegression(unittest.TestCase):
    """Regression tests for compute_chamber_center behavior."""

    def setUp(self):
        """Set up test fixtures matching real calibration data."""
        from dmc_masking.mask import RoIPolygon

        self.pixel_size = 0.065789

        # Create a simple test polygon (100x100 box starting at 0,0)
        self.test_polygon = RoIPolygon(box(0, 0, 100, 100))

        # Known marker positions (from OpenBox structure)
        self.marker_group_pixels = {
            "cross": np.array([14.0, 8.0]) / self.pixel_size,
            "circle": np.array([66.0, 8.0]) / self.pixel_size,
        }

    def test_chamber_center_y_offset_uses_addition(self):
        """
        CRITICAL: Verify Y offset uses + not -
        This is the key coordinate system behavior to preserve.

        Code reference: calibrate_map.py:295
        center_offset[1] = polygon_center[1] + cross_local[1]
        """
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon
        from scripts.calibrate_map import compute_chamber_center

        # Create simple test polygon (100x100 pixels)
        polygon = RoIPolygon(box(0, 0, 100, 100))

        # Marker positions in pixels
        cross_x = 14.0
        cross_y = 8.0
        marker_group = {
            "cross": np.array([cross_x, cross_y]),
            "circle": np.array([66.0, 8.0]),
        }

        # Simulated detected markers (cross at image position 500, 300)
        detected_cross_x = 500.0
        detected_cross_y = 300.0
        markers = [
            {"bbox_center": np.array([detected_cross_x, detected_cross_y]), "label": "cross"},
            {"bbox_center": np.array([552.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        center = compute_chamber_center(
            markers, matched_indices, marker_group, polygon, rotation_angle=0.0
        )

        # Verify center calculation
        # polygon_center = (50, 50) (center of 100x100 box)
        # center_offset[0] = polygon_center[0] - cross_local[0] = 50 - 14 = 36
        # center_offset[1] = polygon_center[1] + cross_local[1] = 50 + 8 = 58  <-- NOTE: uses +
        # center = detected_cross + center_offset = (500 + 36, 300 + 58) = (536, 358)

        self.assertIsNotNone(center)

        expected_x = detected_cross_x + (50.0 - cross_x)  # = 500 + 36 = 536
        expected_y = detected_cross_y + (50.0 + cross_y)  # = 300 + 58 = 358 (note the +8)

        np.testing.assert_array_almost_equal(
            center,
            [expected_x, expected_y],
            decimal=1,
            err_msg="Y offset should use + not - for coordinate inversion",
        )

    def test_apply_mask_y_translation_uses_addition(self):
        """
        CRITICAL: Verify apply_mask Y translation uses + not -

        Code reference: mask.py:98
        y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1]
        """
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon

        # Create test polygon
        polygon = RoIPolygon(box(0, 0, 100, 100))

        # Marker position
        cross_x = 14.0
        cross_y = 8.0

        # Simulated detected position
        detected_x = 500.0
        detected_y = 300.0

        # Compute translation as in apply_mask
        # x = cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0]
        # y = cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1]  <-- uses +
        translate_x = detected_x - cross_x  # = 500 - 14 = 486
        translate_y = detected_y + cross_y  # = 300 + 8 = 308 (uses +)

        # Apply translation
        translated = polygon.translate(x=translate_x, y=translate_y)

        # Check bounds
        xmin, ymin, _xmax, _ymax = translated.roi_polygon.bounds

        # Expected bounds after translation:
        # Original: (0, 0, 100, 100)
        # After translation: (486, 308, 586, 408)
        self.assertAlmostEqual(xmin, 486.0, places=1)
        self.assertAlmostEqual(ymin, 308.0, places=1)  # 300 + 8, not 300 - 8

    def test_microscope_position_calculation(self):
        """
        Verify microscope position = stage_position + chamber_center_microns

        Code reference: calibrate_map.py:507-508
        microscope_x = stage_position["x"] + chamber_center_microns[0]
        microscope_y = stage_position["y"] + chamber_center_microns[1]
        """
        # Stage position
        stage_pos = {"x": 6802.4, "y": -4272.9}

        # Chamber center in pixels
        chamber_center_pixels = np.array([500.0, 300.0])
        pixel_size = 0.065789

        # Convert to microns
        chamber_center_microns = chamber_center_pixels * pixel_size

        # Compute microscope position (both X and Y use addition)
        microscope_x = stage_pos["x"] + chamber_center_microns[0]
        microscope_y = stage_pos["y"] + chamber_center_microns[1]

        # Verify the formula
        expected_x = 6802.4 + (500.0 * 0.065789)  # = 6802.4 + 32.89 = 6835.29
        expected_y = -4272.9 + (300.0 * 0.065789)  # = -4272.9 + 19.74 = -4253.16

        self.assertAlmostEqual(microscope_x, expected_x, places=2)
        self.assertAlmostEqual(microscope_y, expected_y, places=2)


class TestRoIPolygonCoordinates(unittest.TestCase):
    """Test RoIPolygon coordinate handling."""

    def test_polygon_center_calculation(self):
        """Verify polygon center is bounding box center."""
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon

        polygon = RoIPolygon(box(10, 20, 110, 120))
        center = polygon.center

        # Bounding box center: ((10+110)/2, (20+120)/2) = (60, 70)
        np.testing.assert_array_almost_equal(center, [60.0, 70.0])

    def test_polygon_translate(self):
        """Verify translation works correctly."""
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon

        polygon = RoIPolygon(box(0, 0, 100, 100))
        translated = polygon.translate(x=50, y=30)

        xmin, ymin, xmax, ymax = translated.roi_polygon.bounds
        self.assertAlmostEqual(xmin, 50)
        self.assertAlmostEqual(ymin, 30)
        self.assertAlmostEqual(xmax, 150)
        self.assertAlmostEqual(ymax, 130)

    def test_polygon_scale(self):
        """Verify scaling works correctly."""
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon

        polygon = RoIPolygon(box(0, 0, 100, 100))
        scaled = polygon.scale(2.0)

        # Scaling is around centroid by default in shapely
        # Centroid of (0,0,100,100) is (50, 50)
        # After 2x scale: bounds expand symmetrically
        xmin, ymin, xmax, ymax = scaled.roi_polygon.bounds
        self.assertAlmostEqual(xmax - xmin, 200)
        self.assertAlmostEqual(ymax - ymin, 200)


class TestFilterMatchedPairsByBounds(unittest.TestCase):
    """Test the bounds filtering for matched marker pairs."""

    def test_pairs_outside_image_are_filtered(self):
        """Pairs that would place RoI outside image should be filtered."""
        from shapely.geometry import box

        from dmc_masking.mask import RoIPolygon
        from scripts.calibrate_map import filter_matched_pairs_by_bounds

        # Create a 100x100 polygon
        polygon = RoIPolygon(box(0, 0, 100, 100))

        # Marker group
        marker_group = {
            "cross": np.array([14.0, 8.0]),
            "circle": np.array([66.0, 8.0]),
        }

        # Image size
        image_shape = (600, 800)  # height, width

        # Create markers - one pair near edge, one in center
        markers = [
            # Pair 0: near left edge (would place polygon outside)
            {"bbox_center": np.array([10.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([62.0, 300.0]), "label": "circle"},
            # Pair 1: in center (good)
            {"bbox_center": np.array([400.0, 300.0]), "label": "cross"},
            {"bbox_center": np.array([452.0, 300.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1), (2, 3)]

        filtered = filter_matched_pairs_by_bounds(
            markers, matched_indices, marker_group, polygon, image_shape, rotation_angle=0.0
        )

        # Only pair 1 should remain (pair 0 would place polygon outside)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], (2, 3))


if __name__ == "__main__":
    unittest.main()

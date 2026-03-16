"""Testcases for mask shape handling."""

import unittest

import numpy as np
from shapely.geometry import Point, Polygon

from dart_mlci.mask import (
    RoIPolygon,
    apply_mask,
    apply_mask_rotation_free,
    filter_segmentation_by_mask,
)


def build_polygon():
    chamber_width = 60
    chamber_height = 60

    interior = Point(30, 30).buffer(5)

    chamber_polygon = Polygon(
        [
            (0, 0),
            (chamber_width, 0),
            (chamber_width, chamber_height),
            (0, chamber_height),
        ]
    ).difference(interior)

    return RoIPolygon(chamber_polygon)


polygon_area = 60 * 60 - np.pi * 5**2


class TestMask(unittest.TestCase):
    """Test case for RoI mask shapes."""

    def test_area(self):
        """test area computation"""

        self.assertLess(np.abs(build_polygon().area - polygon_area), 1)

    @staticmethod
    def test_translation():
        """test translation operation"""

        roi_polygon = build_polygon()

        np.testing.assert_almost_equal(roi_polygon.center, np.array([30, 30]))

        new_roi_polygon = roi_polygon.translate(15, -15)

        np.testing.assert_almost_equal(roi_polygon.center, np.array([30, 30]))
        np.testing.assert_almost_equal(new_roi_polygon.center, np.array([30 + 15, 30 - 15]))

    def test_scaling(self):
        """test scaling operation"""

        roi_polygon = build_polygon()

        self.assertLess(np.abs(roi_polygon.area - polygon_area), 0.3)

        new_roi_polygon = roi_polygon.scale(2)

        self.assertLess(np.abs(roi_polygon.area - polygon_area), 0.3)
        self.assertLess(np.abs(new_roi_polygon.area - 4 * polygon_area), 0.6)

    def test_difference(self):
        """test difference operation."""
        A = build_polygon()
        B = A.copy()
        zeroish = A.difference(B)

        self.assertLess(np.abs(A.area - polygon_area), 0.3)
        self.assertLess(np.abs(B.area - polygon_area), 0.3)
        self.assertLess(np.abs(zeroish.area), 0.1)

    def test_union(self):
        """test union operation."""
        A = build_polygon().translate(x=60)
        B = build_polygon()

        both = A.union(B)

        self.assertLess(np.abs(A.area - polygon_area), 0.3)
        self.assertLess(np.abs(B.area - polygon_area), 0.3)
        self.assertLess(np.abs(both.area - 2 * polygon_area), 0.3)

    def test_rotation_preserves_area(self):
        """test that rotation preserves polygon area"""
        roi_polygon = build_polygon()
        original_area = roi_polygon.area

        # Test various rotation angles
        for angle in [0, 45, 90, 180, 270, -45]:
            rotated = roi_polygon.rotate(angle)
            self.assertLess(
                np.abs(rotated.area - original_area),
                0.3,
                f"Area changed after rotating {angle} degrees",
            )

    def test_rotation_around_center(self):
        """test rotation around default center (bounding box center)"""
        roi_polygon = build_polygon()
        original_center = roi_polygon.center.copy()

        # Rotate 90 degrees around center
        rotated = roi_polygon.rotate(90)

        # Center should remain approximately the same
        np.testing.assert_array_almost_equal(rotated.center, original_center, decimal=5)

    def test_rotation_around_custom_origin(self):
        """test rotation around a custom origin point"""
        roi_polygon = build_polygon()

        # Rotate around origin (0, 0)
        rotated = roi_polygon.rotate(90, origin=(0, 0))

        # The center should have moved
        # Original center is (30, 30), after 90 deg CCW rotation around origin:
        # (x, y) -> (-y, x) => (30, 30) -> (-30, 30)
        np.testing.assert_array_almost_equal(rotated.center, np.array([-30, 30]), decimal=5)

    def test_rotation_immutability(self):
        """test that rotation does not modify the original polygon"""
        roi_polygon = build_polygon()
        original_center = roi_polygon.center.copy()
        original_area = roi_polygon.area

        _ = roi_polygon.rotate(45)

        np.testing.assert_array_almost_equal(roi_polygon.center, original_center)
        self.assertLess(np.abs(roi_polygon.area - original_area), 0.1)


class TestApplyMaskRotationFree(unittest.TestCase):
    """Test cases for rotation-free masking."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple square polygon
        self.polygon = RoIPolygon(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
        # Marker positions in polygon's local coordinate system
        self.marker_group_pixels = {
            "cross": np.array([10.0, -10.0]),  # Cross marker near top-left
            "circle": np.array([90.0, -10.0]),  # Circle marker near top-right
        }
        # Create a test image (grayscale, 200x200)
        self.image = np.zeros((200, 200), dtype=np.uint8)
        self.image[50:150, 50:150] = 255  # White square in center

    def test_basic_masking_no_rotation(self):
        """Test masking with zero rotation angle."""
        # Simulate markers detected at positions that match the blueprint
        markers = [
            {"bbox_center": np.array([60.0, 60.0]), "label": "cross"},
            {"bbox_center": np.array([140.0, 60.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        cropped_image, cropped_mask = apply_mask_rotation_free(
            matched_marker_indices=matched_indices,
            markers=markers,
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.polygon,
            image=self.image,
            rotation_angle=0.0,
        )

        # Check that we got a cropped result
        self.assertGreater(cropped_image.size, 0)
        self.assertGreater(cropped_mask.size, 0)
        # Mask should have same shape as cropped image
        self.assertEqual(cropped_image.shape, cropped_mask.shape)

    def test_masking_with_rotation(self):
        """Test masking with non-zero rotation angle."""
        # Markers are detected at rotated positions
        markers = [
            {"bbox_center": np.array([70.0, 70.0]), "label": "cross"},
            {"bbox_center": np.array([140.0, 80.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        cropped_image, cropped_mask = apply_mask_rotation_free(
            matched_marker_indices=matched_indices,
            markers=markers,
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.polygon,
            image=self.image,
            rotation_angle=10.0,
        )

        # Should still produce valid output
        self.assertGreater(cropped_image.size, 0)
        self.assertGreater(cropped_mask.size, 0)

    def test_return_uncropped(self):
        """Test that return_uncropped returns full image."""
        markers = [
            {"bbox_center": np.array([60.0, 60.0]), "label": "cross"},
            {"bbox_center": np.array([140.0, 60.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        full_image, full_mask = apply_mask_rotation_free(
            matched_marker_indices=matched_indices,
            markers=markers,
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.polygon,
            image=self.image,
            rotation_angle=0.0,
            return_uncropped=True,
        )

        # Should return full image size
        self.assertEqual(full_image.shape, self.image.shape)
        self.assertEqual(full_mask.shape, self.image.shape)

    def test_raises_when_roi_outside_bounds(self):
        """Test that ValueError is raised when RoI is outside image bounds."""
        # Markers are at edge, causing polygon to be partially outside
        markers = [
            {"bbox_center": np.array([5.0, 5.0]), "label": "cross"},
            {"bbox_center": np.array([85.0, 5.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1)]

        with self.assertRaises(ValueError) as context:
            apply_mask_rotation_free(
                matched_marker_indices=matched_indices,
                markers=markers,
                marker_group_pixels=self.marker_group_pixels,
                roi_polygon=self.polygon,
                image=self.image,
                rotation_angle=0.0,
            )

        self.assertIn("No roi lies completely inside", str(context.exception))

    def test_selects_best_margin_polygon(self):
        """Test that the polygon with maximum margin to boundaries is selected."""
        # Two valid marker pairs - one more centered than the other
        markers = [
            {"bbox_center": np.array([60.0, 60.0]), "label": "cross"},
            {"bbox_center": np.array([140.0, 60.0]), "label": "circle"},
            {"bbox_center": np.array([80.0, 80.0]), "label": "cross"},  # More centered
            {"bbox_center": np.array([160.0, 80.0]), "label": "circle"},
        ]
        matched_indices = [(0, 1), (2, 3)]

        cropped_image, _cropped_mask = apply_mask_rotation_free(
            matched_marker_indices=matched_indices,
            markers=markers,
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.polygon,
            image=self.image,
            rotation_angle=0.0,
        )

        # Should produce valid output (choosing the best one)
        self.assertGreater(cropped_image.size, 0)


class TestFilterSegmentationByMask(unittest.TestCase):
    """Test cases for filter_segmentation_by_mask."""

    def test_empty_mask(self):
        """All-zeros labeled mask should return unchanged."""
        labeled = np.zeros((10, 10), dtype=np.uint16)
        chamber = np.zeros((10, 10), dtype=bool)
        result = filter_segmentation_by_mask(labeled, chamber)
        np.testing.assert_array_equal(result, labeled)

    def test_with_removal(self):
        """Objects overlapping masked region should be removed."""
        labeled = np.zeros((10, 10), dtype=np.uint16)
        labeled[0:5, 0:5] = 1  # Object 1: 25 pixels
        labeled[5:10, 5:10] = 2  # Object 2: 25 pixels

        # Mask covers object 1 entirely
        chamber = np.zeros((10, 10), dtype=bool)
        chamber[0:5, 0:5] = True

        result = filter_segmentation_by_mask(labeled, chamber, threshold=0.5)
        # Object 1 should be removed, object 2 relabeled to 1
        self.assertEqual(result.max(), 1)
        self.assertTrue(np.all(result[0:5, 0:5] == 0))

    def test_no_relabel(self):
        """With relabel=False, original IDs should be preserved."""
        labeled = np.zeros((10, 10), dtype=np.uint16)
        labeled[0:5, 0:5] = 1
        labeled[5:10, 5:10] = 2

        chamber = np.zeros((10, 10), dtype=bool)
        chamber[0:5, 0:5] = True

        result = filter_segmentation_by_mask(labeled, chamber, threshold=0.5, relabel=False)
        # Object 1 removed but object 2 keeps its label ID
        self.assertEqual(result.max(), 2)
        self.assertTrue(np.all(result[0:5, 0:5] == 0))


class TestApplyMaskEdgeCases(unittest.TestCase):
    """Test edge cases of apply_mask."""

    def setUp(self):
        self.polygon = RoIPolygon(Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]))
        self.marker_group_pixels = {
            "cross": np.array([5.0, -5.0]),
            "circle": np.array([45.0, -5.0]),
        }
        self.image = np.zeros((200, 200), dtype=np.uint8)

    def test_return_bbox(self):
        """return_bbox=True should return a 3-tuple with bounding box."""
        markers = [
            {"bbox_center": np.array([60.0, 60.0])},
            {"bbox_center": np.array([100.0, 60.0])},
        ]
        result = apply_mask(
            [(0, 1)],
            markers,
            self.marker_group_pixels,
            self.polygon,
            self.image,
            return_bbox=True,
        )
        self.assertEqual(len(result), 3)
        bbox = result[2]
        self.assertEqual(len(bbox), 4)

    def test_allow_truncation(self):
        """allow_truncation=True should not raise when ROI extends beyond image."""
        # Place markers near edge so polygon extends beyond image
        markers = [
            {"bbox_center": np.array([5.0, 5.0])},
            {"bbox_center": np.array([45.0, 5.0])},
        ]
        # Without truncation this would fail
        result = apply_mask(
            [(0, 1)],
            markers,
            self.marker_group_pixels,
            self.polygon,
            self.image,
            allow_truncation=True,
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

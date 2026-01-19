"""Test cases for Map.compute_affine_transform method."""

import unittest

import numpy as np

from dmc_masking.map import Map, RoIPosition


class TestComputeAffineTransform(unittest.TestCase):
    """Test cases for Map.compute_affine_transform."""

    def setUp(self):
        """Create standard blueprint map with 3 RoIs."""
        blueprint_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
        ]
        self.blueprint_map = Map(blueprint_rois)

    def test_identity_transform(self):
        """When blueprint and target maps have identical positions, transformation returns points unchanged."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        result = transform(test_point)

        np.testing.assert_array_almost_equal(result, test_point)

    def test_translation_only(self):
        """Target map shifted by constant offset. Verify translation is correctly recovered."""
        offset = np.array([10.0, 20.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        expected = test_point + offset
        result = transform(test_point)

        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_90_degrees(self):
        """Target map rotated 90 degrees counter-clockwise around origin."""
        # 90 degree CCW rotation: (x, y) -> (-y, x)
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),  # (0,0) stays at (0,0)
            RoIPosition("0002", np.array([0.0, 100.0])),  # (100,0) -> (0,100)
            RoIPosition("0003", np.array([-100.0, 0.0])),  # (0,100) -> (-100,0)
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        # Test point (1, 0) should become (0, 1) after 90 degree CCW rotation
        test_point = np.array([[1.0, 0.0]])
        expected = np.array([[0.0, 1.0]])
        result = transform(test_point)

        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_transform(self):
        """Target map scaled by factor of 2. Verify scaling is correctly applied."""
        scale = 2.0
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) * scale),
            RoIPosition("0002", np.array([100.0, 0.0]) * scale),
            RoIPosition("0003", np.array([0.0, 100.0]) * scale),
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        expected = test_point * scale
        result = transform(test_point)

        np.testing.assert_array_almost_equal(result, expected)

    def test_combined_transform(self):
        """Target map with rotation + translation + scale. Verify all transformations compose correctly."""
        # Apply: scale by 2, rotate 90 degrees CCW, translate by (10, 20)
        scale = 2.0
        offset = np.array([10.0, 20.0])

        # After scale by 2, rotate 90 CCW: (x, y) -> (-y*scale, x*scale), then translate
        target_rois = [
            RoIPosition("0001", np.array([0.0 * scale, 0.0 * scale]) + offset),  # (0,0)
            RoIPosition(
                "0002", np.array([0.0 * scale, 100.0 * scale]) + offset
            ),  # (100,0) -> (0,200)
            RoIPosition(
                "0003", np.array([-100.0 * scale, 0.0 * scale]) + offset
            ),  # (0,100) -> (-200,0)
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        # Test: (50, 50) -> scale -> (100, 100) -> rotate 90 CCW -> (-100, 100) -> translate -> (-90, 120)
        test_point = np.array([[50.0, 50.0]])
        expected = np.array([[-90.0, 120.0]])
        result = transform(test_point)

        np.testing.assert_array_almost_equal(result, expected)

    def test_insufficient_rois_raises_assertion(self):
        """Target map with only 2 RoIs should raise AssertionError."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
        ]
        target_map = Map(target_rois)

        with self.assertRaises(AssertionError):
            self.blueprint_map.compute_affine_transform(target_map)

    def test_too_many_rois_raises_assertion(self):
        """Target map with 4+ RoIs should raise AssertionError."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
            RoIPosition("0004", np.array([100.0, 100.0])),
        ]
        target_map = Map(target_rois)

        with self.assertRaises(AssertionError):
            self.blueprint_map.compute_affine_transform(target_map)

    def test_missing_roi_id_raises_keyerror(self):
        """Target map has RoI ID not in blueprint should raise KeyError."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("9999", np.array([0.0, 100.0])),  # ID not in blueprint
        ]
        target_map = Map(target_rois)

        with self.assertRaises(KeyError):
            self.blueprint_map.compute_affine_transform(target_map)

    def test_collinear_points_raises_linalgerror(self):
        """Target map with 3 collinear points should raise numpy.linalg.LinAlgError."""
        # Create a blueprint with collinear points
        collinear_blueprint_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([50.0, 0.0])),
            RoIPosition("0003", np.array([100.0, 0.0])),  # All on x-axis (collinear)
        ]
        collinear_blueprint = Map(collinear_blueprint_rois)

        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([50.0, 0.0])),
            RoIPosition("0003", np.array([100.0, 0.0])),
        ]
        target_map = Map(target_rois)

        with self.assertRaises(np.linalg.LinAlgError):
            collinear_blueprint.compute_affine_transform(target_map)

    def test_returned_function_is_callable(self):
        """Verify return value is a callable function that accepts and returns proper shapes."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        self.assertTrue(callable(transform))

        # Verify it accepts 2D array and returns proper shape
        test_point = np.array([[50.0, 50.0]])
        result = transform(test_point)

        self.assertEqual(result.shape, (1, 2))

    def test_transform_multiple_points(self):
        """Apply transformation to array of multiple points. Verify batch processing works correctly."""
        offset = np.array([10.0, 20.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        transform = self.blueprint_map.compute_affine_transform(target_map)

        # Test with multiple points at once
        test_points = np.array(
            [
                [0.0, 0.0],
                [50.0, 50.0],
                [100.0, 100.0],
                [25.0, 75.0],
            ]
        )
        expected = test_points + offset
        result = transform(test_points)

        self.assertEqual(result.shape, (4, 2))
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

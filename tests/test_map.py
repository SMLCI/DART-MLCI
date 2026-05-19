"""Test cases for the Map / RoIPosition classes and related helpers."""

import logging
import unittest

import numpy as np
import pandas as pd
import pytest

from dart_mlci.map import AffineTransformResult, Map, RoIPosition


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

        result = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, test_point)

    def test_translation_only(self):
        """Target map shifted by constant offset. Verify translation is correctly recovered."""
        offset = np.array([10.0, 20.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        expected = test_point + offset
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, expected)

    def test_rotation_90_degrees(self):
        """Target map rotated 90 degrees counter-clockwise around origin."""
        # 90 degree CCW rotation: (x, y) -> (-y, x)
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),  # (0,0) stays at (0,0)
            RoIPosition("0002", np.array([0.0, 100.0])),  # (100,0) -> (0,100)
            RoIPosition("0003", np.array([-100.0, 0.0])),  # (0,100) -> (-100,0)
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        # Test point (1, 0) should become (0, 1) after 90 degree CCW rotation
        test_point = np.array([[1.0, 0.0]])
        expected = np.array([[0.0, 1.0]])
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, expected)

    def test_scale_transform(self):
        """Target map scaled by factor of 2. Verify scaling is correctly applied."""
        scale = 2.0
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) * scale),
            RoIPosition("0002", np.array([100.0, 0.0]) * scale),
            RoIPosition("0003", np.array([0.0, 100.0]) * scale),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        test_point = np.array([[50.0, 50.0]])
        expected = test_point * scale
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, expected)

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

        result = self.blueprint_map.compute_affine_transform(target_map)

        # Test: (50, 50) -> scale -> (100, 100) -> rotate 90 CCW -> (-100, 100) -> translate -> (-90, 120)
        test_point = np.array([[50.0, 50.0]])
        expected = np.array([[-90.0, 120.0]])
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, expected)

    def test_insufficient_rois_raises_assertion(self):
        """Target map with only 2 RoIs should raise AssertionError."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
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

    def test_collinear_points_still_computes(self):
        """Collinear points with lstsq produce a solution (unlike linalg.solve).

        Note: With the switch to lstsq, collinear points no longer raise an error.
        lstsq handles rank-deficient systems by computing a least-squares solution.
        """
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

        # lstsq handles collinear points without raising an error
        result = collinear_blueprint.compute_affine_transform(target_map)
        self.assertIsInstance(result, AffineTransformResult)

    def test_returns_affine_transform_result(self):
        """Verify return value is an AffineTransformResult with callable transform."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        self.assertIsInstance(result, AffineTransformResult)
        self.assertTrue(callable(result.transform))

        # Verify transform accepts 2D array and returns proper shape
        test_point = np.array([[50.0, 50.0]])
        output = result.transform(test_point)

        self.assertEqual(output.shape, (1, 2))

    def test_transform_multiple_points(self):
        """Apply transformation to array of multiple points. Verify batch processing works correctly."""
        offset = np.array([10.0, 20.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

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
        output = result.transform(test_points)

        self.assertEqual(output.shape, (4, 2))
        np.testing.assert_array_almost_equal(output, expected)

    def test_more_than_3_points(self):
        """Verify that compute_affine_transform works with more than 3 points."""
        # Create blueprint with 4 points
        blueprint_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
            RoIPosition("0004", np.array([100.0, 100.0])),
        ]
        blueprint_map = Map(blueprint_rois)

        # Exact translation - 4 points
        offset = np.array([5.0, 10.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
            RoIPosition("0004", np.array([100.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        result = blueprint_map.compute_affine_transform(target_map)

        # Verify transform works correctly
        test_point = np.array([[50.0, 50.0]])
        expected = test_point + offset
        output = result.transform(test_point)

        np.testing.assert_array_almost_equal(output, expected)

    def test_error_metrics_returned(self):
        """Verify that AffineTransformResult has rmse, residuals, and max_error attributes."""
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        # Verify error metrics exist and have correct types
        self.assertIsInstance(result.rmse, float)
        self.assertIsInstance(result.max_error, float)
        self.assertIsInstance(result.residuals, np.ndarray)
        self.assertEqual(len(result.residuals), 3)

    def test_perfect_fit_zero_error(self):
        """Exact point correspondences should give near-zero error."""
        offset = np.array([10.0, 20.0])
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
        ]
        target_map = Map(target_rois)

        result = self.blueprint_map.compute_affine_transform(target_map)

        # Error should be essentially zero for exact correspondences
        self.assertAlmostEqual(result.rmse, 0.0, places=10)
        self.assertAlmostEqual(result.max_error, 0.0, places=10)
        np.testing.assert_array_almost_equal(result.residuals, np.zeros(3), decimal=10)

    def test_noisy_points_nonzero_error(self):
        """Noisy point correspondences should give nonzero error."""
        # Create blueprint with 4 points
        blueprint_rois = [
            RoIPosition("0001", np.array([0.0, 0.0])),
            RoIPosition("0002", np.array([100.0, 0.0])),
            RoIPosition("0003", np.array([0.0, 100.0])),
            RoIPosition("0004", np.array([100.0, 100.0])),
        ]
        blueprint_map = Map(blueprint_rois)

        # Add noise to one point - creates an inconsistent transformation
        offset = np.array([5.0, 10.0])
        noise = np.array([3.0, -2.0])  # Add noise to one point
        target_rois = [
            RoIPosition("0001", np.array([0.0, 0.0]) + offset),
            RoIPosition("0002", np.array([100.0, 0.0]) + offset),
            RoIPosition("0003", np.array([0.0, 100.0]) + offset),
            RoIPosition("0004", np.array([100.0, 100.0]) + offset + noise),  # Noisy
        ]
        target_map = Map(target_rois)

        result = blueprint_map.compute_affine_transform(target_map)

        # Error should be nonzero due to noise
        self.assertGreater(result.rmse, 0.0)
        self.assertGreater(result.max_error, 0.0)
        # At least one residual should be nonzero
        self.assertTrue(np.any(result.residuals > 0))


def _make_map(entries):
    return Map([RoIPosition(rid, np.array(pos, dtype=float)) for rid, pos in entries])


class TestRoIPosition:
    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Invalid dimensions"):
            RoIPosition("0001", np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_subtraction_returns_delta(self):
        a = RoIPosition("0001", np.array([1.0, 2.0]))
        b = RoIPosition("0002", np.array([4.0, 6.0]))
        np.testing.assert_allclose(b - a, [3.0, 4.0])

    def test_subtraction_dim_mismatch_raises(self):
        a = RoIPosition("0001", np.array([1.0, 2.0]))
        b = RoIPosition("0002", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="different position dimensions"):
            _ = a - b

    def test_repr_contains_id(self):
        r = RoIPosition("0050", np.array([1.0, 2.0]))
        assert "0050" in repr(r)


class TestMapBasics:
    def test_empty_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            Map([])
        assert any("empty map" in r.message.lower() for r in caplog.records)

    def test_getitem(self):
        m = _make_map([("0001", [1.0, 2.0])])
        assert m["0001"].id == "0001"

    def test_distance(self):
        m = _make_map([("0001", [0.0, 0.0]), ("0002", [3.0, 4.0])])
        assert m.distance("0001", "0002") == pytest.approx(5.0)

    def test_rel_movement_with_string_ids(self):
        m = _make_map([("0001", [1.0, 2.0]), ("0002", [4.0, 6.0])])
        np.testing.assert_allclose(m.rel_movement_from_to("0001", "0002"), [3.0, 4.0])

    def test_rel_movement_with_array_from(self):
        m = _make_map([("0002", [4.0, 6.0])])
        out = m.rel_movement_from_to(np.array([1.0, 2.0]), "0002")
        np.testing.assert_allclose(out, [3.0, 4.0])


class TestMapToCsv:
    def test_round_trip_without_z(self, tmp_path):
        m = _make_map([("0001", [1.5, 2.5]), ("0002", [3.0, 4.0])])
        out = tmp_path / "map.csv"
        m.to_csv(out)
        df = pd.read_csv(out)
        assert set(df.columns) == {"roi_id", "x", "y", "z"}
        assert (df["z"] == 0.0).all()
        m2 = Map.from_csv(out)
        assert set(m2.roi_positions.keys()) == {"0001", "0002"}
        np.testing.assert_allclose(m2.roi_positions["0001"].position, [1.5, 2.5])

    def test_with_z_positions_uses_mean_for_missing(self, tmp_path):
        m = _make_map([("0001", [0.0, 0.0]), ("0002", [1.0, 1.0]), ("0003", [2.0, 2.0])])
        out = tmp_path / "map.csv"
        m.to_csv(out, z_positions={"0001": 10.0, "0002": 20.0})
        df = pd.read_csv(out).set_index("roi_id")
        df.index = [f"{i:04d}" for i in df.index]
        assert df.loc["0001", "z"] == 10.0
        assert df.loc["0002", "z"] == 20.0
        # 0003 was missing → mean of provided z's
        assert df.loc["0003", "z"] == pytest.approx(15.0)


if __name__ == "__main__":
    unittest.main()

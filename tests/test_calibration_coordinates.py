"""Tests for the calibration coordinates module."""

import unittest

import numpy as np

from dart_mlci.calibration import (
    AffineTransform2D,
    ImageToStageTransform,
    PixelToMicronTransform,
    apply_rotation_to_offset,
    compute_blueprint_to_image_offset,
)


class TestPixelToMicronTransform(unittest.TestCase):
    """Tests for PixelToMicronTransform."""

    def test_basic_scaling(self):
        """Test basic pixel to micron conversion."""
        transform = PixelToMicronTransform(pixel_size=0.065789)

        pixels = np.array([100.0, 200.0])
        microns = transform(pixels)

        np.testing.assert_array_almost_equal(microns, [6.5789, 13.1578], decimal=4)

    def test_inverse(self):
        """Test inverse transform (microns to pixels)."""
        transform = PixelToMicronTransform(pixel_size=0.065789)

        pixels = np.array([500.0, 300.0])
        microns = transform(pixels)
        back_to_pixels = transform.inverse(microns)

        np.testing.assert_array_almost_equal(back_to_pixels, pixels, decimal=6)

    def test_unit_pixel_size(self):
        """Test identity behavior with pixel_size=1."""
        transform = PixelToMicronTransform(pixel_size=1.0)

        coords = np.array([123.456, 789.012])
        result = transform(coords)

        np.testing.assert_array_almost_equal(result, coords)


class TestImageToStageTransform(unittest.TestCase):
    """Tests for ImageToStageTransform."""

    def test_translation(self):
        """Test stage position translation."""
        transform = ImageToStageTransform(stage_position=np.array([6802.4, -4272.9]))

        image_pos = np.array([32.89, 19.74])
        stage_pos = transform(image_pos)

        expected = np.array([6802.4 + 32.89, -4272.9 + 19.74])
        np.testing.assert_array_almost_equal(stage_pos, expected, decimal=6)

    def test_inverse(self):
        """Test inverse transform (stage to image)."""
        transform = ImageToStageTransform(stage_position=np.array([1000.0, 2000.0]))

        image_pos = np.array([50.0, 75.0])
        stage_pos = transform(image_pos)
        back_to_image = transform.inverse(stage_pos)

        np.testing.assert_array_almost_equal(back_to_image, image_pos, decimal=6)

    def test_zero_stage_position(self):
        """Test with zero stage position (identity)."""
        transform = ImageToStageTransform(stage_position=np.array([0.0, 0.0]))

        image_pos = np.array([100.0, 200.0])
        stage_pos = transform(image_pos)

        np.testing.assert_array_almost_equal(stage_pos, image_pos)


class TestAffineTransform2D(unittest.TestCase):
    """Tests for AffineTransform2D."""

    def test_identity(self):
        """Test identity transform."""
        t = AffineTransform2D.identity()
        point = np.array([10.0, 20.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, point)

    def test_translation(self):
        """Test translation transform."""
        t = AffineTransform2D.translation(50.0, 100.0)
        point = np.array([0.0, 0.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [50.0, 100.0])

    def test_scale_uniform(self):
        """Test uniform scaling."""
        t = AffineTransform2D.scale(2.0)
        point = np.array([10.0, 20.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [20.0, 40.0])

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        t = AffineTransform2D.scale(2.0, 3.0)
        point = np.array([10.0, 20.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [20.0, 60.0])

    def test_rotation_90(self):
        """Test 90-degree rotation."""
        t = AffineTransform2D.rotation(90.0)
        point = np.array([1.0, 0.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [0.0, 1.0], decimal=6)

    def test_rotation_180(self):
        """Test 180-degree rotation."""
        t = AffineTransform2D.rotation(180.0)
        point = np.array([1.0, 2.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [-1.0, -2.0], decimal=6)

    def test_mirror_x(self):
        """Test mirror around Y-axis (flip X)."""
        t = AffineTransform2D.mirror_x()
        point = np.array([10.0, 20.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [-10.0, 20.0])

    def test_mirror_y(self):
        """Test mirror around X-axis (flip Y)."""
        t = AffineTransform2D.mirror_y()
        point = np.array([10.0, 20.0])

        result = t(point)

        np.testing.assert_array_almost_equal(result, [10.0, -20.0])

    def test_inverse_roundtrip(self):
        """Test that inverse @ forward = identity."""
        # Create a complex transform (rotate, scale, translate)
        t = (
            AffineTransform2D.translation(100, 200)
            @ AffineTransform2D.rotation(45)
            @ AffineTransform2D.scale(2.0, 3.0)
        )

        point = np.array([50.0, 75.0])
        transformed = t(point)
        roundtrip = t.inverse(transformed)

        np.testing.assert_array_almost_equal(roundtrip, point, decimal=6)

    def test_composition(self):
        """Test transform composition."""
        # Translate then scale should give different result than scale then translate
        t1 = AffineTransform2D.translation(10, 0)
        t2 = AffineTransform2D.scale(2.0)

        point = np.array([5.0, 5.0])

        # Scale first, then translate: (5*2, 5*2) + (10, 0) = (20, 10)
        result_scale_first = (t1 @ t2)(point)
        np.testing.assert_array_almost_equal(result_scale_first, [20.0, 10.0])

        # Translate first, then scale: (5+10, 5) * 2 = (30, 10)
        result_translate_first = (t2 @ t1)(point)
        np.testing.assert_array_almost_equal(result_translate_first, [30.0, 10.0])

    def test_batch_transform(self):
        """Test transforming multiple points at once."""
        t = AffineTransform2D.translation(10, 20)
        points = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        result = t(points)

        expected = np.array([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_from_point_pairs_exact(self):
        """Test fitting transform from exact correspondences."""
        # Create known transform
        true_transform = AffineTransform2D.translation(100, 50) @ AffineTransform2D.scale(2.0)

        # Generate point pairs
        source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        target = true_transform(source)

        # Fit transform
        fitted, fit_result = AffineTransform2D.from_point_pairs(source, target)

        # Should have zero error
        self.assertAlmostEqual(fit_result.rmse, 0.0, places=6)
        self.assertAlmostEqual(fit_result.max_error, 0.0, places=6)

        # Transform should match
        for i in range(len(source)):
            result = fitted(source[i])
            np.testing.assert_array_almost_equal(result, target[i], decimal=6)

    def test_from_point_pairs_noisy(self):
        """Test fitting transform from noisy correspondences."""
        # Create known transform
        true_transform = AffineTransform2D.translation(100, 50)

        # Generate point pairs with small noise
        source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        target = true_transform(source) + np.random.randn(4, 2) * 0.1  # Small noise

        # Fit transform
        _fitted, fit_result = AffineTransform2D.from_point_pairs(source, target)

        # Should have small error
        self.assertLess(fit_result.rmse, 1.0)

    def test_from_point_pairs_minimum_points(self):
        """Test that at least 3 points are required."""
        source = np.array([[0.0, 0.0], [10.0, 0.0]])
        target = np.array([[0.0, 0.0], [10.0, 0.0]])

        with self.assertRaises(ValueError):
            AffineTransform2D.from_point_pairs(source, target)

    def test_matrix_conversion(self):
        """Test conversion to/from 2x3 matrix."""
        t = AffineTransform2D.translation(10, 20) @ AffineTransform2D.scale(2.0)

        matrix_2x3 = t.to_matrix_2x3()
        self.assertEqual(matrix_2x3.shape, (2, 3))

        t_from_2x3 = AffineTransform2D.from_matrix_2x3(matrix_2x3)

        point = np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(t(point), t_from_2x3(point))


class TestBlueprintToImageOffset(unittest.TestCase):
    """Tests for compute_blueprint_to_image_offset."""

    def test_y_inversion(self):
        """Test that Y is inverted (+ instead of -)."""
        center = np.array([50.0, 50.0])
        marker = np.array([14.0, 8.0])

        offset = compute_blueprint_to_image_offset(center, marker, invert_y=True)

        # offset_x = 50 - 14 = 36
        # offset_y = 50 + 8 = 58 (note the +)
        expected = np.array([36.0, 58.0])
        np.testing.assert_array_almost_equal(offset, expected)

    def test_no_y_inversion(self):
        """Test without Y inversion."""
        center = np.array([50.0, 50.0])
        marker = np.array([14.0, 8.0])

        offset = compute_blueprint_to_image_offset(center, marker, invert_y=False)

        # offset_x = 50 - 14 = 36
        # offset_y = 50 - 8 = 42 (standard subtraction)
        expected = np.array([36.0, 42.0])
        np.testing.assert_array_almost_equal(offset, expected)


class TestApplyRotationToOffset(unittest.TestCase):
    """Tests for apply_rotation_to_offset."""

    def test_zero_rotation(self):
        """Test that zero rotation preserves offset."""
        offset = np.array([10.0, 20.0])

        result = apply_rotation_to_offset(offset, 0.0)

        np.testing.assert_array_almost_equal(result, offset)

    def test_90_degree_rotation(self):
        """Test 90-degree rotation."""
        offset = np.array([10.0, 0.0])

        result = apply_rotation_to_offset(offset, 90.0)

        np.testing.assert_array_almost_equal(result, [0.0, 10.0], decimal=6)

    def test_radians(self):
        """Test rotation in radians."""
        offset = np.array([10.0, 0.0])

        result = apply_rotation_to_offset(offset, np.pi / 2, degrees=False)

        np.testing.assert_array_almost_equal(result, [0.0, 10.0], decimal=6)


if __name__ == "__main__":
    unittest.main()

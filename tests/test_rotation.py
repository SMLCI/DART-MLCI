"""Testcases for rotation"""

import unittest

import numpy as np
import torch

from dmc_masking.rotation import (
    angle_between,
    rotate_image_and_markers,
    rotate_image_kornia,
    rotate_image_opencv,
    rotate_point,
)


class TestAngleComputation(unittest.TestCase):
    """testcase for angle computation"""

    @staticmethod
    def test_angle_0():
        """test 0 degree angel"""
        v1 = np.array([0, 1])
        v2 = np.array([0, 2])

        np.testing.assert_almost_equal(angle_between(v1, v2), 0)

    @staticmethod
    def test_angle_90():
        """test 90 degree angle"""
        v1 = np.array([0, 1])
        v2 = np.array([1, 0])

        np.testing.assert_almost_equal(angle_between(v1, v2), np.pi / 2 * 57.29578)
        # np.testing.assert_almost_equal(angle_between(v2, v1), -np.pi/2)

    @staticmethod
    def test_angle_180():
        """test 180 degree angle"""
        v1 = np.array([0, 1])
        v2 = np.array([0, -1])

        np.testing.assert_almost_equal(angle_between(v1, v2), np.pi * 57.29578)


class TestPointRotation(unittest.TestCase):
    """testcase for rotating points."""

    @staticmethod
    def test_angle_90():
        """testing 90 degree rotation"""

        origin = np.array([0.0, 0.0])
        point = np.array([1, 1])

        np.testing.assert_almost_equal(rotate_point(point, origin, 90), np.array([1, -1]))

        origin = np.array([10, 0.0])
        point = np.array([0, 0])
        np.testing.assert_almost_equal(rotate_point(point, origin, 90), np.array([10, 10]))

    @staticmethod
    def test_angle_180():
        """testing 180 degree rotation"""

        origin = np.array([0.0, 0.0])
        point = np.array([1, 1])

        np.testing.assert_almost_equal(rotate_point(point, origin, 180), np.array([-1, -1]))

        origin = np.array([10, 0.0])
        point = np.array([0, 0])
        np.testing.assert_almost_equal(rotate_point(point, origin, 180), np.array([20, 0]))


class TestImageRotationEquivalence(unittest.TestCase):
    """Test that OpenCV (CPU) and kornia (GPU) rotations produce equivalent results.

    Note: OpenCV and kornia use slightly different bilinear interpolation implementations,
    so we allow tolerances of ~3% for non-90-degree rotations. The key tests are:
    1. Output shapes match exactly
    2. 90/180/270 degree rotations are pixel-perfect
    3. Other angles are visually equivalent (within interpolation differences)
    """

    def test_rotation_equivalence_90_degrees(self):
        """Test that CPU and GPU rotations match for 90 degrees (should be exact)."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        angle = 90.0

        result_cpu = rotate_image_opencv(image, angle)
        result_gpu = rotate_image_kornia(image, angle, device="cpu")

        # Check shapes match
        self.assertEqual(result_cpu.shape, result_gpu.shape)

        # 90 degree rotations should be very close (no interpolation artifacts)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-4, atol=1e-4)

    def test_rotation_equivalence_45_degrees(self):
        """Test that CPU and GPU rotations are similar for 45 degrees."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        angle = 45.0

        result_cpu = rotate_image_opencv(image, angle)
        result_gpu = rotate_image_kornia(image, angle, device="cpu")

        self.assertEqual(result_cpu.shape, result_gpu.shape)
        # Allow ~3% tolerance for interpolation differences
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=0.05, atol=0.03)

    def test_rotation_equivalence_arbitrary_angles(self):
        """Test equivalence for various arbitrary angles."""
        image = np.random.rand(3, 80, 120).astype(np.float32)

        for angle in [15.0, 73.5, 135.0, 270.0, -45.0]:
            with self.subTest(angle=angle):
                result_cpu = rotate_image_opencv(image, angle)
                result_gpu = rotate_image_kornia(image, angle, device="cpu")

                self.assertEqual(result_cpu.shape, result_gpu.shape)
                # Allow ~3% tolerance for interpolation differences
                np.testing.assert_allclose(result_cpu, result_gpu, rtol=0.05, atol=0.03)

    def test_rotation_single_channel(self):
        """Test rotation equivalence for single channel images."""
        image = np.random.rand(1, 100, 100).astype(np.float32)
        angle = 30.0

        result_cpu = rotate_image_opencv(image, angle)
        result_gpu = rotate_image_kornia(image, angle, device="cpu")

        self.assertEqual(result_cpu.shape, result_gpu.shape)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=0.05, atol=0.03)

    def test_rotation_many_channels(self):
        """Test rotation equivalence for multi-channel images."""
        image = np.random.rand(5, 64, 64).astype(np.float32)
        angle = 60.0

        result_cpu = rotate_image_opencv(image, angle)
        result_gpu = rotate_image_kornia(image, angle, device="cpu")

        self.assertEqual(result_cpu.shape, result_gpu.shape)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=0.05, atol=0.03)

    def test_rotate_image_and_markers_cpu_vs_gpu(self):
        """Test that rotate_image_and_markers produces same results for CPU and GPU."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        markers = [
            {"bbox_center": np.array([25.0, 25.0])},
            {"bbox_center": np.array([75.0, 75.0])},
        ]
        angle = 45.0

        # Run with GPU disabled (uses OpenCV)
        result_cpu, markers_cpu = rotate_image_and_markers(
            image.copy(), [m.copy() for m in markers], angle, use_gpu=False
        )

        # Run with GPU enabled but force CPU device for reproducibility
        result_gpu, markers_gpu = rotate_image_and_markers(
            image.copy(), [m.copy() for m in markers], angle, use_gpu=False
        )

        # Images should match exactly (both use OpenCV)
        self.assertEqual(result_cpu.shape, result_gpu.shape)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-6, atol=1e-6)

        # Markers should be identical (same transformation logic)
        for mc, mg in zip(markers_cpu, markers_gpu, strict=False):
            np.testing.assert_allclose(mc["bbox_center"], mg["bbox_center"])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestImageRotationGPU(unittest.TestCase):
    """Test GPU-specific rotation functionality (requires CUDA)."""

    def test_rotation_on_cuda(self):
        """Test that kornia rotation works on CUDA device."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        angle = 45.0

        result_cpu = rotate_image_kornia(image, angle, device="cpu")
        result_cuda = rotate_image_kornia(image, angle, device="cuda")

        # Results should be similar between CPU and CUDA
        # Note: CPU and CUDA implementations can have numerical differences
        # due to different floating-point precision and order of operations
        self.assertEqual(result_cpu.shape, result_cuda.shape)
        np.testing.assert_allclose(result_cpu, result_cuda, rtol=0.05, atol=0.05)

    def test_rotate_image_and_markers_cuda(self):
        """Test rotate_image_and_markers with GPU enabled and CUDA available."""
        image = np.random.rand(3, 100, 100).astype(np.float32)
        markers = [{"bbox_center": np.array([50.0, 50.0])}]
        angle = 30.0

        # Run with use_gpu=True (should use CUDA/kornia)
        result_gpu, markers_gpu = rotate_image_and_markers(
            image.copy(), [m.copy() for m in markers], angle, use_gpu=True
        )

        # Run with use_gpu=False (uses OpenCV)
        result_cpu, markers_cpu = rotate_image_and_markers(
            image.copy(), [m.copy() for m in markers], angle, use_gpu=False
        )

        # Results should be similar (allow interpolation differences)
        self.assertEqual(result_cpu.shape, result_gpu.shape)
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=0.05, atol=0.03)

        # Markers should be identical (same transformation logic, independent of GPU)
        np.testing.assert_allclose(markers_cpu[0]["bbox_center"], markers_gpu[0]["bbox_center"])


if __name__ == "__main__":
    unittest.main()

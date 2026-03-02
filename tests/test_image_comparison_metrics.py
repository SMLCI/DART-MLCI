"""Unit tests for image comparison metrics.

Tests the core functionality of SSIM, PSNR, and MSE computation.
"""

import numpy as np
import pytest

from tests.utils import (
    ImageComparisonResult,
    compare_images,
    compute_mse,
    compute_psnr,
    compute_ssim,
)


class TestSSIM:
    """Test SSIM computation."""

    def test_identical_images_uint8(self):
        """Identical images should have SSIM of 1.0."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ssim = compute_ssim(image, image.copy())
        assert ssim == 1.0

    def test_identical_images_float(self):
        """Identical float images should have SSIM of 1.0."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        ssim = compute_ssim(image, image.copy())
        assert ssim == 1.0

    def test_different_images(self):
        """Different images should have SSIM < 1.0."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        ssim = compute_ssim(image1, image2)
        assert 0.0 <= ssim < 1.0

    def test_slightly_different_images(self):
        """Slightly different images should have high SSIM."""
        image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Add small noise
        noise = np.random.randint(-5, 6, (100, 100, 3), dtype=np.int16)
        image2 = np.clip(image1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ssim = compute_ssim(image1, image2)
        assert ssim > 0.9  # Should be very similar

    def test_grayscale_images(self):
        """Test SSIM with grayscale images."""
        image1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image2 = image1.copy()
        ssim = compute_ssim(image1, image2, channel_axis=None)
        assert ssim == 1.0

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.zeros((100, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes must match"):
            compute_ssim(image1, image2)


class TestPSNR:
    """Test PSNR computation."""

    def test_identical_images(self):
        """Identical images should have infinite PSNR."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        psnr = compute_psnr(image, image.copy())
        assert np.isinf(psnr)

    def test_different_images(self):
        """Different images should have finite PSNR."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        psnr = compute_psnr(image1, image2)
        assert np.isfinite(psnr)
        assert psnr < 50  # Should be relatively low for very different images

    def test_noisy_image(self):
        """PSNR should be reasonable for noisy images."""
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add Gaussian noise
        noise = np.random.normal(0, 10, (100, 100, 3))
        image2 = np.clip(image1 + noise, 0, 255).astype(np.uint8)

        psnr = compute_psnr(image1, image2)
        assert 20 < psnr < 40  # Reasonable range for moderate noise

    def test_uint16_images(self):
        """Test PSNR with uint16 images."""
        image1 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        image2 = image1.copy()
        psnr = compute_psnr(image1, image2)
        assert np.isinf(psnr)

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((100, 200), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes must match"):
            compute_psnr(image1, image2)


class TestMSE:
    """Test MSE computation."""

    def test_identical_images(self):
        """Identical images should have MSE of 0."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mse = compute_mse(image, image.copy())
        assert mse == 0.0

    def test_different_images(self):
        """Different images should have MSE > 0."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mse = compute_mse(image1, image2)
        assert mse > 0
        assert mse <= 1.0  # Normalized MSE

    def test_mse_computation(self):
        """Verify MSE computation matches expected value."""
        # Create simple test case
        image1 = np.zeros((10, 10), dtype=np.float32)
        image2 = np.ones((10, 10), dtype=np.float32) * 0.5

        mse = compute_mse(image1, image2)
        expected_mse = 0.25  # (0.5)^2 = 0.25
        assert abs(mse - expected_mse) < 1e-6

    def test_dtype_normalization(self):
        """MSE should normalize different dtypes correctly."""
        # uint8: 128 is 128/255 ≈ 0.502
        image1_u8 = np.ones((10, 10), dtype=np.uint8) * 128

        # float: 0.5
        image1_f32 = np.ones((10, 10), dtype=np.float32) * 0.5

        # Should have very similar MSE against zero
        image2 = np.zeros((10, 10), dtype=np.uint8)

        mse_u8 = compute_mse(image1_u8, image2)
        mse_f32 = compute_mse(image1_f32, image2)

        assert abs(mse_u8 - mse_f32) < 0.01  # Close due to normalization

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((200, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes must match"):
            compute_mse(image1, image2)


class TestCompareImages:
    """Test the unified compare_images function."""

    def test_identical_images(self):
        """Compare identical images."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = compare_images(image, image.copy())

        assert result.ssim == 1.0
        assert np.isinf(result.psnr)
        assert result.mse == 0.0
        assert result.shape_match is True
        assert result.dtype_match is True
        assert result.images_identical is True

    def test_different_images(self):
        """Compare different images."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = compare_images(image1, image2)

        assert result.ssim < 1.0
        assert np.isfinite(result.psnr)
        assert result.mse > 0
        assert result.shape_match is True
        assert result.dtype_match is True
        assert result.images_identical is False

    def test_shape_mismatch(self):
        """Compare images with different shapes."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.zeros((100, 200, 3), dtype=np.uint8)
        result = compare_images(image1, image2)

        assert result.ssim is None
        assert result.psnr is None
        assert result.mse is None
        assert result.shape_match is False
        assert result.dtype_match is True
        assert result.images_identical is False

    def test_dtype_mismatch(self):
        """Compare images with different dtypes."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.zeros((100, 100, 3), dtype=np.uint16)
        result = compare_images(image1, image2)

        assert result.shape_match is True
        assert result.dtype_match is False

    def test_selective_metrics(self):
        """Test computing only selected metrics."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Only SSIM
        result = compare_images(image, image.copy(), metrics=["ssim"])
        assert result.ssim == 1.0
        assert result.psnr is None
        assert result.mse is None

        # Only PSNR and MSE
        result = compare_images(image, image.copy(), metrics=["psnr", "mse"])
        assert result.ssim is None
        assert np.isinf(result.psnr)
        assert result.mse == 0.0


class TestImageComparisonResult:
    """Test ImageComparisonResult dataclass methods."""

    def test_passes_threshold_all_pass(self):
        """Test threshold checking when all metrics pass."""
        result = ImageComparisonResult(
            ssim=0.96,
            psnr=32.0,
            mse=0.0005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        assert result.passes_threshold(
            ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001
        )

    def test_passes_threshold_ssim_fail(self):
        """Test threshold checking when SSIM fails."""
        result = ImageComparisonResult(
            ssim=0.90,
            psnr=32.0,
            mse=0.0005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        assert not result.passes_threshold(
            ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001
        )

    def test_passes_threshold_psnr_fail(self):
        """Test threshold checking when PSNR fails."""
        result = ImageComparisonResult(
            ssim=0.96,
            psnr=25.0,
            mse=0.0005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        assert not result.passes_threshold(
            ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001
        )

    def test_passes_threshold_mse_fail(self):
        """Test threshold checking when MSE fails."""
        result = ImageComparisonResult(
            ssim=0.96,
            psnr=32.0,
            mse=0.005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        assert not result.passes_threshold(
            ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001
        )

    def test_passes_threshold_partial(self):
        """Test threshold checking with only some thresholds specified."""
        result = ImageComparisonResult(
            ssim=0.96,
            psnr=25.0,  # Would fail if checked
            mse=0.0005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        # Only check SSIM and MSE, ignore PSNR
        assert result.passes_threshold(ssim_threshold=0.95, mse_threshold=0.001)

    def test_format_report(self):
        """Test formatting comparison results."""
        result = ImageComparisonResult(
            ssim=0.96,
            psnr=32.5,
            mse=0.0005,
            shape_match=True,
            dtype_match=True,
            images_identical=False,
        )

        report = result.format_report(ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001)

        assert "SSIM: 0.9600" in report
        assert "PSNR: 32.50 dB" in report
        assert "MSE: 0.000500" in report
        assert "✓ PASS" in report  # All should pass
        assert "Shape match: ✓" in report
        assert "Dtype match: ✓" in report

    def test_format_report_with_failures(self):
        """Test formatting report with failed thresholds."""
        result = ImageComparisonResult(
            ssim=0.90,
            psnr=25.0,
            mse=0.005,
            shape_match=True,
            dtype_match=False,
            images_identical=False,
        )

        report = result.format_report(ssim_threshold=0.95, psnr_threshold=30.0, mse_threshold=0.001)

        assert "❌ FAIL" in report
        assert "Dtype match: ❌" in report

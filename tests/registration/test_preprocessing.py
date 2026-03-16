"""Tests for registration preprocessing utilities."""

import cv2
import numpy as np
import pytest

from dart_mlci.registration.preprocessing import (
    apply_bilateral_filter,
    compute_image_gradient_magnitude,
    create_hanning_window,
    enhance_contrast,
    normalize_to_range,
    preprocess_for_registration,
)


class TestPreprocessForRegistration:
    """Test preprocessing pipeline."""

    def test_grayscale_input(self):
        """Test preprocessing with grayscale input."""
        # Create test image
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Preprocess
        result = preprocess_for_registration(image, use_clahe=True, normalize=True)

        # Check output
        assert result.dtype == np.float32
        assert result.shape == image.shape
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_rgb_input(self):
        """Test preprocessing with RGB input."""
        # Create test image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Preprocess
        result = preprocess_for_registration(image, use_clahe=True, normalize=True)

        # Check output (should be grayscale)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.shape == (100, 100)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_without_clahe(self):
        """Test preprocessing without CLAHE."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Preprocess without CLAHE
        result = preprocess_for_registration(image, use_clahe=False, normalize=True)

        # Should still normalize
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_clahe_enhances_contrast(self):
        """Test that CLAHE improves contrast in low-contrast images."""
        # Create low-contrast image (narrow intensity range)
        low_contrast = np.random.randint(100, 120, (100, 100), dtype=np.uint8)

        # Preprocess with and without CLAHE
        without_clahe = preprocess_for_registration(low_contrast, use_clahe=False, normalize=True)
        with_clahe = preprocess_for_registration(low_contrast, use_clahe=True, normalize=True)

        # CLAHE should increase intensity range
        range_without = without_clahe.max() - without_clahe.min()
        range_with = with_clahe.max() - with_clahe.min()

        assert range_with > range_without

    def test_float_input(self):
        """Test preprocessing with float input."""
        # Create float image
        image = np.random.rand(100, 100).astype(np.float32)

        # Preprocess
        result = preprocess_for_registration(image, use_clahe=True, normalize=True)

        # Check output
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_constant_image(self):
        """Test preprocessing on constant (uniform) image."""
        # Constant image
        image = np.full((100, 100), 128, dtype=np.uint8)

        # Should not crash
        result = preprocess_for_registration(image, use_clahe=True, normalize=True)

        assert result.dtype == np.float32
        assert result.shape == image.shape


class TestBilateralFilter:
    """Test bilateral filtering."""

    def test_float32_input(self):
        """Test bilateral filter with float32 input."""
        image = np.random.rand(100, 100).astype(np.float32)

        result = apply_bilateral_filter(image)

        assert result.dtype == np.float32
        assert result.shape == image.shape
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_uint8_input(self):
        """Test bilateral filter with uint8 input."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        result = apply_bilateral_filter(image)

        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_smooths_noise(self):
        """Test that bilateral filter reduces noise."""
        # Create noisy image
        clean = np.ones((100, 100), dtype=np.float32) * 0.5
        noise = np.random.randn(100, 100).astype(np.float32) * 0.1
        noisy = np.clip(clean + noise, 0, 1)

        # Apply filter
        filtered = apply_bilateral_filter(noisy)

        # Standard deviation should be reduced
        assert filtered.std() < noisy.std()


class TestEnhanceContrast:
    """Test CLAHE contrast enhancement."""

    def test_enhances_low_contrast(self):
        """Test contrast enhancement on low-contrast image."""
        # Low contrast image
        low_contrast = np.random.randint(100, 120, (100, 100), dtype=np.uint8)

        # Enhance
        enhanced = enhance_contrast(low_contrast)

        # Should expand intensity range
        assert enhanced.max() - enhanced.min() > low_contrast.max() - low_contrast.min()
        assert enhanced.dtype == np.uint8

    def test_non_uint8_raises_error(self):
        """Test that non-uint8 input raises error."""
        image = np.random.rand(100, 100).astype(np.float32)

        with pytest.raises(ValueError, match="must be uint8"):
            enhance_contrast(image)


class TestHanningWindow:
    """Test Hanning window creation."""

    def test_window_shape(self):
        """Test window has correct shape."""
        window = create_hanning_window((128, 256))

        assert window.shape == (128, 256)
        assert window.dtype == np.float32

    def test_window_range(self):
        """Test window values in [0, 1] range."""
        window = create_hanning_window((100, 100))

        assert 0.0 <= window.min() <= window.max() <= 1.0

    def test_window_center_higher(self):
        """Test that window center has higher values than edges."""
        window = create_hanning_window((100, 100))

        center_val = window[50, 50]
        edge_val = window[0, 0]

        assert center_val > edge_val

    def test_square_vs_rectangular(self):
        """Test window works for both square and rectangular shapes."""
        square = create_hanning_window((100, 100))
        rect = create_hanning_window((100, 200))

        assert square.shape == (100, 100)
        assert rect.shape == (100, 200)


class TestNormalizeToRange:
    """Test normalization function."""

    def test_normalize_to_01(self):
        """Test normalization to [0, 1] range."""
        image = np.random.randint(50, 200, (100, 100)).astype(np.float32)

        normalized = normalize_to_range(image, 0.0, 1.0)

        assert normalized.min() == pytest.approx(0.0, abs=1e-5)
        assert normalized.max() == pytest.approx(1.0, abs=1e-5)
        assert normalized.dtype == np.float32

    def test_normalize_to_custom_range(self):
        """Test normalization to custom range."""
        image = np.random.randint(0, 256, (100, 100)).astype(np.float32)

        normalized = normalize_to_range(image, -1.0, 1.0)

        assert normalized.min() == pytest.approx(-1.0, abs=1e-5)
        assert normalized.max() == pytest.approx(1.0, abs=1e-5)

    def test_constant_image(self):
        """Test normalization on constant image."""
        image = np.full((100, 100), 128.0, dtype=np.float32)

        # Should not crash, returns target_min
        normalized = normalize_to_range(image, 0.0, 1.0)

        assert np.all(normalized == 0.0)
        assert normalized.dtype == np.float32


class TestGradientMagnitude:
    """Test gradient magnitude computation."""

    def test_gradient_shape(self):
        """Test gradient has same shape as input."""
        image = np.random.rand(100, 100).astype(np.float32)

        gradient = compute_image_gradient_magnitude(image)

        assert gradient.shape == image.shape
        assert gradient.dtype == np.float32

    def test_gradient_non_negative(self):
        """Test gradient magnitude is non-negative."""
        image = np.random.rand(100, 100).astype(np.float32)

        gradient = compute_image_gradient_magnitude(image)

        assert gradient.min() >= 0.0

    def test_gradient_detects_edges(self):
        """Test gradient is high at edges."""
        # Create image with sharp edge
        image = np.zeros((100, 100), dtype=np.float32)
        image[:, 50:] = 1.0

        gradient = compute_image_gradient_magnitude(image)

        # Gradient should be high near x=50
        edge_gradient = gradient[:, 48:52].mean()
        interior_gradient = gradient[:, 10:20].mean()

        assert edge_gradient > interior_gradient * 5


class TestPreprocessingIntegration:
    """Integration tests combining multiple preprocessing steps."""

    def test_full_pipeline(self):
        """Test full preprocessing pipeline."""
        # Create test image
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Full pipeline
        preprocessed = preprocess_for_registration(image, use_clahe=True, normalize=True)

        # Apply Hanning window
        window = create_hanning_window(preprocessed.shape)
        windowed = preprocessed * window

        # Check results
        assert windowed.dtype == np.float32
        assert windowed.shape == (200, 200)
        assert 0.0 <= windowed.min() <= windowed.max() <= 1.0

    def test_preprocessing_improves_registration_markers(self):
        """Test preprocessing on synthetic marker pattern."""
        # Create synthetic marker pattern
        image = np.ones((200, 200), dtype=np.uint8) * 100

        # Add low-contrast markers
        cv2.circle(image, (50, 50), 10, 120, -1)
        cv2.circle(image, (150, 50), 10, 120, -1)
        cv2.circle(image, (50, 150), 10, 120, -1)
        cv2.circle(image, (150, 150), 10, 120, -1)

        # Preprocess without CLAHE
        no_clahe = preprocess_for_registration(image, use_clahe=False, normalize=True)

        # Preprocess with CLAHE
        with_clahe = preprocess_for_registration(image, use_clahe=True, normalize=True)

        # Marker regions should have higher contrast with CLAHE
        marker_region = with_clahe[40:60, 40:60]
        background_region = with_clahe[100:120, 100:120]

        contrast_with = abs(marker_region.mean() - background_region.mean())

        marker_region_no = no_clahe[40:60, 40:60]
        background_region_no = no_clahe[100:120, 100:120]

        contrast_without = abs(marker_region_no.mean() - background_region_no.mean())

        # CLAHE should increase contrast
        assert contrast_with > contrast_without

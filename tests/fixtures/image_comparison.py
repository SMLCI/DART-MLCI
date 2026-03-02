"""Pytest fixtures for image comparison and visual regression testing."""

import os

import pytest


@pytest.fixture
def golden_image_dir(tmp_path):
    """Temporary directory for testing golden image operations.

    This fixture provides a temporary directory that can be used
    for testing golden image save/load without affecting real golden files.

    Args:
        tmp_path: pytest's temporary directory fixture.

    Returns:
        Path to temporary golden images directory.

    Example:
        >>> def test_save_load(golden_image_dir):
        ...     # Use golden_image_dir for testing
        ...     pass
    """
    golden_dir = tmp_path / "golden" / "images"
    golden_dir.mkdir(parents=True, exist_ok=True)
    return golden_dir


@pytest.fixture
def image_comparison_thresholds():
    """Default thresholds for image comparison tests.

    Returns a dictionary of default thresholds for different image types.
    Tests can override these values as needed.

    Returns:
        Dictionary with threshold configurations for different categories.

    Example:
        >>> def test_calibration(image_comparison_thresholds):
        ...     thresholds = image_comparison_thresholds["calibration"]
        ...     assert_images_equal(..., **thresholds)
    """
    return {
        "calibration": {
            "ssim_threshold": 0.95,
            "psnr_threshold": 30.0,
            "mse_threshold": 0.001,
        },
        "registration": {
            "ssim_threshold": 0.90,
            "psnr_threshold": 28.0,
            "mse_threshold": 0.005,
        },
        "masking": {
            "ssim_threshold": 0.95,
            "psnr_threshold": 30.0,
            "mse_threshold": 0.001,
        },
        "pipeline": {
            "ssim_threshold": 0.92,
            "psnr_threshold": 28.0,
            "mse_threshold": 0.003,
        },
    }


@pytest.fixture
def regenerate_golden_images():
    """Check if golden images should be regenerated.

    This fixture reads the REGENERATE_GOLDEN environment variable.
    If set to "1", tests should regenerate golden images instead of comparing.

    Returns:
        True if golden images should be regenerated, False otherwise.

    Example:
        >>> def test_output(regenerate_golden_images):
        ...     if regenerate_golden_images:
        ...         save_golden_image(...)
        ...         pytest.skip("Regenerating golden images")
    """
    return os.environ.get("REGENERATE_GOLDEN", "0") == "1"


@pytest.fixture
def diff_output_dir(tmp_path):
    """Temporary directory for saving diff images during tests.

    Args:
        tmp_path: pytest's temporary directory fixture.

    Returns:
        Path to temporary diff output directory.
    """
    diff_dir = tmp_path / "visual_regression"
    diff_dir.mkdir(parents=True, exist_ok=True)
    return diff_dir

"""Test utilities for DMC Masking.

This package provides utilities for testing image processing pipelines,
including pixel-level image comparison and visual regression testing.
"""

from tests.utils.image_comparison import (
    ImageComparisonResult,
    compare_images,
    compute_mse,
    compute_psnr,
    compute_ssim,
)
from tests.utils.visual_regression import (
    DEFAULT_MSE_THRESHOLD,
    DEFAULT_PSNR_THRESHOLD,
    DEFAULT_SSIM_THRESHOLD,
    assert_images_equal,
    generate_diff_image,
    load_golden_image,
    save_golden_image,
)

__all__ = [
    "DEFAULT_MSE_THRESHOLD",
    "DEFAULT_PSNR_THRESHOLD",
    "DEFAULT_SSIM_THRESHOLD",
    "ImageComparisonResult",
    "assert_images_equal",
    "compare_images",
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "generate_diff_image",
    "load_golden_image",
    "save_golden_image",
]

"""Core image comparison metrics for visual regression testing.

This module provides functions to compute structural similarity (SSIM),
peak signal-to-noise ratio (PSNR), and mean squared error (MSE) between images.
"""

import contextlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class ImageComparisonResult:
    """Result of comparing two images.

    Attributes:
        ssim: Structural similarity index (0-1), where 1 is identical.
        psnr: Peak signal-to-noise ratio in dB. Higher is better.
        mse: Mean squared error (normalized). Lower is better.
        shape_match: Whether image shapes match.
        dtype_match: Whether image dtypes match.
        images_identical: Whether images are exactly identical.
    """

    ssim: float | None
    psnr: float | None
    mse: float | None
    shape_match: bool
    dtype_match: bool
    images_identical: bool

    def passes_threshold(
        self,
        ssim_threshold: float | None = None,
        psnr_threshold: float | None = None,
        mse_threshold: float | None = None,
    ) -> bool:
        """Check if metrics pass given thresholds.

        Args:
            ssim_threshold: Minimum SSIM value (0-1). None to skip check.
            psnr_threshold: Minimum PSNR value in dB. None to skip check.
            mse_threshold: Maximum MSE value. None to skip check.

        Returns:
            True if all specified thresholds are met.
        """
        if ssim_threshold is not None and (self.ssim is None or self.ssim < ssim_threshold):
            return False

        if psnr_threshold is not None and (self.psnr is None or self.psnr < psnr_threshold):
            return False

        return not (mse_threshold is not None and (self.mse is None or self.mse > mse_threshold))

    def format_report(
        self,
        ssim_threshold: float | None = None,
        psnr_threshold: float | None = None,
        mse_threshold: float | None = None,
    ) -> str:
        """Format comparison results as a human-readable report.

        Args:
            ssim_threshold: SSIM threshold for pass/fail indicator.
            psnr_threshold: PSNR threshold for pass/fail indicator.
            mse_threshold: MSE threshold for pass/fail indicator.

        Returns:
            Multi-line string with formatted comparison metrics.
        """
        lines = ["Comparison metrics:"]

        if self.ssim is not None:
            status = ""
            if ssim_threshold is not None:
                status = " ✓ PASS" if self.ssim >= ssim_threshold else " ❌ FAIL"
            lines.append(
                f"  - SSIM: {self.ssim:.4f}"
                + (f" (threshold: {ssim_threshold:.2f})" if ssim_threshold else "")
                + status
            )

        if self.psnr is not None:
            status = ""
            if psnr_threshold is not None:
                status = " ✓ PASS" if self.psnr >= psnr_threshold else " ❌ FAIL"
            psnr_str = f"{self.psnr:.2f} dB" if not np.isinf(self.psnr) else "inf (identical)"
            lines.append(
                f"  - PSNR: {psnr_str}"
                + (f" (threshold: {psnr_threshold:.1f} dB)" if psnr_threshold else "")
                + status
            )

        if self.mse is not None:
            status = ""
            if mse_threshold is not None:
                status = " ✓ PASS" if self.mse <= mse_threshold else " ❌ FAIL"
            lines.append(
                f"  - MSE: {self.mse:.6f}"
                + (f" (threshold: {mse_threshold:.6f})" if mse_threshold else "")
                + status
            )

        lines.append("")
        lines.append(f"Shape match: {'✓' if self.shape_match else '❌'}")
        lines.append(f"Dtype match: {'✓' if self.dtype_match else '❌'}")
        lines.append(f"Identical: {'✓' if self.images_identical else '❌'}")

        return "\n".join(lines)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in range [0, 1].

    Args:
        image: Input image (any dtype).

    Returns:
        Image as float32 in [0, 1] range.
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            return image.astype(np.float32)
        else:
            return (image / image.max()).astype(np.float32)
    else:
        return image.astype(np.float32)


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    channel_axis: int | None = -1,
) -> float:
    """Compute structural similarity index (SSIM) between two images.

    SSIM measures perceived quality degradation caused by changes in
    structural information, luminance, and contrast.

    Args:
        image1: First image (HxW or HxWxC).
        image2: Second image (same shape as image1).
        channel_axis: Axis for color channels. -1 for last axis, None for grayscale.

    Returns:
        SSIM value between 0 and 1, where 1 means identical.

    Raises:
        ValueError: If image shapes don't match.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")

    img1_norm = _normalize_image(image1)
    img2_norm = _normalize_image(image2)

    # Handle channel axis for multi-channel images
    if img1_norm.ndim == 3 and channel_axis is not None:
        return ssim(img1_norm, img2_norm, channel_axis=channel_axis, data_range=1.0)
    else:
        return ssim(img1_norm, img2_norm, data_range=1.0)


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute peak signal-to-noise ratio (PSNR) between two images.

    PSNR measures the ratio between the maximum possible signal power
    and the noise power (MSE). Higher values indicate better quality.

    Args:
        image1: First image (any dtype).
        image2: Second image (same shape as image1).

    Returns:
        PSNR in decibels (dB). Returns infinity if images are identical.

    Raises:
        ValueError: If image shapes don't match.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")

    img1_norm = _normalize_image(image1)
    img2_norm = _normalize_image(image2)

    mse = np.mean((img1_norm - img2_norm) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute mean squared error (MSE) between two images.

    MSE is the average squared difference between pixel values.
    Images are normalized to [0, 1] before comparison.

    Args:
        image1: First image (any dtype).
        image2: Second image (same shape as image1).

    Returns:
        MSE value (normalized). 0 means identical images.

    Raises:
        ValueError: If image shapes don't match.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")

    img1_norm = _normalize_image(image1)
    img2_norm = _normalize_image(image2)

    return float(np.mean((img1_norm - img2_norm) ** 2))


def compare_images(
    image1: np.ndarray,
    image2: np.ndarray,
    metrics: list[Literal["ssim", "psnr", "mse"]] | None = None,
) -> ImageComparisonResult:
    """Compare two images using multiple metrics.

    Args:
        image1: First image.
        image2: Second image.
        metrics: List of metrics to compute. If None, computes all.
            Options: "ssim", "psnr", "mse".

    Returns:
        ImageComparisonResult with computed metrics and metadata.
    """
    if metrics is None:
        metrics = ["ssim", "psnr", "mse"]

    shape_match = image1.shape == image2.shape
    dtype_match = image1.dtype == image2.dtype
    images_identical = np.array_equal(image1, image2)

    ssim_value = None
    psnr_value = None
    mse_value = None

    if shape_match:
        if "ssim" in metrics:
            with contextlib.suppress(Exception):
                channel_axis = -1 if image1.ndim == 3 else None
                ssim_value = compute_ssim(image1, image2, channel_axis=channel_axis)

        if "psnr" in metrics:
            with contextlib.suppress(Exception):
                psnr_value = compute_psnr(image1, image2)

        if "mse" in metrics:
            with contextlib.suppress(Exception):
                mse_value = compute_mse(image1, image2)

    return ImageComparisonResult(
        ssim=ssim_value,
        psnr=psnr_value,
        mse=mse_value,
        shape_match=shape_match,
        dtype_match=dtype_match,
        images_identical=images_identical,
    )

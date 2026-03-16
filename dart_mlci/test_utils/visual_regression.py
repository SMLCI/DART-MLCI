"""Visual regression testing helpers.

This module provides utilities for golden image comparison, including
saving/loading golden images and asserting image equality with thresholds.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from dart_mlci.test_utils.image_comparison import compare_images

# Default thresholds for visual regression testing
DEFAULT_SSIM_THRESHOLD = 0.95  # 95% structural similarity
DEFAULT_PSNR_THRESHOLD = 30.0  # 30 dB (good quality)
DEFAULT_MSE_THRESHOLD = 0.001  # Normalized MSE < 0.1%


def get_golden_image_dir() -> Path:
    """Get the golden images directory path.

    Returns:
        Path to tests/golden/images/ directory.
    """
    # Find the project root (where tests/ directory is)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    golden_dir = project_root / "tests" / "golden" / "images"
    return golden_dir


def save_golden_image(
    image: np.ndarray,
    name: str,
    category: str,
    as_npy: bool = False,
) -> Path:
    """Save an image as a golden reference file.

    Args:
        image: Image to save (HxW or HxWxC).
        name: Filename (e.g., "calibration_markers.png").
        category: Subdirectory category (e.g., "calibration", "registration").
        as_npy: If True, save as .npy for exact dtype preservation.
            Otherwise save as PNG (lossy for uint16/float).

    Returns:
        Path where image was saved.

    Example:
        >>> image = load_image("test.png")
        >>> path = save_golden_image(image, "test_output.png", "calibration")
    """
    golden_dir = get_golden_image_dir()
    category_dir = golden_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    if as_npy:
        if not name.endswith(".npy"):
            name = name.replace(".png", ".npy").replace(".jpg", ".npy")
            if not name.endswith(".npy"):
                name += ".npy"
        filepath = category_dir / name
        np.save(filepath, image)
    else:
        filepath = category_dir / name
        # OpenCV expects BGR for color images
        if image.ndim == 3 and image.shape[2] == 3:
            # Assume RGB input, convert to BGR for saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), image_bgr)
        else:
            cv2.imwrite(str(filepath), image)

    return filepath


def load_golden_image(name: str, category: str) -> np.ndarray:
    """Load a golden reference image.

    Args:
        name: Filename (e.g., "calibration_markers.png").
        category: Subdirectory category (e.g., "calibration", "registration").

    Returns:
        Loaded image as numpy array.

    Raises:
        FileNotFoundError: If golden image doesn't exist.

    Example:
        >>> golden = load_golden_image("test_output.png", "calibration")
    """
    golden_dir = get_golden_image_dir()
    filepath = golden_dir / category / name

    if not filepath.exists():
        raise FileNotFoundError(
            f"Golden image not found: {filepath}\n"
            f"Run with REGENERATE_GOLDEN=1 to generate golden images."
        )

    if filepath.suffix == ".npy":
        return np.load(filepath)
    else:
        image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")

        # Convert BGR to RGB for color images
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


def generate_diff_image(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Generate a visual difference image highlighting changes.

    The diff image shows:
    - Identical pixels in grayscale
    - Different pixels highlighted in red

    Args:
        image1: First image.
        image2: Second image (must have same shape).

    Returns:
        RGB diff image with differences highlighted in red.

    Raises:
        ValueError: If image shapes don't match.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")

    # Convert to uint8 for visualization
    if image1.dtype != np.uint8:
        img1 = (255 * (image1 - image1.min()) / (image1.max() - image1.min())).astype(np.uint8)
    else:
        img1 = image1.copy()

    if image2.dtype != np.uint8:
        img2 = (255 * (image2 - image2.min()) / (image2.max() - image2.min())).astype(np.uint8)
    else:
        img2 = image2.copy()

    # Convert to grayscale if needed
    if img1.ndim == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1

    if img2.ndim == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2

    # Find differences
    diff_mask = np.abs(img1_gray.astype(int) - img2_gray.astype(int)) > 5

    # Create RGB diff image
    diff_image = np.stack([img1_gray, img1_gray, img1_gray], axis=-1)

    # Highlight differences in red
    diff_image[diff_mask, 0] = 255  # Red channel
    diff_image[diff_mask, 1] = 0  # Green channel
    diff_image[diff_mask, 2] = 0  # Blue channel

    return diff_image


def assert_images_equal(
    actual: np.ndarray,
    golden_name: str,
    category: str,
    ssim_threshold: float | None = DEFAULT_SSIM_THRESHOLD,
    psnr_threshold: float | None = DEFAULT_PSNR_THRESHOLD,
    mse_threshold: float | None = DEFAULT_MSE_THRESHOLD,
    save_diff: bool = True,
    diff_dir: Path | None = None,
) -> None:
    """Assert that an image matches a golden reference.

    This is the main assertion function for visual regression testing.
    If the assertion fails, it saves a diff image for debugging.

    Args:
        actual: The image to compare.
        golden_name: Name of the golden reference file.
        category: Category subdirectory (e.g., "calibration").
        ssim_threshold: Minimum SSIM value (0-1). None to skip.
        psnr_threshold: Minimum PSNR in dB. None to skip.
        mse_threshold: Maximum MSE value. None to skip.
        save_diff: Whether to save diff image on failure.
        diff_dir: Directory for diff images. If None, uses tests/test_output/visual_regression/

    Raises:
        AssertionError: If images don't match within thresholds.
        FileNotFoundError: If golden image doesn't exist.

    Example:
        >>> result = run_calibration()
        >>> assert_images_equal(
        ...     result.image,
        ...     "calibration_output.png",
        ...     "calibration",
        ...     ssim_threshold=0.95,
        ... )
    """
    # Check if we should regenerate golden images
    if os.environ.get("REGENERATE_GOLDEN", "0") == "1":
        save_golden_image(actual, golden_name, category)
        raise AssertionError(
            f"Regenerated golden image: {category}/{golden_name}\n"
            f"Rerun tests without REGENERATE_GOLDEN=1 to validate."
        )

    # Load golden image
    golden = load_golden_image(golden_name, category)

    # Compare images
    result = compare_images(actual, golden, metrics=["ssim", "psnr", "mse"])

    # Check if images pass thresholds
    passes = result.passes_threshold(
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
        mse_threshold=mse_threshold,
    )

    if not passes:
        # Generate error message
        error_msg = f"Images do not match golden '{golden_name}'\n\n"
        error_msg += result.format_report(
            ssim_threshold=ssim_threshold,
            psnr_threshold=psnr_threshold,
            mse_threshold=mse_threshold,
        )

        # Save diff image
        if save_diff and result.shape_match:
            if diff_dir is None:
                project_root = Path(__file__).resolve().parent.parent.parent
                diff_dir = project_root / "tests" / "test_output" / "visual_regression"

            diff_dir.mkdir(parents=True, exist_ok=True)

            diff_name = golden_name.replace(".png", "_diff.png").replace(".npy", "_diff.png")
            diff_path = diff_dir / category / diff_name
            diff_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                diff_image = generate_diff_image(golden, actual)
                cv2.imwrite(str(diff_path), cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR))
                error_msg += f"\n\nDifference image saved to:\n  {diff_path}"
            except Exception as e:
                error_msg += f"\n\nFailed to generate diff image: {e}"

        error_msg += (
            "\n\nTo regenerate golden images, run:\n"
            "  REGENERATE_GOLDEN=1 pytest tests/test_visual_regression.py"
        )

        raise AssertionError(error_msg)

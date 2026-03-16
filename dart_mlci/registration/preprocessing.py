"""Image preprocessing utilities for registration.

This module provides preprocessing functions to enhance registration quality
on low-contrast microscopy images.
"""

import cv2
import numpy as np


def preprocess_for_registration(
    image: np.ndarray,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess image for registration.

    Applies contrast enhancement and normalization to improve registration
    quality on low-contrast microscopy images.

    Args:
        image: Input image (grayscale or RGB)
        use_clahe: Whether to apply CLAHE contrast enhancement
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_grid_size: CLAHE tile grid size
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Preprocessed image as float32 in [0, 1] range
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Ensure uint8 for CLAHE
    if gray.dtype != np.uint8:
        # Normalize to uint8 range
        gray_min, gray_max = gray.min(), gray.max()
        if gray_max > gray_min:
            gray = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    # Apply CLAHE for contrast enhancement
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray

    # Normalize to [0, 1] float32
    if normalize:
        result = enhanced.astype(np.float32) / 255.0
    else:
        result = enhanced.astype(np.float32)

    return result


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter to reduce noise while preserving edges.

    Args:
        image: Input image (float32 or uint8)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image (same dtype as input)
    """
    # Bilateral filter works on uint8
    if image.dtype == np.float32:
        image_u8 = (image * 255).astype(np.uint8)
        filtered_u8 = cv2.bilateralFilter(image_u8, d, sigma_color, sigma_space)
        return filtered_u8.astype(np.float32) / 255.0
    else:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE contrast enhancement.

    Args:
        image: Input grayscale image (uint8)
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size

    Returns:
        Contrast-enhanced image (uint8)
    """
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8 for CLAHE")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def create_hanning_window(shape: tuple[int, int]) -> np.ndarray:
    """Create 2D Hanning window for reducing edge effects.

    Args:
        shape: (height, width) of window

    Returns:
        2D Hanning window as float32 in [0, 1] range
    """
    height, width = shape

    # Create 1D Hanning windows
    hann_y = np.hanning(height).reshape(-1, 1)
    hann_x = np.hanning(width).reshape(1, -1)

    # Outer product for 2D window
    window = hann_y * hann_x

    return window.astype(np.float32)


def normalize_to_range(
    image: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Normalize image to specified range.

    Args:
        image: Input image
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized image as float32
    """
    img_min, img_max = image.min(), image.max()

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
        normalized = normalized * (target_max - target_min) + target_min
    else:
        # Constant image
        normalized = np.full_like(image, target_min, dtype=np.float32)

    return normalized.astype(np.float32)


def compute_image_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude for edge-based preprocessing.

    Args:
        image: Input grayscale image (float32)

    Returns:
        Gradient magnitude image (float32)
    """
    # Sobel gradients
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude

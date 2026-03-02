"""Utilities for creating synthetic marker images for testing registration."""

import cv2
import numpy as np
from scipy.ndimage import shift as scipy_shift


def create_synthetic_marker_image(
    width: int,
    height: int,
    marker_positions: dict[str, tuple[int, int]],
    background: str = "uniform",
    marker_size: int = 20,
) -> np.ndarray:
    """Create synthetic image with cross/circle markers at specified positions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        marker_positions: Dict with 'cross' and/or 'circle' keys mapping to (x, y) positions
        background: Background type - 'uniform', 'gradient', 'noise', or 'cells'
        marker_size: Size of markers in pixels

    Returns:
        HWC numpy array (uint8) with shape (height, width, 3)
    """
    # Create background
    if background == "uniform":
        image = np.full((height, width, 3), 128, dtype=np.uint8)
    elif background == "gradient":
        # Linear gradient from dark to bright
        grad = np.linspace(50, 200, width, dtype=np.uint8)
        image = np.tile(grad, (height, 1))
        image = np.stack([image] * 3, axis=-1)
    elif background == "noise":
        # Random noise
        image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    elif background == "cells":
        # Simulate cell-like structures
        image = np.full((height, width, 3), 100, dtype=np.uint8)
        # Add random blobs
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(10, 30)
            cv2.circle(image, (x, y), radius, (150, 150, 150), -1)
    else:
        raise ValueError(f"Unknown background type: {background}")

    # Draw markers
    for marker_type, (x, y) in marker_positions.items():
        if marker_type == "cross":
            # Draw cross (bright white)
            cv2.line(image, (x - marker_size, y), (x + marker_size, y), (255, 255, 255), 3)
            cv2.line(image, (x, y - marker_size), (x, y + marker_size), (255, 255, 255), 3)
        elif marker_type == "circle":
            # Draw circle (bright white)
            cv2.circle(image, (x, y), marker_size, (255, 255, 255), 3)

    return image


def apply_known_translation(
    image: np.ndarray,
    dx: float,
    dy: float,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Shift image by known amount using scipy.

    Args:
        image: Input image (HW or HWC)
        dx: Translation in x direction (pixels)
        dy: Translation in y direction (pixels)
        interpolation: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Shifted image with same shape as input
    """
    # Map interpolation string to scipy order
    order_map = {
        "nearest": 0,
        "bilinear": 1,
        "linear": 1,
        "cubic": 3,
    }
    order = order_map.get(interpolation, 1)

    # scipy shift uses (row, col) = (y, x) order
    if image.ndim == 2:
        # Grayscale
        shifted = scipy_shift(image, shift=(dy, dx), order=order, mode="constant", cval=0)
    elif image.ndim == 3:
        # RGB - shift each channel
        shifted = np.zeros_like(image)
        for c in range(image.shape[2]):
            shifted[:, :, c] = scipy_shift(
                image[:, :, c], shift=(dy, dx), order=order, mode="constant", cval=0
            )
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return shifted.astype(image.dtype)


def create_marker_group_pixel(
    marker_positions: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Convert marker positions to marker_group_pixel dict format.

    Args:
        marker_positions: Dict with 'cross' and/or 'circle' keys mapping to (x, y) tuples

    Returns:
        Dict with 'cross' and/or 'circle' keys mapping to numpy arrays
    """
    marker_group = {}
    for key, (x, y) in marker_positions.items():
        marker_group[key] = np.array([x, y])
    return marker_group


def create_synthetic_timelapse(
    num_frames: int,
    width: int,
    height: int,
    marker_positions: dict[str, tuple[int, int]],
    drift_per_frame: tuple[float, float] = (1.0, 0.5),
    background: str = "cells",
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Create synthetic time-lapse sequence with known cumulative drift.

    Args:
        num_frames: Number of frames to generate
        width: Image width
        height: Image height
        marker_positions: Initial marker positions
        drift_per_frame: (dx, dy) drift added per frame
        background: Background type

    Returns:
        Tuple of (frames, cumulative_translations):
            - frames: List of HWC numpy arrays
            - cumulative_translations: List of (dx, dy) tuples for each frame relative to frame 0
    """
    # Create reference frame
    reference = create_synthetic_marker_image(width, height, marker_positions, background)

    frames = [reference]
    translations = [(0.0, 0.0)]

    # Generate drifted frames
    for i in range(1, num_frames):
        cumulative_dx = drift_per_frame[0] * i
        cumulative_dy = drift_per_frame[1] * i

        # Apply translation to reference
        drifted = apply_known_translation(reference, cumulative_dx, cumulative_dy)
        frames.append(drifted)
        translations.append((cumulative_dx, cumulative_dy))

    return frames, translations

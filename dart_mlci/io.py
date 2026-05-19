"""IO functionality for DMC masking.

This module provides file I/O utilities for the dart-mlci package,
including image loading from various formats and structure configuration loading.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .utils import normalize_image


def load_roi_structures(path):
    """Load RoI structure definitions from JSON file.

    Args:
        path: Path to the JSON file containing structure definitions

    Returns:
        Dictionary of structure definitions
    """
    with open(path, encoding="utf-8") as input:
        roi_structures = json.load(input)

    return roi_structures


def load_image(image_path: Path | str) -> np.ndarray:
    """Load and prepare image for the masking pipeline.

    Handles single images as well as TIFF stacks (TxCxHxW format).
    For stacks, extracts the first frame and first channel.
    Supports TIFF, PNG, JPEG, and other common image formats.

    Args:
        image_path: Path to the image file

    Returns:
        Image as HxWx3 numpy array in uint8 format

    Raises:
        ValueError: If the image cannot be loaded or is invalid
        FileNotFoundError: If the image file does not exist

    Example:
        >>> image = load_image("calibration_image.tif")
        >>> print(image.shape)  # (height, width, 3)
        >>> print(image.dtype)  # uint8
    """
    import tifffile

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        # Use tifffile directly to handle multi-dimensional TIFFs
        image = tifffile.imread(str(image_path))

        # Handle multi-dimensional TIFF stacks (TxCxHxW or CxHxW)
        if image.ndim == 4:
            # TxCxHxW format - take first time point and first channel
            image = image[0, 0]
        elif image.ndim == 3:
            # Could be CxHxW, TxHxW, or HxWxC
            if image.shape[0] <= 4:
                # Likely CxHxW - take first channel
                image = image[0]
            elif image.shape[2] <= 4:
                # Likely HxWxC - keep as is
                pass
            else:
                # Likely TxHxW - take first time point
                image = image[0]

        # Normalize to uint8
        if image.dtype != np.uint8:
            image = normalize_image(image)
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

    # Normalize if uint16 (shouldn't happen after above, but safety check)
    if image.dtype == np.uint16:
        image = normalize_image(image)

    # Convert grayscale to RGB
    if len(image.shape) == 2:
        from skimage.color import gray2rgb

        image = gray2rgb(image)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        from skimage.color import gray2rgb

        image = gray2rgb(image[:, :, 0])

    return image


def save_image(
    image: np.ndarray,
    output_path: Path | str,
    mask: np.ndarray | None = None,
) -> Path | None:
    """Save an image (and optional binary mask) to disk.

    TIFF outputs use ``tifffile``; other suffixes go through OpenCV (3-channel
    arrays are RGB→BGR converted). ``CxHxW`` arrays with up to 4 channels are
    transposed to ``HxWxC`` automatically. When ``mask`` is provided it is
    written next to ``output_path`` as ``<stem>_mask<suffix>``.

    Args:
        image: Image array (HxWxC, HxW, or CxHxW with channels ≤ 4).
        output_path: Destination path; suffix selects the writer.
        mask: Optional binary mask (HxW). Saved as uint8 0/255.

    Returns:
        Path to the saved mask, or None if no mask was supplied.
    """
    import tifffile

    output_path = Path(output_path)

    if image.ndim == 3 and image.shape[0] <= 4:
        image_hwc = np.moveaxis(image, 0, -1)
    else:
        image_hwc = image

    suffix = output_path.suffix.lower()
    is_tiff = suffix in {".tif", ".tiff"}

    if is_tiff:
        tifffile.imwrite(str(output_path), image_hwc)
    else:
        if image_hwc.ndim == 3 and image_hwc.shape[2] == 3:
            image_hwc = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_hwc)

    if mask is None:
        return None

    mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
    mask_uint8 = mask.astype(np.uint8) * 255
    if is_tiff:
        tifffile.imwrite(str(mask_path), mask_uint8)
    else:
        cv2.imwrite(str(mask_path), mask_uint8)
    return mask_path

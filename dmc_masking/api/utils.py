"""Utility functions for the DMC Masking API."""

import base64
import io

import numpy as np
import tifffile
from PIL import Image


def base64_to_array(b64_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy array (HxWx3 uint8 format).

    Features:
    - Auto-strips data URI prefix (data:image/png;base64,...)
    - Supports PNG, JPEG, TIFF formats
    - Returns same format as load_image() for consistency
    - Comprehensive error handling

    Args:
        b64_string: Base64-encoded image string, optionally with data URI prefix

    Returns:
        HxWx3 numpy array in uint8 format

    Raises:
        ValueError: For invalid base64 or unsupported formats
    """
    # Strip data URI prefix if present
    if b64_string.startswith("data:"):
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        else:
            raise ValueError("Invalid data URI format: missing comma separator")

    # Decode base64 to bytes
    try:
        img_bytes = base64.b64decode(b64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}") from e

    if len(img_bytes) == 0:
        raise ValueError("Decoded image is empty")

    # Try to load image with PIL first (handles PNG, JPEG)
    try:
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
    except Exception:
        # Try tifffile for TIFF format
        try:
            arr = tifffile.imread(io.BytesIO(img_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}") from e

    # Convert to HxWx3 uint8 format (same as load_image)
    if arr.ndim == 2:
        # Grayscale: convert to RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        # Handle channel-first format (CxHxW) -> (HxWxC)
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            arr = np.transpose(arr, (1, 2, 0))

        # Handle single channel (HxWx1) -> (HxWx3)
        if arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        # Handle RGBA (HxWx4) -> (HxWx3)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # Ensure uint8 dtype
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            # Normalized float -> uint8
            arr = (arr * 255).astype(np.uint8)
        else:
            # Clip to valid range
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Final validation
    if arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got shape {arr.shape}")

    return arr


def array_to_base64_png(arr: np.ndarray, is_mask: bool = False) -> str:
    """
    Convert numpy array to base64-encoded PNG string.

    Handles shape transformations and normalization for consistent output.

    Args:
        arr: Numpy array to encode (any shape, any dtype)
        is_mask: If True, treats array as binary mask (threshold > 0)

    Returns:
        Base64-encoded PNG string (without data URI prefix)
    """
    # Handle different array shapes
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # (C, H, W) -> (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # Normalize to uint8 if needed
    if arr.dtype != np.uint8:
        if is_mask:
            arr = (arr > 0).astype(np.uint8) * 255
        else:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

    # Encode as PNG
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode="L")
    elif arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

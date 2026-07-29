"""Shared rendering functions for cell visualization.

These functions depend on the optional ``acia`` package. If acia is not
installed, importing this module will raise ImportError.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import pint
    from acia.segm.local import THWCSequenceSource
    from acia.viz import colorize_instance_mask, render_scalebar

    _UNIT_REGISTRY = pint.UnitRegistry()
    _ACIA_AVAILABLE = True
except ImportError:
    _ACIA_AVAILABLE = False


def add_scalebar(
    image: np.ndarray,
    pixel_size: float,
    bar_um: float = 10,
) -> np.ndarray:
    """Add scalebar to image using acia's render_scalebar.

    Args:
        image: HxWxC RGB image (uint8).
        pixel_size: Pixel size in micrometers.
        bar_um: Scalebar width in micrometers.

    Returns:
        Image with scalebar added.

    Raises:
        ImportError: If acia is not installed.
    """
    if not _ACIA_AVAILABLE:
        raise ImportError("acia library is required for add_scalebar")

    source = THWCSequenceSource(image[None, :, :, :].astype(np.uint8))
    result = render_scalebar(
        image_source=source,
        xy_position=(0.80, 0.95),
        size_of_pixel=pixel_size * _UNIT_REGISTRY.micrometer,
        bar_width=bar_um * _UNIT_REGISTRY.micrometer,
        bar_height=2 * _UNIT_REGISTRY.micrometer,
        color=(255, 255, 255),
        font_size=20,
        show_text=True,
    )
    return result.image_stack[0]


def render_cell_visualization(
    cropped_image: np.ndarray,
    labeled_mask: np.ndarray,
    chamber_mask: np.ndarray,
    pixel_size: float,
    alpha: float = 0.5,
    scalebar: bool = True,
    scalebar_um: float = 10,
) -> np.ndarray:
    """Render cells with colored masks, chamber overlay, and optional scalebar.

    Args:
        cropped_image: HxWxC RGB image (uint8).
        labeled_mask: HxW instance mask (0=background, 1..N=cells).
        chamber_mask: HxW binary mask (True=outside ROI).
        pixel_size: Pixel size in micrometers.
        alpha: Cell mask transparency (0-1).
        scalebar: Whether to add a scalebar.
        scalebar_um: Scalebar width in micrometers.

    Returns:
        Rendered visualization image (HxWxC, uint8, RGB).

    Raises:
        ImportError: If acia is not installed.
    """
    if not _ACIA_AVAILABLE:
        raise ImportError("acia library is required for render_cell_visualization")

    colored_cells = colorize_instance_mask(labeled_mask, seed=42)

    output = cropped_image.copy().astype(np.float32)
    cell_area = labeled_mask > 0
    output[cell_area] = (
        alpha * colored_cells[cell_area].astype(np.float32) + (1 - alpha) * output[cell_area]
    )

    output[chamber_mask] = 0.3 * output[chamber_mask] + 0.7 * np.array(
        [128, 128, 128], dtype=np.float32
    )

    output = output.astype(np.uint8)

    if scalebar:
        output = add_scalebar(output, pixel_size, scalebar_um)

    return output


def render_cell_visualization_full_frame(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    chamber_mask: np.ndarray,
    pixel_size: float,
    alpha: float = 0.5,
    dim_outside: bool = True,
    scalebar: bool = True,
    scalebar_um: float = 10,
) -> np.ndarray:
    """Render cells on a full-frame image with optional chamber dimming or outline.

    Args:
        image: HxWxC RGB image (uint8).
        labeled_mask: HxW instance mask (0=background, 1..N=cells).
        chamber_mask: HxW binary mask (True=outside ROI).
        pixel_size: Pixel size in micrometers.
        alpha: Cell mask transparency (0-1).
        dim_outside: If True, dim the outside-chamber region (Video B / filtered mode).
            If False, draw the chamber boundary as a white outline instead (Video A /
            unfiltered mode — shows all cells including artifacts).
        scalebar: Whether to add a scalebar.
        scalebar_um: Scalebar width in micrometers.

    Returns:
        Rendered visualization image (HxWxC, uint8, RGB).

    Raises:
        ImportError: If acia is not installed.
    """
    if not _ACIA_AVAILABLE:
        raise ImportError("acia library is required for render_cell_visualization_full_frame")

    colored_cells = colorize_instance_mask(labeled_mask, seed=42)

    output = image.copy().astype(np.float32)
    cell_area = labeled_mask > 0
    output[cell_area] = (
        alpha * colored_cells[cell_area].astype(np.float32) + (1 - alpha) * output[cell_area]
    )

    if dim_outside:
        output[chamber_mask] = 0.3 * output[chamber_mask] + 0.7 * np.array(
            [128, 128, 128], dtype=np.float32
        )
        output = output.astype(np.uint8)
    else:
        output = output.astype(np.uint8)
        roi_uint8 = (~chamber_mask).astype(np.uint8)
        contours, _ = cv2.findContours(roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 255, 255), 2)

    if scalebar:
        output = add_scalebar(output, pixel_size, scalebar_um)

    return output


def backproject_mask_to_original(
    labeled_mask: np.ndarray,
    rotation_angle: float,
    crop_bbox: tuple,
    original_hw: tuple,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Back-project a mask from cropped+rotated space to original image space.

    Reverses the pipeline transforms (uncrop → undo registration translation → un-rotate)
    to place masks back in the coordinate frame of the original raw image.

    Args:
        labeled_mask: HxW instance mask in cropped+rotated coordinate space.
        rotation_angle: Rotation angle (degrees) that was applied to the original image.
        crop_bbox: (minx, miny, maxx, maxy) crop bbox in the (post-registration) rotated space.
        original_hw: (H, W) of the original (pre-rotation) image.
        dx: Registration translation X (applied after rotation, before cropping).
        dy: Registration translation Y (applied after rotation, before cropping).

    Returns:
        HxW mask in original image coordinate space.
    """
    orig_h, orig_w = original_hw
    # Only the crop origin is needed; the extent comes from the mask's own shape.
    minx, miny = (
        int(crop_bbox[0]),
        int(crop_bbox[1]),
    )

    # Compute rotated image dimensions (same formula used during forward rotation)
    image_center = (orig_w / 2, orig_h / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    bound_w = int(orig_h * abs_sin + orig_w * abs_cos)
    bound_h = int(orig_h * abs_cos + orig_w * abs_sin)
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # Step 1: Uncrop — place the mask back in the full rotated+translated canvas
    canvas = np.zeros((bound_h, bound_w), dtype=labeled_mask.dtype)
    crop_h, crop_w = labeled_mask.shape
    dst_y2 = min(miny + crop_h, bound_h)
    dst_x2 = min(minx + crop_w, bound_w)
    src_h = dst_y2 - miny
    src_w = dst_x2 - minx
    if src_h > 0 and src_w > 0:
        canvas[miny:dst_y2, minx:dst_x2] = labeled_mask[:src_h, :src_w]

    # Step 2: Undo registration translation (shift canvas by +dx, +dy)
    if dx != 0.0 or dy != 0.0:
        trans_mat = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, trans_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)

    # Step 3: Un-rotate using the inverse of the forward rotation matrix
    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    backprojected = cv2.warpAffine(canvas, inv_rot_mat, (orig_w, orig_h), flags=cv2.INTER_NEAREST)

    return backprojected

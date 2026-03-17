"""Shared rendering functions for cell visualization.

These functions depend on the optional ``acia`` package. If acia is not
installed, importing this module will raise ImportError.
"""

from __future__ import annotations

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

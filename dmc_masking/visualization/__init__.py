"""Visualization utilities for marker detection and ROI masking.

Submodules:
  - plotting: Matplotlib static plots
  - drawing: OpenCV drawing on video frames
  - video: Video generation utilities
"""

from dmc_masking.visualization.drawing import (
    add_step_title,
    draw_markers_cv,
    draw_matched_pairs_cv,
    draw_roi_polygon,
    prepare_frame,
)
from dmc_masking.visualization.plotting import (
    plot_marker_pairs,
    plot_marker_paris,
    plot_markers,
    plot_markers_on_image,
    render_markers_to_frame,
)
from dmc_masking.visualization.video import (
    COLOR_CIRCLE,
    COLOR_CROSS,
    COLOR_MATCHED_LINE,
    COLOR_PROGRESS_BG,
    COLOR_PROGRESS_FILL,
    COLOR_ROI_POLYGON,
    COLOR_SELECTED,
    COLOR_TEXT,
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    animate_zoom_to_roi,
    draw_progress_bar,
    rotate_image_no_crop,
    write_frames,
)

__all__ = [
    "COLOR_CIRCLE",
    "COLOR_CROSS",
    "COLOR_MATCHED_LINE",
    "COLOR_PROGRESS_BG",
    "COLOR_PROGRESS_FILL",
    "COLOR_ROI_POLYGON",
    "COLOR_SELECTED",
    "COLOR_TEXT",
    "FPS",
    "FRAME_HEIGHT",
    # Constants
    "FRAME_WIDTH",
    "add_step_title",
    "animate_zoom_to_roi",
    "draw_markers_cv",
    "draw_matched_pairs_cv",
    "draw_progress_bar",
    "draw_roi_polygon",
    "plot_marker_pairs",
    "plot_marker_paris",
    "plot_markers",
    # Plotting (matplotlib)
    "plot_markers_on_image",
    # Drawing (OpenCV)
    "prepare_frame",
    "render_markers_to_frame",
    "rotate_image_no_crop",
    # Video utilities
    "write_frames",
]

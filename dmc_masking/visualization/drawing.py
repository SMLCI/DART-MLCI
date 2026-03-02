"""OpenCV-based drawing on video frames."""

import cv2
import numpy as np

from dmc_masking.mask import RoIPolygon
from dmc_masking.visualization.video import (
    COLOR_CIRCLE,
    COLOR_CROSS,
    COLOR_MATCHED_LINE,
    COLOR_ROI_POLYGON,
    COLOR_SELECTED,
    COLOR_TEXT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)


def prepare_frame(
    image: np.ndarray, frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT)
) -> tuple[np.ndarray, float, tuple]:
    """Scale and pad image to fit the target frame size.

    Args:
        image: Input image in HxWxC format (BGR)
        frame_size: Target (width, height)

    Returns:
        Tuple of (frame, scale, offset)
    """
    target_w, target_h = frame_size
    progress_bar_height = 80
    usable_height = target_h - progress_bar_height

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(target_w / w, usable_height / h) * 0.95

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (usable_height - new_h) // 2

    frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return frame, scale, (x_offset, y_offset)


def add_step_title(frame: np.ndarray, title: str) -> np.ndarray:
    """Add a centered title at the top of the frame."""
    _h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 42

    cv2.putText(
        frame, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2, cv2.LINE_AA
    )

    return frame


def draw_markers_cv(
    frame: np.ndarray,
    markers: list,
    scale: float,
    offset: tuple,
    highlight_indices: list | None = None,
) -> np.ndarray:
    """Draw marker annotations on the frame using OpenCV."""
    if highlight_indices is None:
        highlight_indices = []

    for i, marker in enumerate(markers):
        center = marker["bbox_center"]
        label = marker["label"]
        conf = marker.get("conf", 0.0)

        x = int(center[0] * scale + offset[0])
        y = int(center[1] * scale + offset[1])

        if i in highlight_indices:
            color = COLOR_SELECTED
            thickness = 3
        elif label == "cross":
            color = COLOR_CROSS
            thickness = 2
        else:
            color = COLOR_CIRCLE
            thickness = 2

        marker_size = int(15 * scale) if scale > 0.5 else 15
        if label == "cross":
            cv2.line(
                frame,
                (x - marker_size, y - marker_size),
                (x + marker_size, y + marker_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (x - marker_size, y + marker_size),
                (x + marker_size, y - marker_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(frame, (x, y), marker_size, color, thickness, cv2.LINE_AA)

        label_text = f"{label} ({conf:.2f})"
        cv2.putText(
            frame,
            label_text,
            (x + marker_size + 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_matched_pairs_cv(
    frame: np.ndarray,
    markers: list,
    matched_indices: list,
    scale: float,
    offset: tuple,
    selected_pair_idx: int | None = None,
) -> np.ndarray:
    """Draw lines connecting matched marker pairs using OpenCV."""
    for i, (cross_idx, circle_idx) in enumerate(matched_indices):
        cross_center = markers[cross_idx]["bbox_center"]
        circle_center = markers[circle_idx]["bbox_center"]

        x1 = int(cross_center[0] * scale + offset[0])
        y1 = int(cross_center[1] * scale + offset[1])
        x2 = int(circle_center[0] * scale + offset[0])
        y2 = int(circle_center[1] * scale + offset[1])

        if selected_pair_idx is not None and i == selected_pair_idx:
            color = COLOR_SELECTED
            thickness = 4
        else:
            color = COLOR_MATCHED_LINE
            thickness = 2

        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    return frame


def draw_roi_polygon(
    frame: np.ndarray,
    polygon: RoIPolygon,
    scale: float,
    offset: tuple,
    alpha: float = 0.3,
    inverted: bool = False,
) -> np.ndarray:
    """Draw ROI polygon overlay on the frame."""
    coords = np.array(polygon.roi_polygon.exterior.coords)

    coords_transformed = coords * scale
    coords_transformed[:, 0] += offset[0]
    coords_transformed[:, 1] += offset[1]
    coords_int = coords_transformed.astype(np.int32)

    if inverted:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [coords_int], 255)
        overlay = frame.copy()
        overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
        frame[:] = overlay
    else:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [coords_int], COLOR_ROI_POLYGON)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.polylines(frame, [coords_int], True, COLOR_ROI_POLYGON, 2, cv2.LINE_AA)

    return frame

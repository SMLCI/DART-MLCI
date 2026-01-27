"""Visualization utilities for marker detection and ROI masking."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dmc_masking.mask import RoIPolygon

# Video settings
FRAME_WIDTH = 1920 // 2
FRAME_HEIGHT = 1080
FPS = 30

# Colors in BGR format for OpenCV
COLOR_CROSS = (0, 0, 255)  # Red
COLOR_CIRCLE = (255, 0, 0)  # Blue
COLOR_MATCHED_LINE = (0, 255, 0)  # Green
COLOR_SELECTED = (0, 215, 255)  # Gold/Yellow
COLOR_ROI_POLYGON = (255, 255, 0)  # Cyan
COLOR_PROGRESS_BG = (40, 40, 40)  # Dark gray
COLOR_PROGRESS_FILL = (0, 200, 0)  # Green
COLOR_TEXT = (255, 255, 255)  # White


def rotate_image_no_crop(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center without cropping.

    Expands the canvas with black borders to fit the entire rotated image.

    Args:
        image: Input image in HxWxC or HxW format
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image with expanded canvas (black borders where needed)
    """
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]

    center_x, center_y = width / 2, height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    rot_mat[0, 2] += (new_width / 2) - center_x
    rot_mat[1, 2] += (new_height / 2) - center_y

    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated


# ============== Matplotlib-based visualization (for static images) ==============


def plot_markers_on_image(
    image: np.ndarray,
    markers: list,
    matched_indices: list,
    title: str = "",
    output_path: Path | None = None,
    highlight_indices: list | None = None,
    selected_pair_idx: int | None = None,
) -> None:
    """Plot detected markers on an image and optionally save to file.

    Args:
        image: Input image in HxWxC format (BGR from cv2.imread)
        markers: List of detected markers with 'bbox_center' and 'label' keys
        matched_indices: List of matched marker index pairs (cross_idx, circle_idx)
        title: Title for the plot
        output_path: Optional path to save the figure
        highlight_indices: Optional list of marker indices to highlight in gold
        selected_pair_idx: Optional index of the selected pair to highlight
    """
    if highlight_indices is None:
        highlight_indices = []

    if image.ndim == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    plt.figure(figsize=(12, 10))
    plt.imshow(image_rgb)

    colors = {"cross": "red", "circle": "blue"}
    marker_symbols = {"cross": "x", "circle": "o"}

    for i, marker in enumerate(markers):
        center = marker["bbox_center"]
        label = marker["label"]
        conf = marker.get("conf", 0.0)

        if i in highlight_indices:
            color = "gold"
            size = 300
        else:
            color = colors.get(label, "green")
            size = 200

        symbol = marker_symbols.get(label, "s")

        plt.scatter(center[0], center[1], c=color, marker=symbol, s=size, linewidths=3, zorder=5)
        plt.annotate(
            f"{i}: {label} ({conf:.2f})",
            (center[0], center[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    for i, (cross_idx, circle_idx) in enumerate(matched_indices):
        cross_center = markers[cross_idx]["bbox_center"]
        circle_center = markers[circle_idx]["bbox_center"]

        if selected_pair_idx is not None and i == selected_pair_idx:
            line_color = "gold"
            line_width = 4
        else:
            line_color = "g"
            line_width = 2

        plt.plot(
            [cross_center[0], circle_center[0]],
            [cross_center[1], circle_center[1]],
            line_color,
            linewidth=line_width,
            alpha=0.7,
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_markers(image: np.ndarray, markers: list, output_path: Path | None = None) -> None:
    """Simple marker visualization (backward-compatible wrapper).

    Args:
        image: Input image
        markers: List of detected markers
        output_path: Optional path to save the figure
    """
    plot_markers_on_image(image, markers, [], output_path=output_path)


def plot_marker_pairs(
    image: np.ndarray, matched_marker_indices: list, markers: list, output_path: Path | None = None
) -> None:
    """Visualize marker pairs (backward-compatible wrapper).

    Args:
        image: Input image
        matched_marker_indices: List of matched marker index pairs
        markers: List of detected markers
        output_path: Optional path to save the figure
    """
    plot_markers_on_image(image, markers, matched_marker_indices, output_path=output_path)


# Alias for backward compatibility with typo
plot_marker_paris = plot_marker_pairs


def render_markers_to_frame(
    image: np.ndarray,
    markers: list,
    matched_indices: list,
    frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
    highlight_indices: list | None = None,
    selected_pair_idx: int | None = None,
    faded_indices: list | None = None,
) -> np.ndarray:
    """Render markers on image using matplotlib and return as video frame.

    Args:
        image: Input image in HxWxC format (BGR from cv2.imread)
        markers: List of detected markers
        matched_indices: List of matched marker index pairs
        frame_size: Target frame size (width, height)
        highlight_indices: Optional list of marker indices to highlight
        selected_pair_idx: Optional index of selected pair to highlight
        faded_indices: Optional list of marker indices to render with low opacity

    Returns:
        Frame as numpy array suitable for video
    """
    if highlight_indices is None:
        highlight_indices = []
    if faded_indices is None:
        faded_indices = []

    if image.ndim == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    target_w, target_h = frame_size
    # Account for progress bar
    usable_height = target_h - 80

    # Calculate figure size to match frame
    dpi = 100
    fig_w = target_w / dpi
    fig_h = usable_height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(image_rgb)

    colors = {"cross": "red", "circle": "blue"}
    marker_symbols = {"cross": "x", "circle": "o"}

    for i, marker in enumerate(markers):
        center = marker["bbox_center"]
        label = marker["label"]
        conf = marker.get("conf", 0.0)

        is_faded = i in faded_indices
        alpha = 0.3 if is_faded else 1.0

        if i in highlight_indices:
            color = "gold"
            size = 300
        else:
            color = colors.get(label, "green")
            size = 200

        symbol = marker_symbols.get(label, "s")
        ax.scatter(
            center[0],
            center[1],
            c=color,
            marker=symbol,
            s=size,
            linewidths=3,
            zorder=5,
            alpha=alpha,
        )
        ax.annotate(
            f"{i}: {label} ({conf:.2f})",
            (center[0], center[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            alpha=alpha,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7 * alpha},
        )

    for i, (cross_idx, circle_idx) in enumerate(matched_indices):
        cross_center = markers[cross_idx]["bbox_center"]
        circle_center = markers[circle_idx]["bbox_center"]

        if selected_pair_idx is not None and i == selected_pair_idx:
            line_color = "gold"
            line_width = 4
        else:
            line_color = "g"
            line_width = 2

        ax.plot(
            [cross_center[0], circle_center[0]],
            [cross_center[1], circle_center[1]],
            line_color,
            linewidth=line_width,
            alpha=0.7,
        )

    ax.axis("off")
    fig.tight_layout(pad=0)

    # Render to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() as tostring_rgb() was removed in newer matplotlib
    frame_rgba = np.asarray(fig.canvas.buffer_rgba())
    frame_rgb = frame_rgba[:, :, :3]  # Drop alpha channel

    plt.close(fig)

    # Convert RGB to BGR and create full frame with progress bar area
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Resize to usable area if needed
    if frame_bgr.shape[1] != target_w or frame_bgr.shape[0] != usable_height:
        frame_bgr = cv2.resize(frame_bgr, (target_w, usable_height))

    # Create full frame with black progress bar area
    full_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    full_frame[:usable_height, :, :] = frame_bgr

    return full_frame


# ============== OpenCV-based visualization (for video frames) ==============


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


def draw_progress_bar(
    frame: np.ndarray,
    current_step: int,
    total_steps: int,
    step_name: str,
    step_progress: float = 1.0,
) -> np.ndarray:
    """Draw a progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]

    bar_height = 30
    bar_margin = 50
    bar_y = h - 60
    bar_width = w - 2 * bar_margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), COLOR_PROGRESS_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(
        frame, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), (60, 60, 60), -1
    )
    cv2.rectangle(
        frame, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), (100, 100, 100), 2
    )

    overall_progress = (current_step - 1 + step_progress) / total_steps
    fill_width = int(bar_width * overall_progress)

    if fill_width > 0:
        cv2.rectangle(
            frame,
            (bar_margin, bar_y),
            (bar_margin + fill_width, bar_y + bar_height),
            COLOR_PROGRESS_FILL,
            -1,
        )

    step_text = f"Step {current_step}/{total_steps}: {step_name}"
    text_size = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = bar_y - 10

    cv2.putText(
        frame,
        step_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    return frame


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


def write_frames(writer: cv2.VideoWriter, frame: np.ndarray, num_frames: int) -> None:
    """Write the same frame multiple times to the video."""
    for _ in range(num_frames):
        writer.write(frame)


def animate_zoom_to_roi(
    full_image: np.ndarray,
    roi_bounds: tuple,
    num_frames: int,
    frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
) -> list:
    """Generate frames that smoothly zoom from full image view to ROI crop."""
    h, w = full_image.shape[:2]
    minx, miny, maxx, maxy = roi_bounds

    frames = []
    for i in range(num_frames):
        progress = i / max(num_frames - 1, 1)
        eased = 0.5 - 0.5 * np.cos(np.pi * progress)

        curr_minx = max(0, int(0 + (minx - 0) * eased))
        curr_miny = max(0, int(0 + (miny - 0) * eased))
        curr_maxx = min(w, int(w + (maxx - w) * eased))
        curr_maxy = min(h, int(h + (maxy - h) * eased))

        cropped = full_image[curr_miny:curr_maxy, curr_minx:curr_maxx]

        # Use prepare_frame to scale and center the cropped image
        frame, _, _ = prepare_frame(cropped, frame_size)
        frames.append(frame)

    return frames

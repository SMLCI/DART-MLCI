"""Matplotlib-based static visualizations for markers and ROI."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dmc_masking.visualization.video import FRAME_HEIGHT, FRAME_WIDTH


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

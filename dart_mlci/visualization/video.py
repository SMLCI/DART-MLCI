"""Video generation utilities and shared constants."""

import cv2
import numpy as np

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
    from dart_mlci.visualization.drawing import prepare_frame

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

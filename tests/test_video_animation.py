"""Test for generating video animation of the marker detection and masking pipeline."""

import unittest
from pathlib import Path

import cv2
import numpy as np

import dmc_masking
from dmc_masking import MarkerDetectionModel
from dmc_masking.mask import RoIPolygon, SingleRoIStructureLibrary, apply_mask
from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers
from dmc_masking.rotation import compute_marker_group_angles, rotate_image_and_markers

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)

# Video settings
FRAME_WIDTH = 1920
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


def prepare_frame(image: np.ndarray, frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT)) -> np.ndarray:
    """Scale and pad image to fit the target frame size while maintaining aspect ratio.

    Args:
        image: Input image in HxWxC format (BGR)
        frame_size: Target (width, height)

    Returns:
        Padded/scaled image of size frame_size
    """
    target_w, target_h = frame_size

    # Account for progress bar height at bottom
    progress_bar_height = 80
    usable_height = target_h - progress_bar_height

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    # Calculate scale to fit within usable area
    scale = min(target_w / w, usable_height / h) * 0.95  # 95% to leave margin

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create black frame
    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the image (above progress bar area)
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
    """Draw a progress bar at the bottom of the frame.

    Args:
        frame: The frame to draw on
        current_step: Current step number (1-indexed)
        total_steps: Total number of steps
        step_name: Name of the current step
        step_progress: Progress within current step (0.0 to 1.0)

    Returns:
        Frame with progress bar drawn
    """
    h, w = frame.shape[:2]

    # Progress bar dimensions
    bar_height = 30
    bar_margin = 50
    bar_y = h - 60
    bar_width = w - 2 * bar_margin

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), COLOR_PROGRESS_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw progress bar background
    cv2.rectangle(
        frame,
        (bar_margin, bar_y),
        (bar_margin + bar_width, bar_y + bar_height),
        (60, 60, 60),
        -1,
    )
    cv2.rectangle(
        frame,
        (bar_margin, bar_y),
        (bar_margin + bar_width, bar_y + bar_height),
        (100, 100, 100),
        2,
    )

    # Calculate overall progress
    overall_progress = (current_step - 1 + step_progress) / total_steps
    fill_width = int(bar_width * overall_progress)

    # Draw filled portion
    if fill_width > 0:
        cv2.rectangle(
            frame,
            (bar_margin, bar_y),
            (bar_margin + fill_width, bar_y + bar_height),
            COLOR_PROGRESS_FILL,
            -1,
        )

    # Draw step text
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
    """Add a centered title at the top of the frame.

    Args:
        frame: The frame to draw on
        title: Title text

    Returns:
        Frame with title drawn
    """
    h, w = frame.shape[:2]

    # Draw semi-transparent background for title
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw title
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 42

    cv2.putText(
        frame,
        title,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_markers(
    frame: np.ndarray,
    markers: list,
    scale: float,
    offset: tuple,
    highlight_indices: list | None = None,
) -> np.ndarray:
    """Draw marker annotations on the frame.

    Args:
        frame: The frame to draw on
        markers: List of marker dictionaries
        scale: Scale factor applied to image
        offset: (x, y) offset of image in frame
        highlight_indices: Optional list of marker indices to highlight

    Returns:
        Frame with markers drawn
    """
    if highlight_indices is None:
        highlight_indices = []

    for i, marker in enumerate(markers):
        center = marker["bbox_center"]
        label = marker["label"]
        conf = marker.get("conf", 0.0)

        # Transform coordinates to frame space
        x = int(center[0] * scale + offset[0])
        y = int(center[1] * scale + offset[1])

        # Choose color based on marker type and highlight status
        if i in highlight_indices:
            color = COLOR_SELECTED
            thickness = 3
        elif label == "cross":
            color = COLOR_CROSS
            thickness = 2
        else:
            color = COLOR_CIRCLE
            thickness = 2

        # Draw marker symbol
        marker_size = int(15 * scale) if scale > 0.5 else 15
        if label == "cross":
            # Draw X
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
            # Draw circle
            cv2.circle(frame, (x, y), marker_size, color, thickness, cv2.LINE_AA)

        # Draw label with confidence
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


def draw_matched_pairs(
    frame: np.ndarray,
    markers: list,
    matched_indices: list,
    scale: float,
    offset: tuple,
    selected_pair_idx: int | None = None,
) -> np.ndarray:
    """Draw lines connecting matched marker pairs.

    Args:
        frame: The frame to draw on
        markers: List of marker dictionaries
        matched_indices: List of (cross_idx, circle_idx) tuples
        scale: Scale factor applied to image
        offset: (x, y) offset of image in frame
        selected_pair_idx: Index of the selected pair to highlight

    Returns:
        Frame with matched pair lines drawn
    """
    for i, (cross_idx, circle_idx) in enumerate(matched_indices):
        cross_center = markers[cross_idx]["bbox_center"]
        circle_center = markers[circle_idx]["bbox_center"]

        # Transform coordinates
        x1 = int(cross_center[0] * scale + offset[0])
        y1 = int(cross_center[1] * scale + offset[1])
        x2 = int(circle_center[0] * scale + offset[0])
        y2 = int(circle_center[1] * scale + offset[1])

        # Choose color based on selection
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
    """Draw ROI polygon overlay on the frame.

    Args:
        frame: The frame to draw on
        polygon: RoIPolygon object
        scale: Scale factor applied to image
        offset: (x, y) offset of image in frame
        alpha: Transparency of the overlay
        inverted: If True, darken the area OUTSIDE the polygon (mask visualization)

    Returns:
        Frame with ROI polygon drawn
    """
    # Get polygon exterior coordinates
    coords = np.array(polygon.roi_polygon.exterior.coords)

    # Transform to frame space
    coords_transformed = coords * scale
    coords_transformed[:, 0] += offset[0]
    coords_transformed[:, 1] += offset[1]
    coords_int = coords_transformed.astype(np.int32)

    if inverted:
        # Create a mask for the polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [coords_int], 255)

        # Darken the area outside the polygon
        overlay = frame.copy()
        overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)
        frame[:] = overlay
    else:
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [coords_int], COLOR_ROI_POLYGON)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw polygon outline
    cv2.polylines(frame, [coords_int], True, COLOR_ROI_POLYGON, 2, cv2.LINE_AA)

    return frame


def rotate_image_no_crop(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center without cropping.

    Args:
        image: Input image in HxWxC or HxW format
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image with expanded canvas
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


def write_frames(writer: cv2.VideoWriter, frame: np.ndarray, num_frames: int) -> None:
    """Write the same frame multiple times to the video.

    Args:
        writer: VideoWriter object
        frame: Frame to write
        num_frames: Number of times to write the frame
    """
    for _ in range(num_frames):
        writer.write(frame)


def animate_zoom_to_roi(
    full_image: np.ndarray,
    roi_bounds: tuple,
    num_frames: int,
    frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
) -> list:
    """Generate frames that smoothly zoom from full image view to ROI crop.

    Uses cosine easing for smooth acceleration/deceleration.
    Interpolates the crop region from full image bounds to ROI bounds.

    Args:
        full_image: The full rotated image in HxWxC format
        roi_bounds: Tuple of (minx, miny, maxx, maxy) for the ROI region
        num_frames: Number of frames to generate for the animation
        frame_size: Target frame size (width, height)

    Returns:
        List of frames for the zoom animation
    """
    h, w = full_image.shape[:2]
    minx, miny, maxx, maxy = roi_bounds

    frames = []
    for i in range(num_frames):
        progress = i / max(num_frames - 1, 1)
        # Cosine easing for smooth animation
        eased = 0.5 - 0.5 * np.cos(np.pi * progress)

        # Interpolate crop bounds from full image to ROI
        curr_minx = int(0 + (minx - 0) * eased)
        curr_miny = int(0 + (miny - 0) * eased)
        curr_maxx = int(w + (maxx - w) * eased)
        curr_maxy = int(h + (maxy - h) * eased)

        # Ensure valid bounds
        curr_minx = max(0, curr_minx)
        curr_miny = max(0, curr_miny)
        curr_maxx = min(w, curr_maxx)
        curr_maxy = min(h, curr_maxy)

        # Crop the region
        cropped = full_image[curr_miny:curr_maxy, curr_minx:curr_maxx]

        # Resize to fit frame (maintaining aspect ratio)
        crop_h, crop_w = cropped.shape[:2]
        target_w, target_h = frame_size

        # Account for progress bar
        usable_height = target_h - 80
        scale = min(target_w / crop_w, usable_height / crop_h) * 0.95

        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create frame and center the image
        frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (usable_height - new_h) // 2
        frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        frames.append(frame)

    return frames


class TestVideoAnimation(unittest.TestCase):
    """Test case for generating pipeline animation video."""

    def test_create_pipeline_video(self):
        """Generate video animation of the marker detection and masking pipeline."""
        # Create output directory
        output_dir = TEST_RESULTS_DIR / "video_animation"
        output_dir.mkdir(exist_ok=True)

        # Configuration
        pixel_size = 0.065789
        marker_group = {
            "cross": np.array((4, 8), dtype=float),
            "circle": np.array((56, 8), dtype=float),
        }
        marker_group_pixels = marker_group_to_pixel_coordinates(marker_group, pixel_size)

        # Load model
        model = MarkerDetectionModel(
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"
        )

        # Load ROI structure
        srsl = SingleRoIStructureLibrary(
            lookup_path=Path(dmc_masking.__file__).parent.parent
            / "artifacts/chamber_structure.json",
            structure_name="NormaleBox-pillar-inner",
            pixel_size=pixel_size,
        )
        _, roi_polygon, _ = srsl("0000")

        # Load image
        image_path = Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0000.png"
        original_image = cv2.imread(str(image_path))

        # Run pipeline to get all intermediate states
        markers = model.predict_markers(original_image)
        matched_indices = match_markers(markers, marker_group=marker_group_pixels, tolerance=60)

        if len(matched_indices) == 0:
            self.fail("No markers matched in original image")

        # Compute rotation angle
        angles = compute_marker_group_angles(markers, matched_indices, marker_group_pixels)
        rotation_angle = np.mean(angles)

        # Prepare video writer
        output_path = output_dir / "pipeline_animation.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        total_steps = 7

        # ==================== STEP 1: Raw Image ====================
        frame, scale, offset = prepare_frame(original_image)
        frame = add_step_title(frame, "Raw Input Image")
        frame = draw_progress_bar(frame, 1, total_steps, "Raw Image")
        write_frames(writer, frame, int(1.5 * FPS))

        # ==================== STEP 2: Marker Detection ====================
        frame, scale, offset = prepare_frame(original_image)
        frame = draw_markers(frame, markers, scale, offset)
        frame = add_step_title(frame, "Marker Detection")
        frame = draw_progress_bar(frame, 2, total_steps, "Marker Detection")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 3: Marker Pair Matching ====================
        frame, scale, offset = prepare_frame(original_image)
        frame = draw_markers(frame, markers, scale, offset)
        frame = draw_matched_pairs(frame, markers, matched_indices, scale, offset)
        frame = add_step_title(frame, "Marker Pair Matching")
        frame = draw_progress_bar(frame, 3, total_steps, "Marker Matching")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 4: ROI Selection ====================
        # Find the selected pair (the one with ROI inside image)
        # For now, highlight the first matched pair
        selected_pair_idx = 1  # Will be updated based on apply_mask logic

        # Get highlight indices for the selected pair
        cross_idx, circle_idx = matched_indices[selected_pair_idx]
        highlight_indices = [cross_idx, circle_idx]

        frame, scale, offset = prepare_frame(original_image)
        frame = draw_markers(frame, markers, scale, offset, highlight_indices)
        frame = draw_matched_pairs(
            frame, markers, matched_indices, scale, offset, selected_pair_idx
        )
        frame = add_step_title(frame, "ROI Selection (Valid Marker Pair)")
        frame = draw_progress_bar(frame, 4, total_steps, "ROI Selection")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 5: Rotation Animation ====================
        num_rotation_frames = int(2 * FPS)  # 2 seconds of rotation

        for i in range(num_rotation_frames):
            progress = i / num_rotation_frames
            # Use easing function for smoother animation
            eased_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
            current_angle = rotation_angle * eased_progress

            # Rotate image (same direction as pipeline)
            rotated_img = rotate_image_no_crop(original_image, current_angle)

            # Prepare frame
            frame, scale, offset = prepare_frame(rotated_img)
            frame = add_step_title(frame, f"Rotation: {current_angle:.1f}° / {rotation_angle:.1f}°")
            frame = draw_progress_bar(
                frame, 5, total_steps, "Image Rotation", step_progress=progress
            )
            writer.write(frame)

        # ==================== STEP 6: Masking ====================
        # Rotate image and markers for final masking (same direction as pipeline)
        rotated_image_chw = np.moveaxis(original_image, -1, 0)  # HWC to CHW
        rotated_result, rotated_markers = rotate_image_and_markers(
            rotated_image_chw, markers, rotation_angle
        )
        rotated_image_hwc = np.moveaxis(rotated_result, 0, -1)  # CHW to HWC

        # Get the translated ROI polygon for the selected pair
        cross_marker = rotated_markers[cross_idx]
        circle_marker = rotated_markers[circle_idx]

        width = np.abs(cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0])
        expected_width = np.abs(marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0])
        diff = width - expected_width

        translated_polygon = roi_polygon.translate(
            x=cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
            y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1],
        )

        # Initialize masked_overlay for use in both step 6 and 7
        masked_overlay = None

        # Get the uncropped mask from apply_mask
        try:
            _, uncropped_mask = apply_mask(
                matched_indices,
                rotated_markers,
                marker_group_pixels,
                roi_polygon,
                rotated_result,
                return_uncropped=True,
            )

            # Create visualization with mask overlay (darken masked regions)
            masked_overlay = rotated_image_hwc.copy()
            # Apply red tint to masked regions (True = masked out)
            masked_overlay[uncropped_mask] = (
                0.3 * masked_overlay[uncropped_mask] + 0.7 * np.array([0, 0, 128])
            ).astype(np.uint8)

            frame, scale, offset = prepare_frame(masked_overlay)
            frame = add_step_title(frame, "ROI Mask Applied")
            frame = draw_progress_bar(frame, 6, total_steps, "Masking")
            write_frames(writer, frame, int(2 * FPS))

        except ValueError as e:
            print(f"Could not compute mask for step 6: {e}")
            # Fallback to showing just the rotated image with polygon overlay
            frame, scale, offset = prepare_frame(rotated_image_hwc)
            frame = draw_roi_polygon(
                frame, translated_polygon, scale, offset, alpha=0.3, inverted=True
            )
            frame = add_step_title(frame, "ROI Mask Overlay")
            frame = draw_progress_bar(frame, 6, total_steps, "Masking")
            write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 7: Cropping with Zoom Animation ====================
        try:
            # Get ROI bounds for zoom animation
            roi_bounds = tuple(map(int, map(np.round, translated_polygon.roi_polygon.bounds)))

            # Use the masked overlay image (with red tint on masked regions) for zoom
            # If masked_overlay wasn't created in step 6, create it here
            if masked_overlay is None:
                masked_overlay = rotated_image_hwc.copy()
                _, temp_mask = apply_mask(
                    matched_indices,
                    rotated_markers,
                    marker_group_pixels,
                    roi_polygon,
                    rotated_result,
                    return_uncropped=True,
                )
                masked_overlay[temp_mask] = (
                    0.3 * masked_overlay[temp_mask] + 0.7 * np.array([0, 0, 128])
                ).astype(np.uint8)

            # Generate zoom animation frames using the masked overlay image
            num_zoom_frames = int(2 * FPS)  # 2 seconds of zoom
            zoom_frames = animate_zoom_to_roi(masked_overlay, roi_bounds, num_zoom_frames)

            # Write zoom animation frames with progress bar
            for i, zoom_frame in enumerate(zoom_frames):
                progress = i / max(num_zoom_frames - 1, 1)
                zoom_frame = add_step_title(zoom_frame, "Zooming to ROI Region")
                zoom_frame = draw_progress_bar(
                    zoom_frame, 7, total_steps, "Cropping", step_progress=progress
                )
                writer.write(zoom_frame)

            # Show final cropped result with mask visualization
            cropped_image, cropped_mask = apply_mask(
                matched_indices,
                rotated_markers,
                marker_group_pixels,
                roi_polygon,
                rotated_result,
            )

            # Apply mask visualization to cropped result
            cropped_hwc = np.moveaxis(cropped_image, 0, -1)
            masked_display = cropped_hwc.copy()
            # Apply dark red tint to masked regions (consistent with step 6)
            masked_display[cropped_mask] = (
                0.3 * masked_display[cropped_mask] + 0.7 * np.array([0, 0, 128])
            ).astype(np.uint8)

            frame, scale, offset = prepare_frame(masked_display)
            frame = add_step_title(frame, "Final Cropped Result with Mask")
            frame = draw_progress_bar(frame, 7, total_steps, "Cropping", step_progress=1.0)
            write_frames(writer, frame, int(1.5 * FPS))

        except ValueError as e:
            print(f"Could not apply mask: {e}")
            # Just show the rotated image with polygon for remaining time
            write_frames(writer, frame, int(3.5 * FPS))

        # Release video writer
        writer.release()

        # Verify output
        self.assertTrue(output_path.exists(), "Video file was not created")
        print(f"\nVideo saved to: {output_path}")


if __name__ == "__main__":
    unittest.main()

"""Test for generating video animation of the marker detection and masking pipeline."""

import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

# Cell segmentation imports
import torch

try:
    from acia.segm.local import THWCSequenceSource
    from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter
    from acia.viz import render_segmentation_mask

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

import dmc_masking
from dmc_masking import MarkerDetectionModel
from dmc_masking.mask import SingleRoIStructureLibrary, apply_mask
from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers
from dmc_masking.rotation import compute_marker_group_angles, rotate_image_and_markers
from dmc_masking.visualization import (
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    add_step_title,
    animate_zoom_to_roi,
    draw_progress_bar,
    draw_roi_polygon,
    prepare_frame,
    render_markers_to_frame,
    write_frames,
)

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


class TestVideoAnimation(unittest.TestCase):
    """Test case for generating pipeline animation video."""

    @pytest.mark.skipif(not ACIA_AVAILABLE, reason="acia package not installed")
    def test_create_pipeline_video(self):
        """Generate video animation of the marker detection and masking pipeline."""
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
            # Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"
            Path(
                "/home/seiffarth_l/projects/DMC_new/dmc-train/runs/v8_detect_s_imgsz640/weights/best.pt"
            )
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

        # Run pipeline to get intermediate states
        markers = model.predict_markers(original_image)
        matched_indices = match_markers(markers, marker_group=marker_group_pixels, tolerance=60)

        if len(matched_indices) == 0:
            self.fail("No markers matched in original image")

        angles = compute_marker_group_angles(markers, matched_indices, marker_group_pixels)
        rotation_angle = np.mean(angles)

        # Prepare video writer
        output_path = output_dir / "pipeline_animation.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        total_steps = 8

        # ==================== STEP 1: Raw Image ====================
        frame, scale, offset = prepare_frame(original_image)
        frame = add_step_title(frame, "Raw Input Image")
        frame = draw_progress_bar(frame, 1, total_steps, "Raw Image")
        write_frames(writer, frame, int(1.5 * FPS))

        # ==================== STEP 2: Marker Detection ====================
        frame = render_markers_to_frame(original_image, markers, [])
        frame = add_step_title(frame, "Marker Detection")
        frame = draw_progress_bar(frame, 2, total_steps, "Marker Detection")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 3: Marker Pair Matching ====================
        # Calculate unmatched marker indices for fading
        matched_marker_indices = set()
        for cross_idx, circle_idx in matched_indices:
            matched_marker_indices.add(cross_idx)
            matched_marker_indices.add(circle_idx)
        unmatched_indices = [i for i in range(len(markers)) if i not in matched_marker_indices]

        frame = render_markers_to_frame(
            original_image, markers, matched_indices, faded_indices=unmatched_indices
        )
        frame = add_step_title(frame, "Marker Pair Matching")
        frame = draw_progress_bar(frame, 3, total_steps, "Marker Matching")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 4: ROI Selection ====================
        selected_pair_idx = 1
        cross_idx, circle_idx = matched_indices[selected_pair_idx]
        highlight_indices = [cross_idx, circle_idx]

        frame = render_markers_to_frame(
            original_image,
            markers,
            matched_indices,
            highlight_indices=highlight_indices,
            selected_pair_idx=selected_pair_idx,
        )
        frame = add_step_title(frame, "ROI Selection (Valid Marker Pair)")
        frame = draw_progress_bar(frame, 4, total_steps, "ROI Selection")
        write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 5: Rotation Animation ====================
        num_rotation_frames = int(2 * FPS)
        original_image_chw = np.moveaxis(original_image, -1, 0)

        for i in range(num_rotation_frames):
            progress = i / num_rotation_frames
            eased_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
            current_angle = rotation_angle * eased_progress

            # Rotate image and markers together
            rotated_result_chw, current_rotated_markers = rotate_image_and_markers(
                original_image_chw, markers, current_angle
            )
            rotated_img = np.moveaxis(rotated_result_chw, 0, -1)

            # Render with markers (same visualization as steps 2-4)
            frame = render_markers_to_frame(
                rotated_img,
                current_rotated_markers,
                matched_indices,
                highlight_indices=highlight_indices,
                selected_pair_idx=selected_pair_idx,
                faded_indices=unmatched_indices,
            )
            frame = add_step_title(frame, f"Rotation: {current_angle:.1f}° / {rotation_angle:.1f}°")
            frame = draw_progress_bar(
                frame, 5, total_steps, "Image Rotation", step_progress=progress
            )
            writer.write(frame)

        # ==================== STEP 6: Masking ====================
        rotated_image_chw = np.moveaxis(original_image, -1, 0)
        rotated_result, rotated_markers = rotate_image_and_markers(
            rotated_image_chw, markers, rotation_angle
        )
        rotated_image_hwc = np.moveaxis(rotated_result, 0, -1)

        cross_marker = rotated_markers[cross_idx]
        circle_marker = rotated_markers[circle_idx]
        width = np.abs(cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0])
        expected_width = np.abs(marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0])
        diff = width - expected_width

        translated_polygon = roi_polygon.translate(
            x=cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
            y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1],
        )

        masked_overlay = None
        try:
            _, uncropped_mask = apply_mask(
                matched_indices,
                rotated_markers,
                marker_group_pixels,
                roi_polygon,
                rotated_result,
                return_uncropped=True,
            )
            masked_overlay = rotated_image_hwc.copy()
            masked_overlay[uncropped_mask] = (
                0.3 * masked_overlay[uncropped_mask] + 0.7 * np.array([128, 128, 128])
            ).astype(np.uint8)

            frame, scale, offset = prepare_frame(masked_overlay)
            frame = add_step_title(frame, "ROI Mask Applied")
            frame = draw_progress_bar(frame, 6, total_steps, "Masking")
            write_frames(writer, frame, int(2 * FPS))

        except ValueError as e:
            print(f"Could not compute mask for step 6: {e}")
            frame, scale, offset = prepare_frame(rotated_image_hwc)
            frame = draw_roi_polygon(frame, translated_polygon, scale, offset, inverted=True)
            frame = add_step_title(frame, "ROI Mask Overlay")
            frame = draw_progress_bar(frame, 6, total_steps, "Masking")
            write_frames(writer, frame, int(2 * FPS))

        # ==================== STEP 7: Cropping with Zoom Animation ====================
        try:
            roi_bounds = tuple(map(int, map(np.round, translated_polygon.roi_polygon.bounds)))

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
                    0.3 * masked_overlay[temp_mask] + 0.7 * np.array([128, 128, 128])
                ).astype(np.uint8)

            num_zoom_frames = int(2 * FPS)
            zoom_frames = animate_zoom_to_roi(masked_overlay, roi_bounds, num_zoom_frames)

            for i, zoom_frame in enumerate(zoom_frames):
                progress = i / max(num_zoom_frames - 1, 1)
                zoom_frame = add_step_title(zoom_frame, "Zooming to ROI Region")
                zoom_frame = draw_progress_bar(
                    zoom_frame, 7, total_steps, "Cropping", step_progress=progress
                )
                writer.write(zoom_frame)

            # Final cropped result with mask
            cropped_image, cropped_mask = apply_mask(
                matched_indices,
                rotated_markers,
                marker_group_pixels,
                roi_polygon,
                rotated_result,
            )
            cropped_hwc = np.moveaxis(cropped_image, 0, -1)
            masked_display = cropped_hwc.copy()
            masked_display[cropped_mask] = (
                0.3 * masked_display[cropped_mask] + 0.7 * np.array([128, 128, 128])
            ).astype(np.uint8)

            frame, scale, offset = prepare_frame(masked_display)
            frame = add_step_title(frame, "Final Cropped Result with Mask")
            frame = draw_progress_bar(frame, 7, total_steps, "Cropping", step_progress=1.0)
            write_frames(writer, frame, int(1.5 * FPS))

        except ValueError as e:
            print(f"Could not apply mask: {e}")
            write_frames(writer, frame, int(3.5 * FPS))

        # ==================== STEP 8: Cell Segmentation ====================
        try:
            # Prepare image for acia (TxHxWxC format with T=1)
            # Convert BGR to RGB for visualization, use grayscale for segmentation
            cropped_rgb = cv2.cvtColor(cropped_hwc, cv2.COLOR_BGR2RGB)
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)  # TxHxWxC (3 channels)
            source = THWCSequenceSource(segm_input)

            # Run CellPose segmentation (use first channel / grayscale)
            warnings.filterwarnings("ignore", category=FutureWarning)
            segmenter = CellposeSAMSegmenter()
            with torch.no_grad():
                segmentation_result = segmenter(source.to_channel(0))

            # Render segmentation with colorful masks
            rendered = render_segmentation_mask(source, segmentation_result, alpha=0.5)
            segmented_frame = rendered.image_stack[0]  # Get first (only) frame, RGB format

            # Convert RGB to BGR for OpenCV
            segmented_bgr = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

            # Apply gray ROI mask on top
            final_display = segmented_bgr.copy()
            final_display[cropped_mask] = (
                0.3 * final_display[cropped_mask] + 0.7 * np.array([128, 128, 128])
            ).astype(np.uint8)

            # Write final frame
            frame, scale, offset = prepare_frame(final_display)
            frame = add_step_title(frame, "Cell Segmentation with ROI Mask")
            frame = draw_progress_bar(frame, 8, total_steps, "Segmentation", step_progress=1.0)
            write_frames(writer, frame, int(2 * FPS))

        except Exception as e:
            print(f"Could not perform cell segmentation: {e}")
            import traceback

            traceback.print_exc()

        writer.release()
        self.assertTrue(output_path.exists(), "Video file was not created")
        print(f"\nVideo saved to: {output_path}")

        # Generate low quality GIF for README
        gif_path = output_dir / "pipeline_animation.gif"
        cap = cv2.VideoCapture(str(output_path))
        frames = []
        frame_skip = 6  # Take every 6th frame to reduce size (5 fps effective)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                # Resize to lower resolution and convert BGR to RGB
                small_frame = cv2.resize(frame, (320, 360))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            frame_idx += 1
        cap.release()

        # Save as GIF using imageio
        import imageio

        imageio.mimsave(str(gif_path), frames, duration=200, loop=0)  # 200ms = 5fps
        print(f"GIF saved to: {gif_path}")


if __name__ == "__main__":
    unittest.main()

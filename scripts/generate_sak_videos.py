"""Generate pipeline walkthrough videos for all 8 SAK chamber types."""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import torch
    from acia.segm.local import THWCSequenceSource
    from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter
    from acia.viz import render_segmentation_mask

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

import dmc_masking
from dmc_masking import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dmc_masking.chip import ChipStructureLibrary
from dmc_masking.mask import apply_mask
from dmc_masking.match import match_markers
from dmc_masking.rotation import compute_marker_group_angles, rotate_image_and_markers
from dmc_masking.utils import normalize_image
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

ARTIFACTS_DIR = Path(dmc_masking.__file__).parent.parent / "artifacts"

# Image-to-chamber-type mapping (from tests/test_full.py)
CONFIGS = [
    {"file_name": "0000.png", "chamber_type": "NormaleBox-pillar-inner"},
    {"file_name": "0001.png", "chamber_type": "BigBox-pillar-inner"},
    {"file_name": "0003.png", "chamber_type": "OpenBox-inner"},
    {"file_name": "0005.png", "chamber_type": "OpenBox-collector-inner"},
    {"file_name": "0006.png", "chamber_type": "BigBox-inner"},
    {"file_name": "0007.png", "chamber_type": "NormaleBox-inner"},
    {"file_name": "0008.png", "chamber_type": "Mothermachine-2x-inner"},
    {"file_name": "0009.tif", "chamber_type": "Mothermachine-inner"},
]


def load_image(image_path: Path) -> np.ndarray:
    """Load an image, handling .tif normalization."""
    if image_path.suffix == ".tif":
        if tifffile is None:
            raise ImportError("tifffile is required to read .tif images")
        image = tifffile.imread(image_path)
        image = normalize_image(image)
        image = np.stack((image,) * 3, axis=-1)
    else:
        image = cv2.imread(str(image_path))
    return image


def generate_pipeline_video(
    image: np.ndarray,
    chamber_name: str,
    roi_polygon,
    marker_group_pixels: dict[str, np.ndarray],
    model: MarkerDetectionModel,
    output_path: Path,
):
    """Generate a pipeline walkthrough video for one chamber type."""
    total_steps = 8 if ACIA_AVAILABLE else 7

    # Run pipeline
    markers = model.predict_markers(image)
    matched_indices = match_markers(markers, marker_group=marker_group_pixels, tolerance=60)

    if len(matched_indices) == 0:
        print(f"  WARNING: No markers matched for {chamber_name}, skipping.")
        return False

    angles = compute_marker_group_angles(markers, matched_indices, marker_group_pixels)
    rotation_angle = np.mean(angles)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # ==================== STEP 1: Raw Image ====================
    frame, scale, offset = prepare_frame(image)
    frame = add_step_title(frame, "Raw Input Image")
    frame = draw_progress_bar(frame, 1, total_steps, "Raw Image")
    write_frames(writer, frame, int(1.5 * FPS))

    # ==================== STEP 2: Marker Detection ====================
    frame = render_markers_to_frame(image, markers, [])
    frame = add_step_title(frame, "Marker Detection")
    frame = draw_progress_bar(frame, 2, total_steps, "Marker Detection")
    write_frames(writer, frame, int(2 * FPS))

    # ==================== STEP 3: Marker Pair Matching ====================
    matched_marker_indices = set()
    for cross_idx, circle_idx in matched_indices:
        matched_marker_indices.add(cross_idx)
        matched_marker_indices.add(circle_idx)
    unmatched_indices = [i for i in range(len(markers)) if i not in matched_marker_indices]

    frame = render_markers_to_frame(
        image, markers, matched_indices, faded_indices=unmatched_indices
    )
    frame = add_step_title(frame, "Marker Pair Matching")
    frame = draw_progress_bar(frame, 3, total_steps, "Marker Matching")
    write_frames(writer, frame, int(2 * FPS))

    # ==================== STEP 4: ROI Selection ====================
    h, w = image.shape[:2]
    best_pair_idx = 0
    best_margin = -np.inf
    for pair_idx, (ci, oi) in enumerate(matched_indices):
        cross_m, circle_m = markers[ci], markers[oi]
        width = np.abs(cross_m["bbox_center"][0] - circle_m["bbox_center"][0])
        expected_width = np.abs(marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0])
        diff = width - expected_width
        rp = roi_polygon.translate(
            x=cross_m["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
            y=cross_m["bbox_center"][1] + marker_group_pixels["cross"][1],
        )
        xmin, ymin, xmax, ymax = rp.roi_polygon.bounds
        if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
            continue
        margin = min(abs(xmin), abs(w - xmax), abs(ymin), abs(h - ymax))
        if margin > best_margin:
            best_margin = margin
            best_pair_idx = pair_idx

    selected_pair_idx = best_pair_idx
    cross_idx, circle_idx = matched_indices[selected_pair_idx]
    highlight_indices = [cross_idx, circle_idx]

    frame = render_markers_to_frame(
        image,
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
    image_chw = np.moveaxis(image, -1, 0)

    for i in range(num_rotation_frames):
        progress = i / num_rotation_frames
        eased_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
        current_angle = rotation_angle * eased_progress

        rotated_result_chw, current_rotated_markers = rotate_image_and_markers(
            image_chw, markers, current_angle
        )
        rotated_img = np.clip(np.moveaxis(rotated_result_chw, 0, -1), 0, 255).astype(np.uint8)

        frame = render_markers_to_frame(
            rotated_img,
            current_rotated_markers,
            matched_indices,
            highlight_indices=highlight_indices,
            selected_pair_idx=selected_pair_idx,
            faded_indices=unmatched_indices,
        )
        frame = add_step_title(
            frame, f"Rotation: {current_angle:.1f}\u00b0 / {rotation_angle:.1f}\u00b0"
        )
        frame = draw_progress_bar(frame, 5, total_steps, "Image Rotation", step_progress=progress)
        writer.write(frame)

    # ==================== STEP 6: Masking ====================
    rotated_result, rotated_markers = rotate_image_and_markers(image_chw, markers, rotation_angle)
    rotated_image_hwc = np.clip(np.moveaxis(rotated_result, 0, -1), 0, 255).astype(np.uint8)

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
        print(f"  Could not compute mask for step 6: {e}")
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
        print(f"  Could not apply mask: {e}")
        write_frames(writer, frame, int(3.5 * FPS))

    # ==================== STEP 8: Cell Segmentation ====================
    if ACIA_AVAILABLE:
        try:
            cropped_rgb = cv2.cvtColor(cropped_hwc, cv2.COLOR_BGR2RGB)
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)

            warnings.filterwarnings("ignore", category=FutureWarning)
            segmenter = CellposeSAMSegmenter()
            with torch.no_grad():
                segmentation_result = segmenter(source.to_channel(0))

            rendered = render_segmentation_mask(source, segmentation_result, alpha=0.5)
            segmented_frame = rendered.image_stack[0]
            segmented_bgr = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

            final_display = segmented_bgr.copy()
            final_display[cropped_mask] = (
                0.3 * final_display[cropped_mask] + 0.7 * np.array([128, 128, 128])
            ).astype(np.uint8)

            frame, scale, offset = prepare_frame(final_display)
            frame = add_step_title(frame, "Cell Segmentation with ROI Mask")
            frame = draw_progress_bar(frame, 8, total_steps, "Segmentation", step_progress=1.0)
            write_frames(writer, frame, int(2 * FPS))

        except Exception as e:
            print(f"  Could not perform cell segmentation: {e}")

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate pipeline videos for all 8 SAK chamber types."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "sak_videos",
        help="Output directory for generated videos",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load chip structure library (new API)
    chip_config_path = ARTIFACTS_DIR / "chips" / "sak.json"
    lib = ChipStructureLibrary.from_file(chip_config_path)

    # Load model once
    print("Loading marker detection model...")
    model = MarkerDetectionModel(DEFAULT_MODEL_PATH)

    success_count = 0
    for conf in CONFIGS:
        file_name = conf["file_name"]
        chamber_type = conf["chamber_type"]

        print(f"\nProcessing {chamber_type} ({file_name})...")

        image_path = ARTIFACTS_DIR / "images" / "sak" / file_name
        if not image_path.exists():
            print(f"  Image not found: {image_path}, skipping.")
            continue

        image = load_image(image_path)

        # Get polygon and marker positions directly by chamber type
        # (image file names don't correspond to blueprint ROI IDs)
        roi_polygon = lib.polygon_library[chamber_type]
        marker_group_pixels = lib.marker_group_configs[chamber_type]

        output_path = args.output_dir / f"{chamber_type}.mp4"
        ok = generate_pipeline_video(
            image, chamber_type, roi_polygon, marker_group_pixels, model, output_path
        )
        if ok:
            print(f"  Saved: {output_path}")
            success_count += 1

    print(f"\nDone. Generated {success_count}/{len(CONFIGS)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()

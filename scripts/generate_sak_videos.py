"""Generate pipeline walkthrough videos and/or stills for all SAK chamber types."""

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
    from acia.viz import colorize_instance_mask

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

from dart_mlci import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dart_mlci.chip import ChipStructureLibrary
from dart_mlci.constants import ARTIFACTS_DIR, DEFAULT_PIXEL_SIZE_UM
from dart_mlci.mask import apply_mask, filter_segmentation_by_mask
from dart_mlci.match import match_markers
from dart_mlci.rotation import compute_marker_group_angles, rotate_image_and_markers
from dart_mlci.utils import normalize_image
from dart_mlci.visualization import (
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


def _maybe_title(frame, title, args):
    if args.no_titles:
        return frame
    return add_step_title(frame, title)


def _maybe_progress(frame, step, total, name, progress, args):
    if args.no_progress_bar:
        return frame
    return draw_progress_bar(frame, step, total, name, step_progress=progress)


def _crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Crop all-black rows/cols from the edges of *img*."""
    mask = img.sum(axis=-1) > 0
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _add_scalebar_cv(
    frame: np.ndarray,
    source_image_shape: tuple,
    pixel_size_um: float,
    bar_um: float,
    frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
) -> np.ndarray:
    """Draw a `bar_um` micrometer scale bar at the bottom-right of *frame*.

    The bar pixel-width is derived from ``prepare_frame``'s scale factor
    applied to *source_image_shape*, so the bar represents the requested
    physical length regardless of how prepare_frame letterboxed the image.
    """
    if bar_um <= 0 or pixel_size_um <= 0:
        return frame
    target_w, target_h = frame_size
    progress_bar_height = 80
    usable_height = target_h - progress_bar_height
    src_h, src_w = source_image_shape[:2]
    scale = min(target_w / src_w, usable_height / src_h) * 0.95
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    x_offset = (target_w - new_w) // 2
    y_offset = (usable_height - new_h) // 2

    bar_px = max(1, round(bar_um / pixel_size_um * scale))
    bar_h = 6
    margin = 16
    x1 = x_offset + new_w - margin - bar_px
    x2 = x_offset + new_w - margin
    y_bar = y_offset + new_h - margin - bar_h
    cv2.rectangle(frame, (x1, y_bar), (x2, y_bar + bar_h), (255, 255, 255), -1)
    label = f"{bar_um:g} um"
    (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = x2 - tw
    text_y = y_bar - 6
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _save_still(frame, stills_dir, index, name, args, *, source_shape=None):
    if stills_dir is None:
        return
    bar_um = getattr(args, "scalebar_um", 0.0) or 0.0
    pixel_size = getattr(args, "pixel_size", None) or DEFAULT_PIXEL_SIZE_UM
    if bar_um > 0 and source_shape is not None:
        frame = _add_scalebar_cv(frame.copy(), source_shape, pixel_size, bar_um)
    img = _crop_black_borders(frame) if args.crop_black_borders else frame
    path = stills_dir / f"{index:02d}_{name}.png"
    cv2.imwrite(str(path), img)


def _write_video(writer, frame, n):
    if writer is None:
        return
    write_frames(writer, frame, n)


def _hatch_excluded(
    image: np.ndarray,
    excluded: np.ndarray,
    *,
    period: int = 12,
    thickness: int = 2,
    line_alpha: float = 0.8,
    fill_color: tuple[int, int, int] = (128, 128, 128),
    fill_alpha: float = 0.7,
) -> np.ndarray:
    """Mark *excluded* pixels with a translucent gray fill plus black hatching.

    The gray tint shows the extent of the removed region; the diagonal black
    lines on top make it unambiguous as excluded content (so it cannot be
    confused with grayscale microfluidic structures).
    """
    if not excluded.any():
        return image
    out = image.astype(np.float32, copy=True)
    fill = np.array(fill_color, dtype=np.float32)
    out[excluded] = (1.0 - fill_alpha) * out[excluded] + fill_alpha * fill
    h, w = excluded.shape
    yy, xx = np.indices((h, w))
    stripes = ((xx + yy) % period) < thickness
    hatch = excluded & stripes
    if hatch.any():
        out[hatch] *= 1.0 - line_alpha
    return out.astype(np.uint8)


def generate_pipeline_video(
    image: np.ndarray,
    chamber_name: str,
    roi_polygon,
    marker_group_pixels: dict[str, np.ndarray],
    model: MarkerDetectionModel,
    video_path: Path | None,
    stills_dir: Path | None,
    args,
):
    """Render the pipeline for one chamber as a video and/or a set of stills."""
    total_steps = 5

    # Run pipeline
    markers = model.predict_markers(image)
    matched_indices = match_markers(markers, marker_group=marker_group_pixels, tolerance=60)

    if len(matched_indices) == 0:
        print(f"  WARNING: No markers matched for {chamber_name}, skipping.")
        return False

    angles = compute_marker_group_angles(markers, matched_indices, marker_group_pixels)
    rotation_angle = np.mean(angles)

    # Video writer only if requested
    writer = None
    if video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # ==================== STEP 1: Detection ====================
    frame, scale, offset = prepare_frame(image)
    frame = _maybe_title(frame, "Detection: Raw Input Image", args)
    frame = _maybe_progress(frame, 1, total_steps, "Detection", 0.0, args)
    _write_video(writer, frame, int(1.5 * FPS))
    _save_still(frame, stills_dir, 1, "detection_raw", args, source_shape=image.shape)

    frame = render_markers_to_frame(image, markers, [])
    frame = _maybe_title(frame, "Detection: Marker Detection", args)
    frame = _maybe_progress(frame, 1, total_steps, "Detection", 1.0, args)
    _write_video(writer, frame, int(2 * FPS))
    _save_still(frame, stills_dir, 2, "detection_markers", args, source_shape=image.shape)

    # ==================== STEP 2: Matching ====================
    matched_marker_indices = set()
    for cross_idx, circle_idx in matched_indices:
        matched_marker_indices.add(cross_idx)
        matched_marker_indices.add(circle_idx)
    unmatched_indices = [i for i in range(len(markers)) if i not in matched_marker_indices]

    frame = render_markers_to_frame(
        image, markers, matched_indices, faded_indices=unmatched_indices
    )
    frame = _maybe_title(frame, "Matching: Marker Pair Matching", args)
    frame = _maybe_progress(frame, 2, total_steps, "Matching", 0.5, args)
    _write_video(writer, frame, int(2 * FPS))
    _save_still(frame, stills_dir, 3, "matching_pairs", args, source_shape=image.shape)

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
    frame = _maybe_title(frame, "Matching: ROI Selection (Valid Marker Pair)", args)
    frame = _maybe_progress(frame, 2, total_steps, "Matching", 1.0, args)
    _write_video(writer, frame, int(2 * FPS))
    _save_still(frame, stills_dir, 4, "matching_selected", args, source_shape=image.shape)

    # ==================== STEP 3: Rotation ====================
    num_rotation_frames = int(2 * FPS)
    image_chw = np.moveaxis(image, -1, 0)
    last_rotation_frame = None

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
        frame = _maybe_title(
            frame, f"Rotation: {current_angle:.1f}deg / {rotation_angle:.1f}deg", args
        )
        frame = _maybe_progress(frame, 3, total_steps, "Rotation", progress, args)
        if writer is not None:
            writer.write(frame)
        last_rotation_frame = frame

    # Final rotation still (only the last frame, no intermediates)
    if last_rotation_frame is not None:
        _save_still(
            last_rotation_frame,
            stills_dir,
            5,
            "rotation_final",
            args,
            source_shape=image.shape,
        )

    # ==================== STEP 4: Masking ====================
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
    cropped_hwc = None
    cropped_mask = None
    try:
        _, uncropped_mask = apply_mask(
            matched_indices,
            rotated_markers,
            marker_group_pixels,
            roi_polygon,
            rotated_result,
            return_uncropped=True,
        )
        masked_overlay = _hatch_excluded(rotated_image_hwc.copy(), uncropped_mask)

        frame, scale, offset = prepare_frame(masked_overlay)
        frame = _maybe_title(frame, "Masking: ROI Mask Applied", args)
        frame = _maybe_progress(frame, 4, total_steps, "Masking", 0.3, args)
        _write_video(writer, frame, int(2 * FPS))
        _save_still(
            frame,
            stills_dir,
            6,
            "masking_overlay",
            args,
            source_shape=masked_overlay.shape,
        )

    except ValueError as e:
        print(f"  Could not compute mask for step 4: {e}")
        frame, scale, offset = prepare_frame(rotated_image_hwc)
        frame = draw_roi_polygon(frame, translated_polygon, scale, offset, inverted=True)
        frame = _maybe_title(frame, "Masking: ROI Mask Overlay", args)
        frame = _maybe_progress(frame, 4, total_steps, "Masking", 0.3, args)
        _write_video(writer, frame, int(2 * FPS))
        _save_still(
            frame,
            stills_dir,
            6,
            "masking_overlay",
            args,
            source_shape=rotated_image_hwc.shape,
        )

    # Step 4 continued: Cropping with Zoom Animation
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
            masked_overlay = _hatch_excluded(masked_overlay, temp_mask)

        num_zoom_frames = int(2 * FPS)
        zoom_frames = animate_zoom_to_roi(masked_overlay, roi_bounds, num_zoom_frames)
        last_zoom_frame = None

        for i, zoom_frame in enumerate(zoom_frames):
            progress = i / max(num_zoom_frames - 1, 1)
            zoom_frame = _maybe_title(zoom_frame, "Masking: Zooming to ROI Region", args)
            zoom_frame = _maybe_progress(
                zoom_frame, 4, total_steps, "Masking", 0.3 + 0.7 * progress, args
            )
            if writer is not None:
                writer.write(zoom_frame)
            last_zoom_frame = zoom_frame

        if last_zoom_frame is not None:
            # Zoom-final still shows the cropped chamber view
            roi_w = roi_bounds[2] - roi_bounds[0]
            roi_h = roi_bounds[3] - roi_bounds[1]
            _save_still(
                last_zoom_frame,
                stills_dir,
                7,
                "masking_zoomed",
                args,
                source_shape=(roi_h, roi_w, 3),
            )

        # Final cropped result with mask
        cropped_image, cropped_mask = apply_mask(
            matched_indices,
            rotated_markers,
            marker_group_pixels,
            roi_polygon,
            rotated_result,
        )
        cropped_hwc = np.moveaxis(cropped_image, 0, -1)
        masked_display = _hatch_excluded(cropped_hwc.copy(), cropped_mask)

        frame, scale, offset = prepare_frame(masked_display)
        frame = _maybe_title(frame, "Masking: Final Cropped Result", args)
        frame = _maybe_progress(frame, 4, total_steps, "Masking", 1.0, args)
        _write_video(writer, frame, int(1.5 * FPS))
        _save_still(
            frame,
            stills_dir,
            8,
            "masking_cropped",
            args,
            source_shape=masked_display.shape,
        )

    except ValueError as e:
        print(f"  Could not apply mask: {e}")
        _write_video(writer, frame, int(3.5 * FPS))

    # ==================== STEP 5: Segmentation ====================
    if ACIA_AVAILABLE and cropped_hwc is not None and cropped_mask is not None:
        try:
            cropped_rgb = cv2.cvtColor(cropped_hwc, cv2.COLOR_BGR2RGB)
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)

            warnings.filterwarnings("ignore", category=FutureWarning)
            segmenter = CellposeSAMSegmenter()
            with torch.no_grad():
                segmentation_result = segmenter(source.to_channel(0))

            h, w = cropped_rgb.shape[:2]
            labeled_mask = segmentation_result.toMasks(h, w, binary_mask=False)[0]

            # --- Substep 5a: Raw Instance Segmentation ---
            # Build a saturated HSV palette so cell colors never look gray —
            # otherwise they would blend into the gray-hatched mask overlay.
            rng = np.random.default_rng(42)
            n_ids = int(labeled_mask.max()) + 1
            hsv = np.zeros((n_ids, 1, 3), dtype=np.uint8)
            hsv[..., 0] = rng.integers(0, 180, size=(n_ids, 1))
            hsv[..., 1] = rng.integers(200, 256, size=(n_ids, 1))
            hsv[..., 2] = rng.integers(200, 256, size=(n_ids, 1))
            color_lut = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(n_ids, 3)
            color_lut[0] = (0, 0, 0)

            colored_cells = colorize_instance_mask(labeled_mask, color_lut=color_lut)
            output = cropped_rgb.copy().astype(np.float32)
            cell_area = labeled_mask > 0
            output[cell_area] = 0.5 * colored_cells[cell_area] + 0.5 * output[cell_area]
            output = _hatch_excluded(output.astype(np.uint8), cropped_mask)
            display_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            frame, scale, offset = prepare_frame(display_bgr)
            frame = _maybe_title(frame, "Segmentation: Raw Instance Segmentation", args)
            frame = _maybe_progress(frame, 5, total_steps, "Segmentation", 0.5, args)
            _write_video(writer, frame, int(2 * FPS))
            _save_still(
                frame,
                stills_dir,
                9,
                "segmentation_raw",
                args,
                source_shape=display_bgr.shape,
            )

            # --- Substep 5b: Structure-Aware Filtering ---
            filtered_mask = filter_segmentation_by_mask(labeled_mask, cropped_mask, relabel=False)
            colored_filtered = colorize_instance_mask(filtered_mask, color_lut=color_lut)
            output = cropped_rgb.copy().astype(np.float32)
            cell_area = filtered_mask > 0
            output[cell_area] = 0.5 * colored_filtered[cell_area] + 0.5 * output[cell_area]
            output = _hatch_excluded(output.astype(np.uint8), cropped_mask)
            display_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            frame, scale, offset = prepare_frame(display_bgr)
            frame = _maybe_title(frame, "Segmentation: Structure-Aware Filtering", args)
            frame = _maybe_progress(frame, 5, total_steps, "Segmentation", 1.0, args)
            _write_video(writer, frame, int(2 * FPS))
            _save_still(
                frame,
                stills_dir,
                10,
                "segmentation_filtered",
                args,
                source_shape=display_bgr.shape,
            )

        except Exception as e:
            print(f"  Could not perform cell segmentation: {e}")

    if writer is not None:
        writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate pipeline videos and/or stills for all SAK chamber types."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "sak_videos",
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--stills-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "sak_stills",
        help="Output root for stills; one subdir per chamber type",
    )
    parser.add_argument(
        "--mode",
        choices=["video", "stills", "both"],
        default="video",
        help="What to produce (default: video)",
    )
    parser.add_argument("--no-titles", action="store_true", help="Omit the step-title text overlay")
    parser.add_argument(
        "--no-progress-bar", action="store_true", help="Omit the bottom progress-bar overlay"
    )
    parser.add_argument(
        "--crop-black-borders",
        action="store_true",
        help="Crop all-black letterbox borders from saved stills "
        "(auto-enabled when --no-titles and --no-progress-bar are both set)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_UM,
        help="Pixel size in microns per pixel (camera attribute, default: %(default)s)",
    )
    args = parser.parse_args()

    # Auto-enable border cropping when both overlays are suppressed (publication-clean stills)
    if args.no_titles and args.no_progress_bar:
        args.crop_black_borders = True

    produce_video = args.mode in ("video", "both")
    produce_stills = args.mode in ("stills", "both")

    if produce_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if produce_stills:
        args.stills_dir.mkdir(parents=True, exist_ok=True)

    chip_config_path = ARTIFACTS_DIR / "chips" / "sak.json"
    lib = ChipStructureLibrary.from_file(chip_config_path, pixel_size=args.pixel_size)

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

        roi_polygon = lib.polygon_library[chamber_type]
        marker_group_pixels = lib.marker_group_configs[chamber_type]

        video_path = args.output_dir / f"{chamber_type}.mp4" if produce_video else None
        stills_dir = None
        if produce_stills:
            stills_dir = args.stills_dir / chamber_type
            stills_dir.mkdir(parents=True, exist_ok=True)

        ok = generate_pipeline_video(
            image,
            chamber_type,
            roi_polygon,
            marker_group_pixels,
            model,
            video_path,
            stills_dir,
            args,
        )
        if ok:
            if video_path is not None:
                print(f"  Saved video: {video_path}")
            if stills_dir is not None:
                print(f"  Saved stills: {stills_dir}")
            success_count += 1

    summary_target = args.output_dir if produce_video else args.stills_dir
    print(
        f"\nDone. Processed {success_count}/{len(CONFIGS)} chambers (output root: {summary_target})"
    )


if __name__ == "__main__":
    main()

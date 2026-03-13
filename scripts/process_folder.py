#!/usr/bin/env python
"""Process folder-based experiment with TIFF stacks organized by chamber type.

Processes time-lapse TIFF stacks from a folder structure where subfolders map
to chamber types via a JSON config. Each TIFF is a (T, H, W) stack.

Example usage:
    python scripts/process_folder.py --config dart_experiment/folder_config.json
    python scripts/process_folder.py --config dart_experiment/folder_config.json --max-files 1 --verbose
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

try:
    import matplotlib.pyplot as plt
    from brokenaxes import brokenaxes

    BROKENAXES_AVAILABLE = True
except ImportError:
    BROKENAXES_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pint
    from acia.segm.local import THWCSequenceSource
    from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter
    from acia.viz import colorize_instance_mask, render_scalebar

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

try:
    from acia.segm.processor.omnipose import OmniposeSegmenter

    OMNIPOSE_AVAILABLE = True
except ImportError:
    OMNIPOSE_AVAILABLE = False

import dmc_masking
from dmc_masking import (
    ChipStructureLibrary,
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    PhaseCorrelationRegistration,
    RoIMaskingStep,
    TimelapseRegistration,
)
from dmc_masking.mask import filter_segmentation_by_mask
from dmc_masking.utils import normalize_image

# Preferred display order for chamber types in timing tables (numbers skip 4)
_CHAMBER_ORDER = [
    "NormaleBox-inner",
    "BigBox-inner",
    "OpenBox-inner",
    "NormaleBox-pillar-inner",
    "BigBox-pillar-inner",
    "OpenBox-collector-inner",
    "Mothermachine-2x-inner",
]
_CHAMBER_NUMBERS = {
    name: num for name, num in zip(_CHAMBER_ORDER, [1, 2, 3, 5, 6, 7, 8], strict=False)
}

SEGMENTER_CHOICES = ["cellpose-sam", "omnipose"]

if ACIA_AVAILABLE:
    UNIT_REGISTRY = pint.UnitRegistry()


def create_segmenter(name: str):
    """Create a segmenter instance by name."""
    if name == "cellpose-sam":
        if not ACIA_AVAILABLE:
            print("Error: acia library not available. Install it with: pip install acia")
            sys.exit(1)
        return CellposeSAMSegmenter()
    elif name == "omnipose":
        if not OMNIPOSE_AVAILABLE:
            print("Error: omnipose not available. Install it with: pip install cellpose_omni")
            sys.exit(1)
        return OmniposeSegmenter()
    else:
        print(f"Error: unknown segmenter '{name}'. Choose from: {SEGMENTER_CHOICES}")
        sys.exit(1)


@dataclass
class FrameTimings:
    """Timing information for a single frame's pipeline steps (in seconds)."""

    detection: float = 0.0
    matching: float = 0.0
    rotation: float = 0.0
    registration: float = 0.0
    masking: float = 0.0
    segmentation: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.detection
            + self.matching
            + self.rotation
            + self.registration
            + self.masking
            + self.segmentation
        )

    def as_dict(self) -> dict:
        return {
            "t_detection": self.detection,
            "t_matching": self.matching,
            "t_rotation": self.rotation,
            "t_registration": self.registration,
            "t_masking": self.masking,
            "t_segmentation": self.segmentation,
            "t_total": self.total,
        }


@dataclass
class StackResult:
    """Result for a single TIFF stack."""

    folder: str
    file_name: str
    chamber_type: str
    n_frames: int = 0
    n_success: int = 0
    n_cells_total: int = 0
    success: bool = False
    error: str = ""
    output_dir: str = ""
    frame_timings: list[FrameTimings] = field(default_factory=list)


def add_scalebar(image: np.ndarray, pixel_size: float, bar_um: float = 10) -> np.ndarray:
    source = THWCSequenceSource(image[None, :, :, :].astype(np.uint8))
    result = render_scalebar(
        image_source=source,
        xy_position=(0.80, 0.95),
        size_of_pixel=pixel_size * UNIT_REGISTRY.micrometer,
        bar_width=bar_um * UNIT_REGISTRY.micrometer,
        bar_height=2 * UNIT_REGISTRY.micrometer,
        color=(255, 255, 255),
        font_size=20,
        show_text=True,
    )
    return result.image_stack[0]


def render_frame_visualization(
    cropped_image: np.ndarray,
    labeled_mask: np.ndarray,
    chamber_mask: np.ndarray,
    pixel_size: float,
    alpha: float = 0.5,
) -> np.ndarray:
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
    output = add_scalebar(output, pixel_size)
    return output


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)

    required = ["input_dir", "output_dir", "folders"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


def collect_files(config: dict, base_dir: Path) -> list[dict]:
    """Collect all TIFF files from configured folders."""
    input_dir = base_dir / config["input_dir"]
    files = []

    for folder_name, chamber_type in config["folders"].items():
        folder_path = input_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: folder not found: {folder_path}")
            continue

        for tif_path in sorted(folder_path.glob("*.tif")):
            files.append(
                {
                    "path": tif_path,
                    "folder": folder_name,
                    "file_name": tif_path.stem,
                    "chamber_type": chamber_type,
                }
            )

    return files


def process_stack(
    file_info: dict,
    output_dir: Path,
    detection_step: MarkerDetectionStep,
    chip_lib: ChipStructureLibrary,
    segmenter,
    pixel_size: float,
    enable_registration: bool = False,
    registration_method: str = "ncc",
    save_cropped: bool = False,
    render_stacks: bool = False,
    verbose: bool = False,
    device: str | None = None,
    flip: bool = False,
    allow_truncation: bool = False,
) -> StackResult:
    """Process a single TIFF stack through the full pipeline."""
    tif_path = file_info["path"]
    folder = file_info["folder"]
    file_name = file_info["file_name"]
    chamber_type = file_info["chamber_type"]

    result = StackResult(
        folder=folder,
        file_name=file_name,
        chamber_type=chamber_type,
    )

    # Create output directory
    stack_output_dir = output_dir / folder / file_name
    stack_output_dir.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(stack_output_dir)

    # Load TIFF stack
    try:
        stack = tifffile.imread(str(tif_path))
    except Exception as e:
        result.error = f"Failed to load: {e}"
        return result

    if stack.ndim == 2:
        stack = stack[None, :, :]  # Single frame -> (1, H, W)
    elif stack.ndim != 3:
        result.error = f"Unexpected stack shape: {stack.shape}"
        return result

    n_frames = stack.shape[0]
    result.n_frames = n_frames

    if verbose:
        print(f"    Stack shape: {stack.shape}, dtype: {stack.dtype}")

    # Get chamber-specific pipeline components
    if chamber_type not in chip_lib.polygon_library:
        result.error = f"Unknown chamber type: {chamber_type}"
        return result

    roi_polygon = chip_lib.polygon_library[chamber_type]
    marker_group = chip_lib.marker_group_configs[chamber_type]

    matching_step = MarkerMatchingStep(marker_group, tolerance=60)
    rotation_step = ImageRotationStep()
    masking_step = RoIMaskingStep(marker_group, roi_polygon, allow_truncation=allow_truncation)

    # Phase 1: Detection & Rotation (per-frame)
    if verbose:
        print("    Phase 1: Detection & rotation...")

    frame_data = []
    for t in range(n_frames):
        frame_raw = stack[t]
        if flip:
            frame_raw = frame_raw[::-1]
        frame_uint8 = normalize_image(frame_raw)
        frame_rgb = np.stack((frame_uint8,) * 3, axis=-1)  # HxWx3

        timings = FrameTimings()
        frame_info = {
            "timepoint": t,
            "success": False,
            "rotation_result": None,
            "n_cells": 0,
            "cell_areas": [],
            "dx": 0.0,
            "dy": 0.0,
            "reg_score": float("nan"),
            "error": "",
            "timings": timings,
        }

        try:
            t0 = time.perf_counter()
            det_result = detection_step(frame_rgb)
            timings.detection = time.perf_counter() - t0

            t0 = time.perf_counter()
            match_result = matching_step(det_result)
            timings.matching = time.perf_counter() - t0

            matched_indices = match_result.get("matched_marker_indices", [])
            if not matched_indices:
                raise ValueError("No valid marker pairs found")

            t0 = time.perf_counter()
            rot_result = rotation_step(match_result)
            timings.rotation = time.perf_counter() - t0

            frame_info["rotation_result"] = rot_result
            frame_info["success"] = True
        except Exception as e:
            frame_info["error"] = str(e)
            if verbose:
                print(f"      Frame {t}: Failed - {e}")

        frame_data.append(frame_info)

    n_rotation_ok = sum(1 for f in frame_data if f["success"])
    if verbose:
        print(f"      {n_rotation_ok}/{n_frames} frames rotated successfully")

    if n_rotation_ok == 0:
        result.error = "All frames failed detection/rotation"
        _save_meta(stack_output_dir, frame_data, n_frames)
        return result

    # Phase 2: Optional Registration
    if enable_registration and n_rotation_ok > 1:
        if verbose:
            print("    Phase 2: Registration...")

        if registration_method == "phase":
            registration = PhaseCorrelationRegistration(
                marker_group_pixel=marker_group,
                padding=100,
            )
        else:
            registration = TimelapseRegistration(
                marker_group_pixel=marker_group,
                max_translation=20,
                padding=50,
                device=device,
            )

        # Find reference frame (first successful)
        ref_idx = next(i for i, f in enumerate(frame_data) if f["success"])
        ref_rot = frame_data[ref_idx]["rotation_result"]
        ref_image = _to_hwc_numpy(ref_rot["image"])

        frame_data[ref_idx]["dx"] = 0.0
        frame_data[ref_idx]["dy"] = 0.0
        frame_data[ref_idx]["reg_score"] = 1.0

        for i, fd in enumerate(frame_data):
            if not fd["success"] or i == ref_idx:
                continue
            try:
                target_image = _to_hwc_numpy(fd["rotation_result"]["image"])
                t0 = time.perf_counter()
                dx, dy, score = registration.compute_translation(ref_image, target_image)
                fd["timings"].registration = time.perf_counter() - t0
                fd["dx"] = dx
                fd["dy"] = dy
                fd["reg_score"] = score
            except Exception:
                fd["dx"] = 0.0
                fd["dy"] = 0.0
                fd["reg_score"] = float("nan")
                if verbose:
                    print(f"      Frame {fd['timepoint']}: Registration failed, using identity")

        # Apply translations to images AND markers
        for _i, fd in enumerate(frame_data):
            if not fd["success"]:
                continue
            dx, dy = fd["dx"], fd["dy"]
            if dx == 0.0 and dy == 0.0:
                continue

            rot_result = fd["rotation_result"]
            image = rot_result["image"]

            # Apply translation to image
            translated_image = registration.apply_translation(
                image if isinstance(image, np.ndarray) else _to_hwc_numpy(image),
                -dx,
                -dy,
            )
            rot_result["image"] = translated_image

            # Apply translation to marker positions
            for marker in rot_result["markers"]:
                center = marker["bbox_center"]
                marker["bbox_center"] = (center[0] - dx, center[1] - dy)

    # Phase 3: Masking, Segmentation & Filtering (per-frame)
    if verbose:
        print("    Phase 3: Masking & segmentation...")

    cropped_images = []
    cell_masks = []
    chamber_masks = []

    for fd in frame_data:
        if not fd["success"]:
            continue

        timings = fd["timings"]
        try:
            rot_result = fd["rotation_result"]

            t0 = time.perf_counter()
            mask_result = masking_step(rot_result)
            timings.masking = time.perf_counter() - t0

            cropped_image = mask_result["image"]
            chamber_mask = mask_result["mask"]

            # Ensure HWC format
            if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                cropped_image = np.moveaxis(cropped_image, 0, -1)

            # Segment cells
            t0 = time.perf_counter()
            cropped_rgb = cv2.cvtColor(cropped_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            height, width = cropped_rgb.shape[:2]
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)

            with torch.no_grad():
                seg_result = segmenter(source.to_channel(0))

            masks = seg_result.toMasks(height, width, binary_mask=False)
            labeled_mask = masks[0]

            # Post-segmentation structure filter
            labeled_mask = filter_segmentation_by_mask(labeled_mask, chamber_mask)
            timings.segmentation = time.perf_counter() - t0

            n_cells = int(labeled_mask.max())
            fd["n_cells"] = n_cells
            # Compute per-cell areas (in pixels)
            cell_areas = []
            for cid in range(1, n_cells + 1):
                cell_areas.append(int(np.sum(labeled_mask == cid)))
            fd["cell_areas"] = cell_areas
            cropped_images.append(cropped_image)
            cell_masks.append(labeled_mask)
            chamber_masks.append(chamber_mask)

        except Exception as e:
            fd["success"] = False
            fd["error"] = f"Segmentation: {e}"
            if verbose:
                print(f"      Frame {fd['timepoint']}: Segmentation failed - {e}")

    n_success = sum(1 for f in frame_data if f["success"])
    result.n_success = n_success

    if n_success == 0:
        result.error = "All frames failed during masking/segmentation"
        _save_meta(stack_output_dir, frame_data, n_frames)
        return result

    # Phase 4: Assemble & save stacks
    if verbose:
        print("    Phase 4: Saving stacks...")

    # Determine max dimensions
    shapes = np.array([img.shape for img in cropped_images])
    max_h = np.max(shapes[:, 0])
    max_w = np.max(shapes[:, 1])
    n_channels = cropped_images[0].shape[2] if cropped_images[0].ndim == 3 else 1

    image_stack_list = []
    mask_stack_list = []
    chamber_stack_list = []

    success_idx = 0
    for fd in frame_data:
        if fd["success"]:
            img = cropped_images[success_idx]
            cell_mask = cell_masks[success_idx]
            chamber = chamber_masks[success_idx]

            ph = max_h - img.shape[0]
            pw = max_w - img.shape[1]

            if img.ndim == 3:
                img_padded = np.pad(img, [(0, ph), (0, pw), (0, 0)])
            else:
                img_padded = np.pad(img, [(0, ph), (0, pw)])

            mask_padded = np.pad(cell_mask, [(0, ph), (0, pw)])
            chamber_padded = np.pad(chamber, [(0, ph), (0, pw)])

            image_stack_list.append(img_padded)
            mask_stack_list.append(mask_padded)
            chamber_stack_list.append(chamber_padded)
            success_idx += 1
        else:
            if n_channels > 1:
                image_stack_list.append(np.zeros((max_h, max_w, n_channels), dtype=np.uint8))
            else:
                image_stack_list.append(np.zeros((max_h, max_w), dtype=np.uint8))
            mask_stack_list.append(np.zeros((max_h, max_w), dtype=np.uint16))
            chamber_stack_list.append(np.zeros((max_h, max_w), dtype=np.uint8))

    cell_mask_stack = np.stack(mask_stack_list, axis=0)
    chamber_mask_stack = np.stack(chamber_stack_list, axis=0)

    # Save cell masks
    tifffile.imwrite(
        stack_output_dir / "stack.tif",
        cell_mask_stack.astype(np.uint16),
        compression="zlib",
        compressionargs={"level": 6},
        metadata={"axes": "TYX"},
    )

    # Save chamber masks
    tifffile.imwrite(
        stack_output_dir / "stack_chamber.tif",
        chamber_mask_stack.astype(np.uint8),
        compression="zlib",
        compressionargs={"level": 6},
        metadata={"axes": "TYX"},
    )

    # Save cropped images (optional)
    if save_cropped:
        image_stack = np.stack(image_stack_list, axis=0)
        if image_stack.ndim == 4:
            image_stack_tcyx = np.moveaxis(image_stack, -1, 1)
            axes = "TCYX"
        else:
            image_stack_tcyx = image_stack[:, None, :, :]
            axes = "TCYX"

        tifffile.imwrite(
            stack_output_dir / "stack_cropped.tif",
            image_stack_tcyx.astype(np.uint8),
            compression="zlib",
            compressionargs={"level": 6},
            metadata={"axes": axes},
        )

    # Save metadata CSV
    _save_meta(stack_output_dir, frame_data, n_frames)

    # Save per-cell CSV (timepoint, cell_id, area_px, area_um2)
    _save_cells(stack_output_dir, frame_data, pixel_size)

    # Render video (optional)
    if render_stacks and ACIA_AVAILABLE:
        _render_video(
            stack_output_dir,
            frame_data,
            cropped_images,
            cell_masks,
            chamber_masks,
            pixel_size,
            verbose,
        )

    result.success = True
    result.n_cells_total = sum(fd["n_cells"] for fd in frame_data)
    result.frame_timings = [fd["timings"] for fd in frame_data]

    if verbose:
        print(f"    Done: {n_success}/{n_frames} frames, {result.n_cells_total} cells total")

    return result


def _to_hwc_numpy(image) -> np.ndarray:
    """Convert image to HWC numpy format."""
    if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] <= 4:
            image = np.moveaxis(image, 0, -1)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] <= 4:
            image = np.moveaxis(image, 0, -1)
    return image


def _save_meta(output_dir: Path, frame_data: list[dict], n_frames: int):
    """Save per-frame metadata CSV including pipeline step timings."""
    rows = []
    for fd in frame_data:
        row = {
            "timepoint": fd["timepoint"],
            "success": fd["success"],
            "n_cells": fd["n_cells"],
            "error": fd["error"],
            "dx": fd["dx"],
            "dy": fd["dy"],
            "reg_score": fd["reg_score"],
        }
        row.update(fd["timings"].as_dict())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "meta.csv", index=False)


def _save_cells(output_dir: Path, frame_data: list[dict], pixel_size: float):
    """Save per-cell CSV with timepoint, cell_id, area in pixels and µm²."""
    area_per_pixel = pixel_size**2  # µm² per pixel
    rows = []
    for fd in frame_data:
        if not fd["success"]:
            continue
        for cell_id, area_px in enumerate(fd["cell_areas"], 1):
            rows.append(
                {
                    "timepoint": fd["timepoint"],
                    "cell_id": cell_id,
                    "area_px": area_px,
                    "area_um2": area_px * area_per_pixel,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "cells.csv", index=False)


def _append_fallback_frame(rendered_frames: list[np.ndarray], render_dir: Path, t: int, label: str):
    """Append a black fallback frame with a label so no frames are skipped."""
    if rendered_frames:
        h, w = rendered_frames[0].shape[:2]
    else:
        h, w = 512, 512
    black = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        black,
        f"Frame {t}: {label}",
        (10, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    frame_path = render_dir / f"frame_{t:03d}.png"
    cv2.imwrite(str(frame_path), black)
    rendered_frames.append(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))


def _render_video(
    output_dir: Path,
    frame_data: list[dict],
    cropped_images: list[np.ndarray],
    cell_masks: list[np.ndarray],
    chamber_masks: list[np.ndarray],
    pixel_size: float,
    verbose: bool,
    fps: float = 5.0,
):
    """Render per-frame PNGs and MP4 video."""
    render_dir = output_dir / "rendered"
    render_dir.mkdir(exist_ok=True)

    rendered_frames = []
    success_idx = 0

    for t, fd in enumerate(frame_data):
        if fd["success"]:
            try:
                rendered = render_frame_visualization(
                    cropped_images[success_idx].astype(np.uint8),
                    cell_masks[success_idx],
                    chamber_masks[success_idx],
                    pixel_size,
                )
                annotation = f"Frame {t} | Cells={fd['n_cells']}"
                cv2.putText(
                    rendered,
                    annotation,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frame_path = render_dir / f"frame_{t:03d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                rendered_frames.append(rendered)
            except Exception as e:
                if verbose:
                    print(f"      Warning: Failed to render frame {t}: {e}")
                # Fall back to black frame so no frames are missing
                _append_fallback_frame(rendered_frames, render_dir, t, f"RENDER ERROR: {e}")
            success_idx += 1
        else:
            _append_fallback_frame(rendered_frames, render_dir, t, "FAILED")

    if rendered_frames:
        video_path = output_dir / "timelapse.mp4"
        try:
            # Pad all frames to consistent max dimensions
            max_h = max(f.shape[0] for f in rendered_frames)
            max_w = max(f.shape[1] for f in rendered_frames)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (max_w, max_h))
            for frame_rgb in rendered_frames:
                h, w = frame_rgb.shape[:2]
                if h != max_h or w != max_w:
                    padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                    padded[:h, :w] = frame_rgb
                    frame_rgb = padded
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            writer.release()
            if verbose:
                print(f"      Video saved: {video_path.name} ({len(rendered_frames)} frames)")
        except Exception as e:
            if verbose:
                print(f"      Warning: Failed to create video: {e}")


def resegment_stack(
    stack_output_dir: Path,
    segmenter,
    pixel_size: float,
    render_stacks: bool = False,
    verbose: bool = False,
) -> tuple[int, int, int, list[float]]:
    """Re-run segmentation on existing cropped images and chamber masks.

    Args:
        stack_output_dir: Directory containing stack_cropped.tif and stack_chamber.tif
        segmenter: Segmentation model instance
        pixel_size: Pixel size in µm/px
        render_stacks: Whether to render visualization PNGs + video
        verbose: Print detailed progress

    Returns:
        (n_frames, n_success, n_cells_total, seg_times)
    """
    cropped_path = stack_output_dir / "stack_cropped.tif"
    chamber_path = stack_output_dir / "stack_chamber.tif"

    if not cropped_path.exists() or not chamber_path.exists():
        raise FileNotFoundError(
            f"Missing stack_cropped.tif or stack_chamber.tif in {stack_output_dir}. "
            "Run full pipeline with --save-cropped first."
        )

    cropped_stack = tifffile.imread(str(cropped_path))  # TCYX
    chamber_stack = tifffile.imread(str(chamber_path))  # TYX

    # Convert TCYX -> TYXC
    if cropped_stack.ndim == 4:
        cropped_stack = np.moveaxis(cropped_stack, 1, -1)  # TCYX -> TYXC

    n_frames = cropped_stack.shape[0]
    n_success = 0
    n_cells_total = 0
    seg_times = []

    mask_stack_list = []
    cropped_images = []
    cell_masks = []
    chamber_masks = []
    frame_data = []

    for t in range(n_frames):
        cropped_image = cropped_stack[t]  # HWC or HW
        chamber_mask = chamber_stack[t].astype(bool)  # HW

        fd = {
            "timepoint": t,
            "success": False,
            "n_cells": 0,
            "cell_areas": [],
            "error": "",
        }

        # Skip blank frames (all zero = failed frame from previous run)
        if cropped_image.max() == 0:
            frame_data.append(fd)
            mask_stack_list.append(np.zeros(chamber_mask.shape, dtype=np.uint16))
            continue

        try:
            t0 = time.perf_counter()
            cropped_rgb = cv2.cvtColor(cropped_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            height, width = cropped_rgb.shape[:2]
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)

            with torch.no_grad():
                seg_result = segmenter(source.to_channel(0))

            masks = seg_result.toMasks(height, width, binary_mask=False)
            labeled_mask = masks[0]
            labeled_mask = filter_segmentation_by_mask(labeled_mask, chamber_mask)
            seg_time = time.perf_counter() - t0
            seg_times.append(seg_time)

            n_cells = int(labeled_mask.max())
            fd["success"] = True
            fd["n_cells"] = n_cells
            for cid in range(1, n_cells + 1):
                fd["cell_areas"].append(int(np.sum(labeled_mask == cid)))

            n_success += 1
            n_cells_total += n_cells

            mask_stack_list.append(labeled_mask)
            cropped_images.append(cropped_image)
            cell_masks.append(labeled_mask)
            chamber_masks.append(chamber_mask)

        except Exception as e:
            fd["error"] = f"Segmentation: {e}"
            mask_stack_list.append(np.zeros(chamber_mask.shape, dtype=np.uint16))
            if verbose:
                print(f"      Frame {t}: Segmentation failed - {e}")

        frame_data.append(fd)

    # Save updated cell mask stack
    cell_mask_stack = np.stack(mask_stack_list, axis=0)
    tifffile.imwrite(
        stack_output_dir / "stack.tif",
        cell_mask_stack.astype(np.uint16),
        compression="zlib",
        compressionargs={"level": 6},
        metadata={"axes": "TYX"},
    )

    # Save per-cell CSV
    _save_cells_from_frame_data(stack_output_dir, frame_data, pixel_size)

    # Render video
    if render_stacks and ACIA_AVAILABLE and cropped_images:
        _render_video(
            stack_output_dir,
            frame_data,
            cropped_images,
            cell_masks,
            chamber_masks,
            pixel_size,
            verbose,
        )

    if verbose:
        print(f"    Done: {n_success}/{n_frames} frames, {n_cells_total} cells total")

    return n_frames, n_success, n_cells_total, seg_times


def _save_cells_from_frame_data(output_dir: Path, frame_data: list[dict], pixel_size: float):
    """Save per-cell CSV from frame_data dicts (used by resegment_stack)."""
    area_per_pixel = pixel_size**2
    rows = []
    for fd in frame_data:
        if not fd.get("success"):
            continue
        for cell_id, area_px in enumerate(fd["cell_areas"], 1):
            rows.append(
                {
                    "timepoint": fd["timepoint"],
                    "cell_id": cell_id,
                    "area_px": area_px,
                    "area_um2": area_px * area_per_pixel,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "cells.csv", index=False)


def generate_summary(
    results: list[StackResult], segmenter: str | None = None, enable_registration: bool = True
) -> tuple[str, str]:
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    total_frames = sum(r.n_frames for r in results)
    total_frames_ok = sum(r.n_success for r in results)
    total_cells = sum(r.n_cells_total for r in results)

    lines = [
        "# Folder Processing Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Stacks | {total} |",
        f"| Passed | {passed} ({100 * passed / total:.1f}%) |" if total > 0 else "| Passed | 0 |",
        f"| Failed | {failed} |",
        f"| Total Frames | {total_frames} |",
        f"| Frames OK | {total_frames_ok} |",
        f"| Total Cells | {total_cells} |",
        *([f"| Segmenter | {segmenter} |"] if segmenter else []),
        "",
    ]

    # Per-folder breakdown
    folders = sorted(set(r.folder for r in results))
    lines.extend(
        [
            "## Per-Folder Results",
            "",
            "| Folder | Chamber Type | Stacks | Passed | Frames OK | Cells |",
            "|--------|-------------|--------|--------|-----------|-------|",
        ]
    )
    for folder in folders:
        folder_results = [r for r in results if r.folder == folder]
        n = len(folder_results)
        n_ok = sum(1 for r in folder_results if r.success)
        frames_ok = sum(r.n_success for r in folder_results)
        cells = sum(r.n_cells_total for r in folder_results)
        ct = folder_results[0].chamber_type if folder_results else ""
        lines.append(f"| {folder} | {ct} | {n} | {n_ok} | {frames_ok} | {cells} |")
    lines.append("")

    # Per-chamber-type timing breakdown - fixed display order & numbering
    all_ct = set(r.chamber_type for r in results)
    # Sort: known types in preferred order first, then any unknown types alphabetically
    chamber_types = [ct for ct in _CHAMBER_ORDER if ct in all_ct]
    chamber_types += sorted(all_ct - set(_CHAMBER_ORDER))
    step_names = [
        "detection",
        "matching",
        "rotation",
        "registration",
        "masking",
        "segmentation",
        "total",
    ]
    if not enable_registration:
        step_names = [s for s in step_names if s != "registration"]

    # Collect timings grouped by chamber type
    chamber_timings: dict[str, list[FrameTimings]] = {ct: [] for ct in chamber_types}
    for r in results:
        if r.frame_timings:
            chamber_timings[r.chamber_type].extend(r.frame_timings)

    # Also collect all timings for an "All" row
    all_timings = [ft for timings in chamber_timings.values() for ft in timings]

    if all_timings:
        lines.extend(
            [
                "## Pipeline Step Timings (ms, mean ± std per frame)",
                "",
                "| Chamber Type | N | Detection | Matching | Rotation | "
                + ("Registration | " if enable_registration else "")
                + "Masking | Segmentation | Total | Total w/o Seg | FPS w/ Seg | FPS w/o Seg |",
                "|---|---|---|---|---|"
                + ("---|" if enable_registration else "")
                + "---|---|---|---|---|---|",
            ]
        )

        def _fmt_timing_row(label: str, timings_list: list[FrameTimings]) -> str:
            n = len(timings_list)
            if n == 0:
                return f"| {label} | 0 | " + " | ".join(["-"] * (len(step_names) + 3)) + " |"
            vals = {
                s: np.array([getattr(ft, s) if s != "total" else ft.total for ft in timings_list])
                for s in step_names
            }
            cells = [f"{vals[s].mean()*1000:.1f} ± {vals[s].std()*1000:.1f}" for s in step_names]
            total_wo_seg = vals["total"] - vals["segmentation"]
            fps_w_seg = 1.0 / vals["total"]
            fps_wo_seg = 1.0 / total_wo_seg
            cells.append(f"{total_wo_seg.mean()*1000:.1f} ± {total_wo_seg.std()*1000:.1f}")
            cells.append(f"{fps_w_seg.mean():.1f} ± {fps_w_seg.std():.1f}")
            cells.append(f"{fps_wo_seg.mean():.1f} ± {fps_wo_seg.std():.1f}")
            return f"| {label} | {n} | " + " | ".join(cells) + " |"

        for ct in chamber_types:
            num = _CHAMBER_NUMBERS.get(ct)
            label = f"{ct} ({num})" if num is not None else ct
            lines.append(_fmt_timing_row(label, chamber_timings[ct]))
        if len(chamber_types) > 1:
            lines.append(_fmt_timing_row("**All**", all_timings))
        lines.append("")

    # Failed stacks
    failed_results = [r for r in results if not r.success]
    if failed_results:
        lines.extend(
            [
                "## Failed Stacks",
                "",
                "| Folder | File | Error |",
                "|--------|------|-------|",
            ]
        )
        for r in failed_results:
            error_msg = (r.error or "").replace("|", "\\|")
            lines.append(f"| {r.folder} | {r.file_name} | {error_msg} |")
        lines.append("")

    latex_table = _generate_latex_timing_table(
        chamber_types, chamber_timings, all_timings, step_names
    )
    return "\n".join(lines), latex_table


def _generate_latex_timing_table(
    chamber_types: list[str],
    chamber_timings: dict[str, list[FrameTimings]],
    all_timings: list[FrameTimings],
    step_names: list[str],
) -> str:
    """Generate a LaTeX tabular for the pipeline step timings."""
    if not all_timings:
        return ""

    has_registration = "registration" in step_names
    header_labels = [
        "Chamber Type",
        "N",
        "Det. (ms)",
        "Match (ms)",
        "Rot. (ms)",
        *(["Reg. (ms)"] if has_registration else []),
        "Mask (ms)",
        "Seg. (ms)",
        "Total (ms)",
        "w/o Seg (ms)",
        "FPS w/ Seg",
        "FPS w/o Seg",
    ]

    def _fmt_latex_row(label: str, timings_list: list[FrameTimings]) -> str:
        n = len(timings_list)
        if n == 0:
            return f"  {label} & 0 & " + " & ".join(["-"] * (len(step_names) + 3)) + " \\\\"
        vals = {
            s: np.array([getattr(ft, s) if s != "total" else ft.total for ft in timings_list])
            for s in step_names
        }
        cells = [f"{vals[s].mean()*1000:.1f} $\\pm$ {vals[s].std()*1000:.1f}" for s in step_names]
        total_wo_seg = vals["total"] - vals["segmentation"]
        fps_w_seg = 1.0 / vals["total"]
        fps_wo_seg = 1.0 / total_wo_seg
        cells.append(f"{total_wo_seg.mean()*1000:.1f} $\\pm$ {total_wo_seg.std()*1000:.1f}")
        cells.append(f"{fps_w_seg.mean():.1f} $\\pm$ {fps_w_seg.std():.1f}")
        cells.append(f"{fps_wo_seg.mean():.1f} $\\pm$ {fps_wo_seg.std():.1f}")
        return f"  {label} & {n} & " + " & ".join(cells) + " \\\\"

    tex_lines = [
        "\\begin{tabular}{l r || " + " ".join(["r"] * len(step_names)) + " || r r r r}",
        "  \\toprule",
        "  " + " & ".join(header_labels) + " \\\\",
        "  \\midrule",
    ]

    for ct in chamber_types:
        num = _CHAMBER_NUMBERS.get(ct)
        label = f"{ct} ({num})" if num is not None else ct
        tex_lines.append(_fmt_latex_row(label, chamber_timings[ct]))

    if len(chamber_types) > 1:
        tex_lines.append("  \\midrule")
        tex_lines.append(_fmt_latex_row("\\textbf{All}", all_timings))

    tex_lines.append("  \\bottomrule")
    tex_lines.append("\\end{tabular}")

    return "\n".join(tex_lines)


def _plot_pipeline_timing_chart(
    all_timings: list[FrameTimings], output_dir: Path, enable_registration: bool = True
) -> None:
    """Generate a horizontal bar chart of pipeline step timings with a broken x-axis."""
    if not BROKENAXES_AVAILABLE:
        return

    all_steps = ["detection", "matching", "rotation", "registration", "masking", "segmentation"]
    all_labels = ["Detection", "Matching", "Rotation", "Registration", "Masking", "Segmentation"]
    if not enable_registration:
        filtered = [
            (s, lbl) for s, lbl in zip(all_steps, all_labels, strict=False) if s != "registration"
        ]
        all_steps, all_labels = zip(*filtered, strict=False)
    step_names = list(all_steps)
    step_labels = [f"{i}. {name}" for i, name in enumerate(all_labels, start=1)]

    # Compute mean/std in ms
    means = []
    stds = []
    for s in step_names:
        vals = np.array([getattr(ft, s) for ft in all_timings]) * 1000  # to ms
        means.append(vals.mean())
        stds.append(vals.std())

    means = np.array(means)
    stds = np.array(stds)

    # Waterfall layout: each bar starts where the previous one ends
    starts = np.concatenate([[0], np.cumsum(means[:-1])])

    # Determine break ranges dynamically
    fast_means = means[:-1]
    fast_stds = stds[:-1]

    # Left panel must cover all fast steps (they all fit within 0..fast_total)
    fast_total = float(np.sum(fast_means))
    fast_upper_candidate = fast_total + float(np.max(fast_stds)) * 1.5
    fast_upper = float(np.ceil(fast_upper_candidate / 10) * 10)
    fast_upper = max(fast_upper, 10)

    # Segmentation bar spans starts[-1]..starts[-1]+means[-1]
    seg_bar_start = float(starts[-1])
    seg_bar_end = float(starts[-1] + means[-1])
    seg_visible_end = float(seg_bar_end + stds[-1])

    seg_lower = float(np.floor(seg_bar_start * 0.9 / 100) * 100)
    seg_upper = float(np.ceil(seg_visible_end * 1.05 / 100) * 100)

    # Only use broken axis if segmentation bar end is far beyond fast panel
    use_broken = seg_bar_end > fast_upper * 3

    fig = plt.figure(figsize=(8, 2.8))

    if use_broken:
        bax = brokenaxes(
            xlims=((0, fast_upper), (seg_lower, seg_upper)),
            width_ratios=(1, 1.2),
            hspace=0.05,
            d=0,
            fig=fig,
        )
    else:
        # No break needed, use regular axes
        bax = fig.add_subplot(111)

    y_pos = np.arange(len(step_names))[::-1]  # reversed so detection is at top
    color = "#4C72B0"

    bax.barh(
        y_pos,
        means,
        left=starts,
        xerr=stds,
        height=0.6,
        color=color,
        edgecolor="white",
        capsize=3,
        error_kw={"linewidth": 1.0},
    )

    if use_broken:
        # Set tick labels on the left-most axis
        bax.axs[0].set_yticks(y_pos)
        bax.axs[0].set_yticklabels(step_labels)
        for ax in bax.axs:
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
        # Add explicit x-ticks to left panel
        left_ax = bax.axs[0]
        tick_step = 10 if fast_upper <= 50 else 20
        left_ax.set_xticks(np.arange(0, fast_upper + 1, tick_step))
    else:
        bax.set_yticks(y_pos)
        bax.set_yticklabels(step_labels)
        bax.xaxis.grid(True, alpha=0.3)
        bax.set_axisbelow(True)

    bax.set_xlabel("Time (ms)")

    # Add value annotations on bars
    if use_broken:
        left_ax = bax.axs[0]
        right_ax = bax.axs[1]
        # Fast steps: label to the right of bar end
        for i, (m, s) in enumerate(zip(means[:-1], stds[:-1], strict=False)):
            label_x = starts[i] + m + s + fast_upper * 0.02
            left_ax.text(
                label_x,
                y_pos[i],
                f"{m:.1f} ms",
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        # Segmentation: label inside the bar on the right axis
        seg_idx = len(step_names) - 1
        seg_label_x = seg_lower + (seg_bar_end - seg_lower) * 0.05
        right_ax.text(
            seg_label_x,
            y_pos[seg_idx],
            f"{means[seg_idx]:.1f} ms",
            va="center",
            ha="left",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    else:
        total_end = float(starts[-1] + means[-1])
        for i, (m, s) in enumerate(zip(means, stds, strict=False)):
            bar_end = starts[i] + m
            if m > total_end * 0.3:
                bax.text(
                    starts[i] + m * 0.05,
                    y_pos[i],
                    f"{m:.1f} ms",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            else:
                bax.text(
                    bar_end + s + total_end * 0.01,
                    y_pos[i],
                    f"{m:.1f} ms",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="black",
                )

    # Draw Gantt-style connectors between consecutive bars
    arrow_ax = bax.axs[0] if use_broken else bax
    for i in range(len(step_names) - 1):
        connector_x = float(starts[i] + means[i])  # end of bar i = start of bar i+1
        arrow_ax.annotate(
            "",
            xy=(connector_x, y_pos[i + 1] + 0.35),
            xytext=(connector_x, y_pos[i] - 0.35),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    plt.tight_layout()

    for ext in ["pdf", "png", "svg"]:
        save_kwargs = {"dpi": 200} if ext == "png" else {}
        fig.savefig(output_dir / f"pipeline_timings.{ext}", bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    print(f"Pipeline timing chart: {output_dir / 'pipeline_timings.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Process folder-based experiment with TIFF stacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda:0/cpu (default: auto)"
    )
    parser.add_argument("--save-cropped", action="store_true", help="Save cropped chamber images")
    parser.add_argument(
        "--render-stacks", action="store_true", help="Generate per-frame PNGs + MP4 video"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already-processed stacks"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit files processed (for testing)"
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed progress output")
    parser.add_argument(
        "--segmenter",
        type=str,
        default="cellpose-sam",
        choices=SEGMENTER_CHOICES,
        help="Segmentation method (default: cellpose-sam)",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Re-run only segmentation on existing cropped images (requires prior --save-cropped run)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate summary and LaTeX table from existing results/timings CSVs without re-processing",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(str(args.config))

    # Resolve paths relative to project root (parent of scripts/)
    base_dir = Path(dmc_masking.__file__).parent.parent
    input_dir = base_dir / config["input_dir"]
    output_dir = base_dir / config["output_dir"]
    chip_config_path = base_dir / config.get("chip_config", "artifacts/chips/sak.json")
    model_path = base_dir / config.get("model_path", "artifacts/models/v26_detect_s_imgsz1280.pt")
    pixel_size = config.get("pixel_size", 0.065789)
    enable_registration = config.get("registration", False)
    registration_method = config.get("registration_method", "ncc")
    flip = config.get("flip", False)
    allow_truncation = config.get("allow_truncation", False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths (skip for --summary-only which only needs output_dir)
    if not args.summary_only:
        if not input_dir.exists():
            print(f"Error: input directory not found: {input_dir}")
            sys.exit(1)
        if not chip_config_path.exists():
            print(f"Error: chip config not found: {chip_config_path}")
            sys.exit(1)
        if not model_path.exists():
            print(f"Error: model not found: {model_path}")
            sys.exit(1)

    # --summary-only: regenerate summary from existing CSVs
    if args.summary_only:
        segmenter_name = config.get("segmenter", args.segmenter)
        results_csv = output_dir / "results.csv"
        timings_csv = output_dir / "timings.csv"
        if not results_csv.exists():
            print(f"Error: results.csv not found in {output_dir}")
            sys.exit(1)

        results_df = pd.read_csv(results_csv)
        timings_df = pd.read_csv(timings_csv) if timings_csv.exists() else pd.DataFrame()

        # Reconstruct StackResult objects with FrameTimings
        results: list[StackResult] = []
        for _, row in results_df.iterrows():
            r = StackResult(
                folder=row["folder"],
                file_name=row["file_name"],
                chamber_type=row["chamber_type"],
                n_frames=int(row["n_frames"]),
                n_success=int(row["n_success"]),
                n_cells_total=int(row["n_cells_total"]),
                success=bool(row["success"]),
                error=str(row.get("error", "") or ""),
                output_dir=str(row.get("output_dir", "")),
            )
            if not timings_df.empty:
                ft_rows = timings_df[
                    (timings_df["folder"] == row["folder"])
                    & (timings_df["file_name"] == row["file_name"])
                ]
                for _, ft_row in ft_rows.iterrows():
                    r.frame_timings.append(
                        FrameTimings(
                            detection=ft_row["t_detection"],
                            matching=ft_row["t_matching"],
                            rotation=ft_row["t_rotation"],
                            registration=ft_row["t_registration"],
                            masking=ft_row["t_masking"],
                            segmentation=ft_row["t_segmentation"],
                        )
                    )
            results.append(r)

        summary, latex_table = generate_summary(
            results, segmenter=segmenter_name, enable_registration=enable_registration
        )
        with open(output_dir / "summary.md", "w") as f:
            f.write(summary)
        if latex_table:
            with open(output_dir / "timings_table.tex", "w") as f:
                f.write(latex_table)
            print(f"LaTeX table: {output_dir / 'timings_table.tex'}")
        # Generate timing bar chart
        try:
            all_timings_flat = [ft for r in results for ft in r.frame_timings]
            if all_timings_flat:
                _plot_pipeline_timing_chart(
                    all_timings_flat, output_dir, enable_registration=enable_registration
                )
        except Exception as e:
            print(f"Warning: could not generate timing chart: {e}")
        print(f"\n{summary}")
        print(f"\nSummary: {output_dir / 'summary.md'}")
        sys.exit(0)

    # Collect files
    files = collect_files(config, base_dir)
    if not files:
        print("No TIFF files found in configured folders.")
        sys.exit(1)

    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Found {len(files)} TIFF stacks across {len(config['folders'])} folders")
    print(f"  Chip config: {chip_config_path}")
    print(f"  Model: {model_path}")
    print(f"  Pixel size: {pixel_size}")
    print(f"  Registration: {enable_registration} ({registration_method})")
    if flip:
        print("  Flip: enabled (vertical flip before processing)")
    if allow_truncation:
        print("  Allow truncation: enabled (ROI may extend beyond image bounds)")
    print(f"  Output: {output_dir}")

    # Initialize pipeline components
    print("\nInitializing pipeline...")

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    segmenter_name = config.get("segmenter", args.segmenter)
    print(f"  Segmenter: {segmenter_name}")
    segmenter = create_segmenter(segmenter_name)

    if not args.segmentation_only:
        chip_lib = ChipStructureLibrary.from_file(chip_config_path, pixel_size=pixel_size)
        detection_step = MarkerDetectionStep(str(model_path), device=args.device, verbose=False)

    print("Pipeline initialized.\n")

    # Process each stack
    results = []
    start_time = time.time()

    for i, file_info in enumerate(files, 1):
        folder = file_info["folder"]
        fname = file_info["file_name"]
        chamber = file_info["chamber_type"]
        stack_out_dir = output_dir / folder / fname

        if args.segmentation_only:
            # Re-run segmentation only on existing cropped data
            print(f"[{i}/{len(files)}] {folder}/{fname} ({chamber}) [segmentation-only]")
            result = StackResult(
                folder=folder,
                file_name=fname,
                chamber_type=chamber,
                output_dir=str(stack_out_dir),
            )
            try:
                n_frames, n_success, n_cells, seg_times = resegment_stack(
                    stack_out_dir,
                    segmenter,
                    pixel_size,
                    render_stacks=args.render_stacks,
                    verbose=args.verbose,
                )
                result.n_frames = n_frames
                result.n_success = n_success
                result.n_cells_total = n_cells
                result.success = n_success > 0
                # Store segmentation timings
                for st in seg_times:
                    ft = FrameTimings(segmentation=st)
                    result.frame_timings.append(ft)
                print(f"  -> OK: {n_success}/{n_frames} frames, {n_cells} cells")
            except Exception as e:
                result.error = str(e)
                print(f"  -> FAILED: {e}")
            results.append(result)
            continue

        # Check skip-existing
        if args.skip_existing and (stack_out_dir / "stack.tif").exists():
            print(f"[{i}/{len(files)}] {folder}/{fname} -> SKIPPED (exists)")
            # Load existing meta for summary
            try:
                meta = pd.read_csv(stack_out_dir / "meta.csv")
                r = StackResult(
                    folder=folder,
                    file_name=fname,
                    chamber_type=chamber,
                    n_frames=len(meta),
                    n_success=int(meta["success"].sum()),
                    n_cells_total=int(meta["n_cells"].sum()),
                    success=True,
                    output_dir=str(stack_out_dir),
                )
                results.append(r)
            except Exception:
                pass
            continue

        print(f"[{i}/{len(files)}] {folder}/{fname} ({chamber})")

        result = process_stack(
            file_info=file_info,
            output_dir=output_dir,
            detection_step=detection_step,
            chip_lib=chip_lib,
            segmenter=segmenter,
            pixel_size=pixel_size,
            enable_registration=enable_registration,
            registration_method=registration_method,
            save_cropped=args.save_cropped,
            render_stacks=args.render_stacks,
            verbose=args.verbose,
            device=args.device,
            flip=flip,
            allow_truncation=allow_truncation,
        )
        results.append(result)

        if result.success:
            msg = f"  -> OK: {result.n_success}/{result.n_frames} frames, {result.n_cells_total} cells"
            if result.frame_timings:
                avg_total = np.mean([ft.total for ft in result.frame_timings])
                msg += f", {avg_total:.3f}s/frame avg"
            print(msg)
        else:
            print(f"  -> FAILED: {result.error}")

    elapsed = time.time() - start_time
    print(f"\nProcessed {len(results)} stacks in {elapsed:.1f}s")

    # Save results CSV
    results_df = pd.DataFrame(
        [
            {
                "folder": r.folder,
                "file_name": r.file_name,
                "chamber_type": r.chamber_type,
                "n_frames": r.n_frames,
                "n_success": r.n_success,
                "n_cells_total": r.n_cells_total,
                "success": r.success,
                "error": r.error,
                "output_dir": r.output_dir,
            }
            for r in results
        ]
    )
    results_df.to_csv(output_dir / "results.csv", index=False)

    # Save detailed per-frame timings CSV
    timing_rows = []
    for r in results:
        for ft in r.frame_timings:
            row = {"folder": r.folder, "file_name": r.file_name, "chamber_type": r.chamber_type}
            row.update(ft.as_dict())
            timing_rows.append(row)
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(output_dir / "timings.csv", index=False)
        print(f"Timings: {output_dir / 'timings.csv'}")

    # Save summary
    summary, latex_table = generate_summary(
        results, segmenter=segmenter_name, enable_registration=enable_registration
    )
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)
    if latex_table:
        with open(output_dir / "timings_table.tex", "w") as f:
            f.write(latex_table)
        print(f"LaTeX table: {output_dir / 'timings_table.tex'}")
    # Generate timing bar chart
    try:
        all_timings_flat = [ft for r in results for ft in r.frame_timings]
        if all_timings_flat:
            _plot_pipeline_timing_chart(
                all_timings_flat, output_dir, enable_registration=enable_registration
            )
    except Exception as e:
        print(f"Warning: could not generate timing chart: {e}")
    print(f"\n{summary}")
    print(f"\nResults: {output_dir / 'results.csv'}")
    print(f"Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

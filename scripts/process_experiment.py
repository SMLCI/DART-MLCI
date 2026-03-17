#!/usr/bin/env python
"""Process experiment dataset with full DMC masking and cell segmentation pipeline.

Applies the DMC masking pipeline (marker detection, matching, rotation, ROI masking)
followed by cell segmentation to all images in an experiment dataset.

Example usage:
    python scripts/process_experiment.py --dataset-dir /path/to/experiment --output-dir /path/to/output
    python scripts/process_experiment.py --dataset-dir /path/to/experiment --output-dir ./output --max-images 5 --verbose
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pint
    from acia.segm.local import THWCSequenceSource
    from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter
    from acia.viz import colorize_instance_mask

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

from dart_mlci import (
    ChamberPipelineCache,
    MarkerDetectionStep,
    create_structure_library,
)
from dart_mlci.constants import DEFAULT_MODEL_PATH, DEFAULT_STRUCTURE_LIBRARY_PATH
from dart_mlci.io import load_image
from dart_mlci.visualization import plot_markers_on_image

# Unit registry for scalebar (only if pint is available)
if ACIA_AVAILABLE:
    UNIT_REGISTRY = pint.UnitRegistry()


from dart_mlci.visualization.rendering import add_scalebar, render_cell_visualization


def render_cropped_visualization(
    cropped_image: np.ndarray,
    labeled_mask: np.ndarray,
    chamber_mask: np.ndarray,
    pixel_size: float,
    scalebar_um: float = 10,
    alpha: float = 0.5,
) -> np.ndarray:
    """Render visualization with colored cells, chamber mask, and scalebar."""
    return render_cell_visualization(
        cropped_image=cropped_image,
        labeled_mask=labeled_mask,
        chamber_mask=chamber_mask,
        pixel_size=pixel_size,
        alpha=alpha,
        scalebar=True,
        scalebar_um=scalebar_um,
    )


def render_uncropped_visualization(
    rotated_image: np.ndarray,
    labeled_mask: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    chamber_mask_cropped: np.ndarray,
    pixel_size: float,
    scalebar_um: float = 10,
    alpha: float = 0.5,
) -> np.ndarray:
    """Render uncropped visualization with ROI highlighted and cells overlaid.

    Args:
        rotated_image: HxWxC full rotated image (uint8, RGB)
        labeled_mask: HxW instance mask in cropped coordinates (0=bg, 1..N=cells)
        crop_bbox: (minx, miny, maxx, maxy) bounding box of the crop
        chamber_mask_cropped: HxW binary mask in cropped coordinates (True=outside ROI)
        pixel_size: Pixel size in micrometers
        scalebar_um: Scalebar width in micrometers
        alpha: Cell mask transparency (0-1)

    Returns:
        Rendered visualization image (HxWxC, uint8, RGB)
    """
    minx, miny, maxx, maxy = crop_bbox
    output = rotated_image.copy().astype(np.float32)
    h, w = output.shape[:2]

    # 1. Create a dimmed version outside the ROI region
    # First dim the entire image
    dimmed = output * 0.4

    # 2. Create full-size masks for ROI visualization
    full_mask = np.ones((h, w), dtype=bool)  # True = dimmed
    full_mask[miny:maxy, minx:maxx] = chamber_mask_cropped  # Inside crop: use chamber mask

    # Apply dimming outside ROI
    output[full_mask] = dimmed[full_mask]

    # 3. Colorize and overlay cell masks
    colored_cells = colorize_instance_mask(labeled_mask, seed=42)
    cell_area = labeled_mask > 0

    # Map cell colors to full image coordinates
    crop_region = output[miny:maxy, minx:maxx]
    crop_region[cell_area] = (
        alpha * colored_cells[cell_area].astype(np.float32) + (1 - alpha) * crop_region[cell_area]
    )

    # 4. Draw ROI bounding box outline
    output = output.astype(np.uint8)
    # Draw rectangle (2 pixel thick cyan border)
    cv2.rectangle(output, (minx, miny), (maxx - 1, maxy - 1), (0, 255, 255), 2)

    # 5. Add scalebar
    output = add_scalebar(output, pixel_size, scalebar_um)

    return output


# Pipeline step names for tracking
STEP_LOADING = "Loading"
STEP_STRUCTURE = "Structure"
STEP_DETECTION = "Detection"
STEP_MATCHING = "Matching"
STEP_ROTATION = "Rotation"
STEP_MASKING = "Masking"
STEP_SEGMENTATION = "Segmentation"
STEP_SAVING = "Saving"

# Time-lapse stacking specific steps
STEP_ROTATION_PHASE = "rotation_phase"
STEP_REGISTRATION = "registration"
STEP_SEGMENTATION_PHASE = "segmentation_phase"
STEP_STACKING = "stacking"
STEP_STACK_PROCESSING = "stack_processing"

ALL_STEPS = [
    STEP_LOADING,
    STEP_STRUCTURE,
    STEP_DETECTION,
    STEP_MATCHING,
    STEP_ROTATION,
    STEP_MASKING,
    STEP_SEGMENTATION,
    STEP_SAVING,
    STEP_ROTATION_PHASE,
    STEP_REGISTRATION,
    STEP_SEGMENTATION_PHASE,
    STEP_STACKING,
    STEP_STACK_PROCESSING,
]


@dataclass
class ImageResult:
    """Stores result for a single image processing attempt."""

    image_file: str
    roi_id: str
    success: bool = False
    failed_step: str | None = None
    error_message: str | None = None
    n_cells: int = 0
    structure_name: str | None = None
    output_path: str | None = None
    # Partial results for debug visualization
    image: np.ndarray | None = field(default=None, repr=False)
    markers: list | None = field(default=None, repr=False)
    matched_indices: list | None = field(default=None, repr=False)
    angle: float | None = None


class ExperimentProcessor:
    """Processes experiment images through the full DMC masking + segmentation pipeline."""

    def __init__(
        self,
        model_path: Path,
        structure_library_path: Path,
        pixel_size: float = 0.065789,
        device: str | None = None,
        verbose: bool = False,
        save_cropped: bool = False,
        render_images: bool = False,
        scalebar_um: float = 10.0,
        enable_stacking: bool = False,
        enable_registration: bool = False,
        registration_method: str = "ncc",
        reference_frame: int = 0,
        max_translation: int = 20,
        registration_padding: int = 50,
        max_fails: int = 5,
        render_stacks: bool = False,
        stack_video_fps: float = 5.0,
    ):
        """Initialize the experiment processor.

        Args:
            model_path: Path to the YOLO model weights
            structure_library_path: Path to chamber structure JSON file
            pixel_size: Pixel size in micrometers
            device: Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). None for auto.
            verbose: If True, show detailed progress
            save_cropped: If True, save cropped images alongside masks
            render_images: If True, save rendered visualization images
            scalebar_um: Scalebar width in micrometers for rendered images
            enable_stacking: If True, enable time-lapse stacking mode
            enable_registration: If True, enable translation-based registration
            registration_method: Registration method: 'ncc' or 'phase' (default: 'ncc')
            reference_frame: Reference timepoint index for registration
            max_translation: Maximum translation search range in pixels
            registration_padding: Padding around marker region in pixels
            max_fails: Maximum failed frames before failing stack
            render_stacks: If True, render visualizations and videos for stacks
            stack_video_fps: Frame rate for stack time-lapse videos
        """
        self.pixel_size = pixel_size
        self.device = device
        self.verbose = verbose
        self.save_cropped = save_cropped
        self.render_images = render_images
        self.scalebar_um = scalebar_um
        self.enable_stacking = enable_stacking
        self.enable_registration = enable_registration
        self.registration_method = registration_method
        self.reference_frame = reference_frame
        self.max_translation = max_translation
        self.registration_padding = registration_padding
        self.max_fails = max_fails
        self.render_stacks = render_stacks
        self.stack_video_fps = stack_video_fps

        # Initialize structure library
        self.structure_library = create_structure_library(
            structure_library_path=structure_library_path,
            pixel_size=pixel_size,
        )

        # Initialize detection step (shared across all images)
        self.detection_step = MarkerDetectionStep(model_path, device=device, verbose=verbose)

        # Cache for chamber-specific pipeline components
        self._pipeline_cache = ChamberPipelineCache(self.structure_library)

        # Initialize segmenter
        self.segmenter = None
        if not ACIA_AVAILABLE:
            raise ImportError("acia library not available. Install it with: pip install acia")
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        self.segmenter = CellposeSAMSegmenter()

    def _get_chamber_components(self, structure_name: str) -> dict:
        """Get or create pipeline components for a structure type.

        Args:
            structure_name: Chamber structure name (e.g., "NormaleBox-inner")

        Returns:
            Component dict with pipeline steps
        """
        components = self._pipeline_cache.get(structure_name)

        # Add registration if enabled (only once per structure type)
        if self.enable_registration and "registration" not in components:
            from dart_mlci import PhaseCorrelationRegistration, TimelapseRegistration

            marker_group = components["marker_group"]
            if self.registration_method == "phase":
                components["registration"] = PhaseCorrelationRegistration(
                    marker_group_pixel=marker_group,
                    padding=self.registration_padding,
                )
            else:
                components["registration"] = TimelapseRegistration(
                    marker_group_pixel=marker_group,
                    max_translation=self.max_translation,
                    padding=self.registration_padding,
                    device=self.device,
                )

        return components

    def process_image(
        self,
        image_path: Path,
        roi_id: str,
        output_path: Path,
        cropped_output_path: Path | None = None,
        render_cropped_path: Path | None = None,
        render_uncropped_path: Path | None = None,
    ) -> ImageResult:
        """Process a single image through the pipeline.

        Args:
            image_path: Path to the image file
            roi_id: ROI ID for determining chamber structure
            output_path: Path to save the segmentation mask
            cropped_output_path: Optional path to save cropped image
            render_cropped_path: Optional path to save cropped rendered visualization
            render_uncropped_path: Optional path to save uncropped rendered visualization

        Returns:
            ImageResult with success status and cell count
        """
        result = ImageResult(image_file=str(image_path.name), roi_id=roi_id)

        # Step 1: Loading
        try:
            image = load_image(image_path)
            if image is None or image.size == 0:
                raise ValueError("Image is empty or failed to load")
            result.image = image  # Store for debug visualization
        except Exception as e:
            result.failed_step = STEP_LOADING
            result.error_message = str(e)
            return result

        # Step 2: Get structure from roi_id
        try:
            structure_name, _roi_polygon, _marker_group = self.structure_library(roi_id)
            result.structure_name = structure_name
            components = self._get_chamber_components(structure_name)
        except Exception as e:
            result.failed_step = STEP_STRUCTURE
            result.error_message = str(e)
            return result

        # Step 3: Detection
        try:
            detection_result = self.detection_step(image)
            result.markers = detection_result.get("markers", [])  # Store for debug
        except Exception as e:
            result.failed_step = STEP_DETECTION
            result.error_message = str(e)
            return result

        # Step 4: Matching
        try:
            matching_result = components["matching_step"](detection_result)
            result.matched_indices = matching_result.get("matched_marker_indices", [])
            if not result.matched_indices:
                raise ValueError("No valid marker pairs found")
        except Exception as e:
            result.failed_step = STEP_MATCHING
            result.error_message = str(e)
            return result

        # Step 5: Rotation
        try:
            rotation_result = components["rotation_step"](matching_result)
            result.angle = rotation_result.get("angle", 0.0)  # Store for debug
            # Store rotated image for uncropped rendering (before masking crops it)
            rotated_image_for_render = None
            if render_uncropped_path is not None:
                rotated_img = rotation_result["image"]
                # Ensure HWC format
                if rotated_img.ndim == 3 and rotated_img.shape[0] <= 4:
                    rotated_img = np.moveaxis(rotated_img, 0, -1)
                rotated_image_for_render = rotated_img.copy()
        except Exception as e:
            result.failed_step = STEP_ROTATION
            result.error_message = str(e)
            return result

        # Step 6: Masking
        try:
            # Request bbox when rendering is needed
            need_bbox = render_cropped_path is not None or render_uncropped_path is not None
            masking_result = components["masking_step"](rotation_result, return_bbox=need_bbox)
            cropped_image = masking_result["image"]
            chamber_mask = masking_result["mask"]
            crop_bbox = masking_result.get("crop_bbox")
            # Convert CxHxW to HxWxC if needed
            if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                cropped_image = np.moveaxis(cropped_image, 0, -1)
        except Exception as e:
            result.failed_step = STEP_MASKING
            result.error_message = str(e)
            return result

        # Step 7: Segmentation
        try:
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            height, width = cropped_rgb.shape[:2]
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)  # TxHxWxC
            source = THWCSequenceSource(segm_input)

            with torch.no_grad():
                segmentation_result = self.segmenter(source.to_channel(0))

            # Extract instance-labeled mask (binary_mask=False gives uint16 labels)
            masks = segmentation_result.toMasks(height, width, binary_mask=False)
            labeled_mask = masks[0]  # First (only) frame
            result.n_cells = int(labeled_mask.max())  # Count unique cell IDs
        except Exception as e:
            result.failed_step = STEP_SEGMENTATION
            result.error_message = str(e)
            return result

        # Step 8: Saving
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                output_path,
                labeled_mask.astype(np.uint16),
                compression="zlib",
            )
            result.output_path = str(output_path)

            # Optionally save cropped image
            if cropped_output_path is not None:
                cropped_output_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    cropped_output_path,
                    cropped_image.astype(np.uint8),
                    compression="zlib",
                )

            # Optionally save rendered visualizations
            if render_cropped_path is not None:
                render_cropped_path.parent.mkdir(parents=True, exist_ok=True)
                rendered_cropped = render_cropped_visualization(
                    cropped_image=cropped_rgb,
                    labeled_mask=labeled_mask,
                    chamber_mask=chamber_mask,
                    pixel_size=self.pixel_size,
                    scalebar_um=self.scalebar_um,
                )
                cv2.imwrite(
                    str(render_cropped_path),
                    cv2.cvtColor(rendered_cropped, cv2.COLOR_RGB2BGR),
                )

            if render_uncropped_path is not None and rotated_image_for_render is not None:
                render_uncropped_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert rotated image to RGB for rendering
                rotated_rgb = cv2.cvtColor(
                    rotated_image_for_render.astype(np.uint8), cv2.COLOR_BGR2RGB
                )
                rendered_uncropped = render_uncropped_visualization(
                    rotated_image=rotated_rgb,
                    labeled_mask=labeled_mask,
                    crop_bbox=crop_bbox,
                    chamber_mask_cropped=chamber_mask,
                    pixel_size=self.pixel_size,
                    scalebar_um=self.scalebar_um,
                )
                cv2.imwrite(
                    str(render_uncropped_path),
                    cv2.cvtColor(rendered_uncropped, cv2.COLOR_RGB2BGR),
                )
        except Exception as e:
            result.failed_step = STEP_SAVING
            result.error_message = str(e)
            return result

        # All steps passed
        result.success = True
        return result

    def process_timelapse_stack(
        self,
        timelapse_df: pd.DataFrame,
        roi_id: str,
        roi_output_dir: Path,
    ) -> tuple[ImageResult, pd.DataFrame]:
        """Process a time-lapse stack for a single ROI.

        Args:
            timelapse_df: DataFrame with metadata for all frames in this ROI
            roi_id: ROI identifier
            roi_output_dir: Output directory for this ROI (e.g., output_dir/roi_0000/)

        Returns:
            Tuple of (overall_result, enhanced_metadata_df)
        """
        # Make a copy to avoid modifying input
        meta_df = timelapse_df.copy()
        n_frames = len(meta_df)

        if self.verbose:
            print(f"\nProcessing time-lapse stack for ROI {roi_id} ({n_frames} frames)")

        # Initialize processing status columns
        meta_df["processing_success"] = False
        meta_df["failed_step"] = ""
        meta_df["error_message"] = ""
        meta_df["n_cells"] = 0
        meta_df["registration_dx"] = 0.0
        meta_df["registration_dy"] = 0.0
        meta_df["registration_score"] = float("nan")

        # Overall result for the stack
        overall_result = ImageResult(image_file=f"stack_{roi_id}", roi_id=roi_id)

        # Get structure configuration
        try:
            structure_name, _roi_polygon, _marker_group = self.structure_library(roi_id)
            overall_result.structure_name = structure_name
            components = self._get_chamber_components(structure_name)
        except Exception as e:
            overall_result.failed_step = STEP_STRUCTURE
            overall_result.error_message = str(e)
            return overall_result, meta_df

        # Storage for per-frame results
        frame_results = []

        # Phase 1: Process all frames through rotation (with per-frame error handling)
        if self.verbose:
            print("Phase 1: Loading and rotating frames...")

        for _idx, row in meta_df.iterrows():
            frame_status = {
                "timepoint": row["timepoint"],
                "image_path": row["image_file"],
                "success": False,
                "failed_step": None,
                "error": "",
                "has_rotation_result": False,
                "rotation_result": None,
                "has_segmentation_result": False,
                "n_cells": 0,
                "dx": 0.0,
                "dy": 0.0,
                "reg_score": float("nan"),
            }

            # Initialize variables for exception handling
            detection_result = None
            matched_indices = []

            try:
                # Load image
                image_path = Path(row["image_file"])
                if not image_path.is_absolute():
                    # Assume relative to dataset dir (handled by caller)
                    pass
                image = load_image(image_path)
                if image is None or image.size == 0:
                    raise ValueError("Image is empty or failed to load")

                # Detect markers
                detection_result = self.detection_step(image)
                # Match markers
                matching_result = components["matching_step"](detection_result)
                matched_indices = matching_result.get("matched_marker_indices", [])
                if not matched_indices:
                    raise ValueError("No valid marker pairs found")

                # Rotate image
                rotation_result = components["rotation_step"](matching_result)

                # Store successful rotation result
                frame_status["has_rotation_result"] = True
                frame_status["rotation_result"] = rotation_result

            except Exception as e:
                # Determine which step failed
                if detection_result is None:
                    frame_status["failed_step"] = "marker_detection"
                elif not matched_indices:
                    frame_status["failed_step"] = "marker_matching"
                else:
                    frame_status["failed_step"] = "rotation"
                frame_status["error"] = str(e)
                if self.verbose:
                    print(
                        f"  Frame {frame_status['timepoint']}: Failed at {frame_status['failed_step']}"
                    )

            frame_results.append(frame_status)

        # Check failure threshold after rotation phase
        n_failed_rotation = sum(1 for f in frame_results if not f["has_rotation_result"])
        if n_failed_rotation > self.max_fails:
            overall_result.failed_step = STEP_ROTATION_PHASE
            overall_result.error_message = (
                f"Too many frames failed rotation: {n_failed_rotation} > {self.max_fails}"
            )
            # Update metadata with failure info
            for idx, frame_status in enumerate(frame_results):
                meta_df.loc[idx, "processing_success"] = frame_status["success"]
                meta_df.loc[idx, "failed_step"] = frame_status["failed_step"] or ""
                meta_df.loc[idx, "error_message"] = frame_status["error"]
            # Save metadata even on failure for debugging
            try:
                roi_output_dir.mkdir(parents=True, exist_ok=True)
                meta_df.to_csv(roi_output_dir / "meta.csv", index=False)
            except Exception:
                pass
            return overall_result, meta_df

        # Phase 2: Optional registration (with per-frame error handling)
        if self.enable_registration and "registration" in components:
            if self.verbose:
                print("Phase 2: Computing registration translations...")

            registration = components["registration"]

            # Get reference frame
            ref_idx = self.reference_frame
            if ref_idx >= len(frame_results) or not frame_results[ref_idx]["has_rotation_result"]:
                # Find first successful frame as reference
                ref_idx = next(
                    (i for i, f in enumerate(frame_results) if f["has_rotation_result"]), None
                )
                if ref_idx is None:
                    overall_result.failed_step = STEP_REGISTRATION
                    overall_result.error_message = "No successful frames for registration"
                    return overall_result, meta_df

            ref_rotation_result = frame_results[ref_idx]["rotation_result"]
            ref_image = ref_rotation_result["image"]

            # Convert to numpy HWC if tensor
            if isinstance(ref_image, torch.Tensor):
                ref_image_np = ref_image.cpu().numpy()
                if ref_image_np.ndim == 3 and ref_image_np.shape[0] <= 4:
                    ref_image_np = np.moveaxis(ref_image_np, 0, -1)  # CHW -> HWC
            else:
                ref_image_np = ref_image

            # Set reference frame translation to identity
            frame_results[ref_idx]["dx"] = 0.0
            frame_results[ref_idx]["dy"] = 0.0
            frame_results[ref_idx]["reg_score"] = 1.0

            # Compute translation for each successful frame
            for i, frame_status in enumerate(frame_results):
                if not frame_status["has_rotation_result"]:
                    continue
                if i == ref_idx:
                    continue  # Skip reference frame

                try:
                    target_result = frame_status["rotation_result"]
                    target_image = target_result["image"]

                    # Convert to numpy HWC if tensor
                    if isinstance(target_image, torch.Tensor):
                        target_image_np = target_image.cpu().numpy()
                        if target_image_np.ndim == 3 and target_image_np.shape[0] <= 4:
                            target_image_np = np.moveaxis(target_image_np, 0, -1)
                    else:
                        target_image_np = target_image

                    # Compute translation
                    dx, dy, score = registration.compute_translation(
                        ref_image_np,
                        target_image_np,
                    )

                    frame_status["dx"] = dx
                    frame_status["dy"] = dy
                    frame_status["reg_score"] = score

                except Exception:
                    # Registration failed, use identity translation
                    frame_status["dx"] = 0.0
                    frame_status["dy"] = 0.0
                    frame_status["reg_score"] = float("nan")
                    if self.verbose:
                        print(
                            f"  Frame {frame_status['timepoint']}: Registration failed, using identity"
                        )

        # Phase 3: Segment each frame with optional translation (with per-frame error handling)
        if self.verbose:
            print("Phase 3: Segmenting cells in each frame...")

        cropped_images = []
        cell_masks = []
        chamber_masks = []

        for _i, frame_status in enumerate(frame_results):
            if not frame_status["has_rotation_result"]:
                # Frame failed rotation, will be zero-filled later
                continue

            try:
                rotation_result = frame_status["rotation_result"]

                # Apply translation if registration enabled
                if self.enable_registration and "registration" in components:
                    registration = components["registration"]
                    dx, dy = frame_status["dx"], frame_status["dy"]

                    # Apply translation to rotated image
                    rotated_image = rotation_result["image"]
                    translated_image = registration.apply_translation(
                        rotated_image,
                        -dx,
                        -dy,
                        return_tensor=isinstance(rotated_image, torch.Tensor),
                    )
                    rotation_result["image"] = translated_image

                # Apply masking and cropping
                masking_result = components["masking_step"](rotation_result)

                # Get cropped image and chamber mask
                cropped_image = masking_result["image"]
                chamber_mask = masking_result["mask"]

                # Ensure HWC format for cropped image
                if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                    cropped_image = np.moveaxis(cropped_image, 0, -1)

                # Segment cells (same format as single-image mode)
                cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                height, width = cropped_rgb.shape[:2]
                segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)  # TxHxWxC
                source = THWCSequenceSource(segm_input)

                with torch.no_grad():
                    segmentation_result = self.segmenter(source.to_channel(0))

                # Extract instance-labeled mask (binary_mask=False gives uint16 labels)
                masks = segmentation_result.toMasks(height, width, binary_mask=False)
                labeled_mask = masks[0]  # First (only) frame

                # Store results
                frame_status["has_segmentation_result"] = True
                frame_status["success"] = True
                frame_status["n_cells"] = int(np.max(labeled_mask))

                cropped_images.append(cropped_image)
                cell_masks.append(labeled_mask)
                chamber_masks.append(chamber_mask)

            except Exception as e:
                frame_status["failed_step"] = frame_status["failed_step"] or "segmentation"
                frame_status["error"] = str(e)
                if self.verbose:
                    print(f"  Frame {frame_status['timepoint']}: Segmentation failed - {e}")

        # Check failure threshold after segmentation
        n_failed_total = sum(1 for f in frame_results if not f["success"])
        if n_failed_total > self.max_fails:
            overall_result.failed_step = STEP_SEGMENTATION_PHASE
            overall_result.error_message = (
                f"Too many frames failed: {n_failed_total} > {self.max_fails}"
            )
            # Update metadata
            for idx, frame_status in enumerate(frame_results):
                meta_df.loc[idx, "processing_success"] = frame_status["success"]
                meta_df.loc[idx, "failed_step"] = frame_status["failed_step"] or ""
                meta_df.loc[idx, "error_message"] = frame_status["error"]
                meta_df.loc[idx, "n_cells"] = frame_status["n_cells"]
                meta_df.loc[idx, "registration_dx"] = frame_status["dx"]
                meta_df.loc[idx, "registration_dy"] = frame_status["dy"]
                meta_df.loc[idx, "registration_score"] = frame_status["reg_score"]
            # Save metadata even on failure for debugging
            try:
                roi_output_dir.mkdir(parents=True, exist_ok=True)
                meta_df.to_csv(roi_output_dir / "meta.csv", index=False)
            except Exception:
                pass
            return overall_result, meta_df

        # Phase 4: Homogenize sizes and create stacks (with zero-filling for failed frames)
        if self.verbose:
            print("Phase 4: Creating stacks...")

        successful_frames = [f for f in frame_results if f["has_segmentation_result"]]
        if not successful_frames:
            overall_result.failed_step = STEP_STACKING
            overall_result.error_message = "No successful frames to stack"
            # Update metadata before returning
            for idx, frame_status in enumerate(frame_results):
                meta_df.loc[idx, "processing_success"] = frame_status["success"]
                meta_df.loc[idx, "failed_step"] = frame_status["failed_step"] or ""
                meta_df.loc[idx, "error_message"] = frame_status["error"]
                meta_df.loc[idx, "n_cells"] = frame_status["n_cells"]
                meta_df.loc[idx, "registration_dx"] = frame_status["dx"]
                meta_df.loc[idx, "registration_dy"] = frame_status["dy"]
                meta_df.loc[idx, "registration_score"] = frame_status["reg_score"]
            # Save metadata even on failure for debugging
            try:
                roi_output_dir.mkdir(parents=True, exist_ok=True)
                meta_df.to_csv(roi_output_dir / "meta.csv", index=False)
                if self.verbose:
                    print(f"  Saved failed stack metadata to {roi_output_dir / 'meta.csv'}")
            except Exception:
                pass  # Don't let metadata save failure mask the real error
            return overall_result, meta_df

        # Determine max dimensions from successful frames
        shapes = np.array([img.shape for img in cropped_images])
        max_height = np.max(shapes[:, 0])
        max_width = np.max(shapes[:, 1])
        n_channels = cropped_images[0].shape[2] if cropped_images[0].ndim == 3 else 1

        # Create stacks with zero-filling for failed frames
        image_stack_list = []
        mask_stack_list = []
        chamber_mask_stack_list = []

        success_idx = 0  # Index into successful frame arrays
        for _i, frame_status in enumerate(frame_results):
            if frame_status["has_segmentation_result"]:
                # Successful frame - pad to max dimensions
                img = cropped_images[success_idx]
                cell_mask = cell_masks[success_idx]
                chamber_mask = chamber_masks[success_idx]

                img_h, img_w = img.shape[:2]
                ph = max_height - img_h
                pw = max_width - img_w

                if img.ndim == 3:
                    img_padded = np.pad(img, [(0, ph), (0, pw), (0, 0)], mode="constant")
                else:
                    img_padded = np.pad(img, [(0, ph), (0, pw)], mode="constant")

                cell_mask_padded = np.pad(cell_mask, [(0, ph), (0, pw)], mode="constant")
                chamber_mask_padded = np.pad(chamber_mask, [(0, ph), (0, pw)], mode="constant")

                image_stack_list.append(img_padded)
                mask_stack_list.append(cell_mask_padded)
                chamber_mask_stack_list.append(chamber_mask_padded)

                success_idx += 1
            else:
                # Failed frame - create zero arrays
                if n_channels > 1:
                    zero_img = np.zeros((max_height, max_width, n_channels), dtype=np.uint8)
                else:
                    zero_img = np.zeros((max_height, max_width), dtype=np.uint8)
                zero_cell_mask = np.zeros((max_height, max_width), dtype=np.uint16)
                zero_chamber_mask = np.zeros((max_height, max_width), dtype=np.uint8)

                image_stack_list.append(zero_img)
                mask_stack_list.append(zero_cell_mask)
                chamber_mask_stack_list.append(zero_chamber_mask)

        # Stack into numpy arrays
        image_stack = np.stack(image_stack_list, axis=0)  # TxHxWxC or TxHxW
        cell_mask_stack = np.stack(mask_stack_list, axis=0)  # TxHxW
        chamber_mask_stack = np.stack(chamber_mask_stack_list, axis=0)  # TxHxW

        # Phase 5: Save outputs
        if self.verbose:
            print("Phase 5: Saving stacks...")

        try:
            roi_output_dir.mkdir(parents=True, exist_ok=True)

            # Save cell masks (uint16, compressed)
            tifffile.imwrite(
                roi_output_dir / "stack.tif",
                cell_mask_stack.astype(np.uint16),
                compression="zlib",
                compressionargs={"level": 6},
                metadata={"axes": "TYX"},
            )

            # Save chamber masks (uint8, compressed)
            tifffile.imwrite(
                roi_output_dir / "stack_chamber.tif",
                chamber_mask_stack.astype(np.uint8),
                compression="zlib",
                compressionargs={"level": 6},
                metadata={"axes": "TYX"},
            )

            # Save cropped images (optional, uint8, compressed)
            if self.save_cropped:
                # Convert TxHxWxC to TxCxHxW for TIFF
                if image_stack.ndim == 4:
                    image_stack_tcyx = np.moveaxis(image_stack, -1, 1)
                    axes = "TCYX"
                else:
                    image_stack_tcyx = image_stack[:, None, :, :]  # Add channel dim
                    axes = "TCYX"

                tifffile.imwrite(
                    roi_output_dir / "stack_cropped.tif",
                    image_stack_tcyx.astype(np.uint8),
                    compression="zlib",
                    compressionargs={"level": 6},
                    metadata={"axes": axes},
                )

            # Render stack visualizations and create video (optional)
            if self.render_stacks:
                if self.verbose:
                    print("  Rendering frame visualizations and creating video...")

                render_dir = roi_output_dir / "rendered"
                render_dir.mkdir(exist_ok=True)

                rendered_frames = []
                success_idx = 0

                for t, frame_status in enumerate(frame_results):
                    if frame_status["has_segmentation_result"]:
                        # Get the actual frame data
                        img = cropped_images[success_idx]
                        mask = cell_masks[success_idx]
                        chamber = chamber_masks[success_idx]

                        # Render visualization
                        try:
                            rendered = render_cropped_visualization(
                                cropped_image=img.astype(np.uint8),
                                labeled_mask=mask,
                                chamber_mask=chamber,
                                pixel_size=self.pixel_size,
                                scalebar_um=self.scalebar_um,
                                alpha=0.5,
                            )

                            # Add frame number and timestamp annotation
                            # Try to get timestamp from metadata
                            if "time" in meta_df.columns:
                                timestamp = meta_df.loc[t, "time"]
                            elif "timestamp" in meta_df.columns:
                                timestamp = meta_df.loc[t, "timestamp"]
                            else:
                                timestamp = t
                            annotation_text = (
                                f"Frame {t} | t={timestamp:.2f} | Cells={frame_status['n_cells']}"
                            )
                            cv2.putText(
                                rendered,
                                annotation_text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                            # Save frame
                            frame_path = render_dir / f"frame_{t:03d}.png"
                            cv2.imwrite(str(frame_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                            rendered_frames.append(rendered)

                        except Exception as e:
                            if self.verbose:
                                print(f"    Warning: Failed to render frame {t}: {e}")

                        success_idx += 1
                    else:
                        # Failed frame - create a black frame with error message
                        if len(rendered_frames) > 0:
                            # Use same size as previous frames
                            h, w = rendered_frames[0].shape[:2]
                        else:
                            h, w = 512, 512  # Default size

                        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        error_text = f"Frame {t}: FAILED"
                        cv2.putText(
                            black_frame,
                            error_text,
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        frame_path = render_dir / f"frame_{t:03d}.png"
                        cv2.imwrite(str(frame_path), black_frame)
                        rendered_frames.append(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB))

                # Create video from rendered frames
                if rendered_frames:
                    video_path = roi_output_dir / "timelapse.mp4"
                    try:
                        h, w = rendered_frames[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            fourcc,
                            self.stack_video_fps,
                            (w, h),
                        )

                        for frame_rgb in rendered_frames:
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)

                        video_writer.release()

                        if self.verbose:
                            print(
                                f"    Video saved: {video_path.name} ({len(rendered_frames)} frames @ {self.stack_video_fps} fps)"
                            )

                    except Exception as e:
                        if self.verbose:
                            print(f"    Warning: Failed to create video: {e}")

            # Update metadata DataFrame
            for idx, frame_status in enumerate(frame_results):
                meta_df.loc[idx, "processing_success"] = frame_status["success"]
                meta_df.loc[idx, "failed_step"] = frame_status["failed_step"] or ""
                meta_df.loc[idx, "error_message"] = frame_status["error"]
                meta_df.loc[idx, "n_cells"] = frame_status["n_cells"]
                meta_df.loc[idx, "registration_dx"] = frame_status["dx"]
                meta_df.loc[idx, "registration_dy"] = frame_status["dy"]
                meta_df.loc[idx, "registration_score"] = frame_status["reg_score"]

            # Save enhanced metadata
            meta_df.to_csv(roi_output_dir / "meta.csv", index=False)

            # Mark overall success
            overall_result.success = True
            overall_result.n_cells = int(np.max(cell_mask_stack))

        except Exception as e:
            overall_result.failed_step = STEP_SAVING
            overall_result.error_message = str(e)
            return overall_result, meta_df

        if self.verbose:
            print(f"  Successfully processed {len(successful_frames)}/{n_frames} frames")

        return overall_result, meta_df


def validate_roi_stack_complete(
    roi_output_dir: Path,
    save_cropped: bool = False,
    render_stacks: bool = False,
    verbose: bool = False,
) -> bool:
    """Check if an ROI stack has been fully and correctly processed.

    Args:
        roi_output_dir: Path to ROI output directory (e.g., output/roi_0000/)
        save_cropped: Whether cropped images should be present
        render_stacks: Whether rendered frames should be present
        verbose: Print validation details

    Returns:
        True if ROI is complete and valid, False otherwise
    """
    if not roi_output_dir.exists():
        return False

    # Check required files
    required_files = [
        roi_output_dir / "stack.tif",
        roi_output_dir / "stack_chamber.tif",
        roi_output_dir / "meta.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            if verbose:
                print(f"    Missing required file: {file_path.name}")
            return False

    # Check optional files based on settings
    if save_cropped:
        cropped_stack = roi_output_dir / "stack_cropped.tif"
        if not cropped_stack.exists():
            if verbose:
                print(f"    Missing cropped stack: {cropped_stack.name}")
            return False

    # Validate metadata
    try:
        meta_df = pd.read_csv(roi_output_dir / "meta.csv")

        # Check required columns
        required_cols = ["timepoint", "processing_success", "failed_step", "n_cells"]
        missing_cols = [col for col in required_cols if col not in meta_df.columns]
        if missing_cols:
            if verbose:
                print(f"    Metadata missing columns: {missing_cols}")
            return False

        # Check if at least some frames were successful
        n_success = meta_df["processing_success"].sum()
        if n_success == 0:
            if verbose:
                print("    No successful frames in metadata")
            return False

        # Validate TIFF files are readable
        try:
            cell_stack = tifffile.imread(roi_output_dir / "stack.tif")
            tifffile.imread(roi_output_dir / "stack_chamber.tif")  # verify readable

            # Check dimensions match metadata
            n_frames_meta = len(meta_df)
            n_frames_stack = cell_stack.shape[0] if cell_stack.ndim >= 3 else 1

            if n_frames_stack != n_frames_meta:
                if verbose:
                    print(f"    Frame count mismatch: stack={n_frames_stack}, meta={n_frames_meta}")
                return False

        except Exception as e:
            if verbose:
                print(f"    Failed to read stack files: {e}")
            return False

        return True

    except Exception as e:
        if verbose:
            print(f"    Metadata validation error: {e}")
        return False


def save_debug_visualization(result: ImageResult, output_path: Path) -> None:
    """Save debug visualization for a failed image.

    Args:
        result: ImageResult with partial results
        output_path: Path to save the visualization
    """
    if result.image is None:
        # Loading failed, nothing to visualize
        return

    # Determine what to visualize based on failed step
    markers = result.markers or []
    matched_indices = result.matched_indices or []

    # Build title with failure info
    title = (
        f"{result.image_file}\n"
        f"ROI: {result.roi_id} | Structure: {result.structure_name or 'unknown'}\n"
        f"Failed at: {result.failed_step}\n"
        f"{result.error_message}"
    )

    # Use plot_markers_on_image for visualization
    plot_markers_on_image(
        image=result.image,
        markers=markers,
        matched_indices=matched_indices,
        title=title,
        output_path=output_path,
    )


def load_experiment_metadata(raw_images_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load experiment metadata from parquet or csv file.

    Args:
        raw_images_dir: Path to raw_images directory containing meta.parquet or meta.csv

    Returns:
        Tuple of (DataFrame with experiment metadata, path to metadata file)
    """
    # Try parquet first, then csv
    meta_parquet = raw_images_dir / "meta.parquet"
    meta_csv = raw_images_dir / "meta.csv"

    if meta_parquet.exists():
        df = pd.read_parquet(meta_parquet)
        meta_path = meta_parquet
    elif meta_csv.exists():
        df = pd.read_csv(meta_csv)
        meta_path = meta_csv
    else:
        raise FileNotFoundError(
            f"Metadata file not found. Expected meta.parquet or meta.csv in {raw_images_dir}"
        )

    # Validate required columns
    required_cols = ["image_file", "roi_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {meta_path.name}: {missing}")

    return df, meta_path


def generate_summary(results: list[ImageResult]) -> str:
    """Generate markdown summary of processing results.

    Args:
        results: List of ImageResult objects

    Returns:
        Formatted markdown summary string
    """
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    total_cells = sum(r.n_cells for r in results if r.success)

    # Count failures by step
    step_counts = {step: 0 for step in ALL_STEPS}
    for r in results:
        if not r.success and r.failed_step:
            step_counts[r.failed_step] += 1

    # Build markdown summary
    lines = [
        "# Experiment Processing Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Images | {total} |",
        f"| Passed | {passed} ({100 * passed / total:.1f}%) |" if total > 0 else "| Passed | 0 |",
        f"| Failed | {failed} ({100 * failed / total:.1f}%) |" if total > 0 else "| Failed | 0 |",
        f"| Total Cells | {total_cells} |",
        "",
    ]

    # Add failure breakdown if there are failures
    if failed > 0:
        lines.extend(
            [
                "## Failures by Pipeline Step",
                "",
                "| Step | Errors |",
                "|------|--------|",
            ]
        )
        for step in ALL_STEPS:
            if step_counts[step] > 0:
                lines.append(f"| {step} | {step_counts[step]} |")
        lines.append("")

        lines.extend(
            [
                "## Failed Images",
                "",
                "| Image | ROI ID | Failed Step | Error |",
                "|-------|--------|-------------|-------|",
            ]
        )
        for r in results:
            if not r.success:
                error_msg = (r.error_message or "").replace("|", "\\|")
                lines.append(f"| {r.image_file} | {r.roi_id} | {r.failed_step} | {error_msg} |")
        lines.append("")

    return "\n".join(lines)


def export_results_csv(results: list[ImageResult], output_path: Path) -> None:
    """Export detailed results to CSV file.

    Args:
        results: List of ImageResult objects
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(
        [
            {
                "image_file": r.image_file,
                "roi_id": r.roi_id,
                "structure_name": r.structure_name or "",
                "success": r.success,
                "n_cells": r.n_cells,
                "failed_step": r.failed_step or "",
                "error_message": r.error_message or "",
                "output_path": r.output_path or "",
            }
            for r in results
        ]
    )
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Process experiment dataset with DMC masking and cell segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_experiment.py --dataset-dir /path/to/experiment --output-dir /path/to/output
  python scripts/process_experiment.py --dataset-dir /path/to/experiment --output-dir ./output --max-images 5 --verbose

Dataset structure:
  <dataset-dir>/
      raw_images/           # Subfolder containing metadata and image files
          meta.parquet      # (or meta.csv) Metadata with columns: image_file, roi_id, ...
          image1.tif
          image2.tif
          ...

Output structure:
  <output-dir>/
      image1.tif            # Segmentation mask (uint16, 0=background, 1..N=cell IDs)
      image2.tif
      ...
      results.csv           # Per-image results (success, n_cells, errors)
      summary.md            # Overall statistics (markdown)
      debug_failed/         # (with --debug) Visualizations of failed images
          image_matching.png
          ...
      rendered/             # (with --render-images) Visualization images
          cropped/          # Cropped chamber with colored cells + scalebar
              image1.png
              ...
          uncropped/        # Full rotated image with ROI highlighted + cells
              image1.png
              ...
        """,
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset directory containing raw_images/ subfolder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for segmentation masks (required)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.065789,
        help="Pixel size in micrometers (default: 0.065789)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to YOLO model (default: artifacts/models/v26_detect_s_imgsz1280.pt)",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="Path to chamber structure JSON file (default: artifacts/chamber_structure.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda:0', 'cuda:1', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to process in single-image mode (for testing)",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Limit number of ROIs to process in stacking mode (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--save-cropped",
        action="store_true",
        help="Also save cropped images alongside masks",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug visualizations for failed images",
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help="Save rendered visualization images (PNG) with colored cells and scalebar",
    )
    parser.add_argument(
        "--scalebar-size",
        type=float,
        default=10.0,
        help="Scalebar width in micrometers for rendered images (default: 10)",
    )
    parser.add_argument(
        "--enable-stacking",
        action="store_true",
        help="Enable time-lapse stacking mode (groups images by roi_id and timestamp)",
    )
    parser.add_argument(
        "--enable-registration",
        action="store_true",
        help="Enable translation-only registration to align frames in time-lapse stacks",
    )
    parser.add_argument(
        "--registration-method",
        choices=["ncc", "phase"],
        default="ncc",
        help="Registration method: 'ncc' (normalized cross-correlation) or 'phase' (phase correlation). Default: ncc",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Reference timepoint index for registration (default: 0)",
    )
    parser.add_argument(
        "--max-translation",
        type=int,
        default=20,
        help="Maximum translation search range in pixels (default: 20)",
    )
    parser.add_argument(
        "--registration-padding",
        type=int,
        default=50,
        help="Padding around marker region for registration in pixels (default: 50)",
    )
    parser.add_argument(
        "--max-fails",
        type=int,
        default=5,
        help="Maximum number of failed frames before failing entire stack (default: 5)",
    )
    parser.add_argument(
        "--render-stacks",
        action="store_true",
        help="Render visualizations for each frame in stacks and create time-lapse videos",
    )
    parser.add_argument(
        "--stack-video-fps",
        type=float,
        default=5.0,
        help="Frame rate for stack time-lapse videos (default: 5.0 fps)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ROIs that have already been processed successfully",
    )

    args = parser.parse_args()

    # Validate dataset directory
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    # Set default paths
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH

    if args.structure_library is None:
        args.structure_library = DEFAULT_STRUCTURE_LIBRARY_PATH

    # Validate paths
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    if not args.structure_library.exists():
        raise FileNotFoundError(f"Structure library not found: {args.structure_library}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create cropped output directory if needed
    cropped_dir = None
    if args.save_cropped:
        cropped_dir = args.output_dir / "cropped_images"
        cropped_dir.mkdir(parents=True, exist_ok=True)

    # Create debug output directory if needed
    debug_dir = None
    if args.debug:
        debug_dir = args.output_dir / "debug_failed"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Create render output directories if needed
    render_cropped_dir = None
    render_uncropped_dir = None
    if args.render_images:
        render_cropped_dir = args.output_dir / "rendered" / "cropped"
        render_uncropped_dir = args.output_dir / "rendered" / "uncropped"
        render_cropped_dir.mkdir(parents=True, exist_ok=True)
        render_uncropped_dir.mkdir(parents=True, exist_ok=True)

    # Determine raw_images directory
    raw_images_dir = args.dataset_dir / "raw_images"
    if not raw_images_dir.exists():
        # Maybe dataset_dir itself is the raw_images folder
        raw_images_dir = args.dataset_dir

    # Load metadata (from raw_images directory)
    print(f"Loading metadata from {raw_images_dir}...")
    df, meta_path = load_experiment_metadata(raw_images_dir)
    print(f"Found {len(df)} images in dataset (from {meta_path.name})")

    # Limit images if requested
    if args.max_images is not None:
        df = df.head(args.max_images)
        print(f"Processing first {len(df)} images (--max-images)")

    # Images are in the same directory as the metadata file
    image_base_dir = raw_images_dir

    # Initialize processor
    print("\nInitializing processor...")
    print(f"  Model: {args.model_path}")
    print(f"  Structure library: {args.structure_library}")
    print(f"  Pixel size: {args.pixel_size}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Output: {args.output_dir}")

    processor = ExperimentProcessor(
        model_path=args.model_path,
        structure_library_path=args.structure_library,
        pixel_size=args.pixel_size,
        device=args.device,
        verbose=args.verbose,
        save_cropped=args.save_cropped,
        render_images=args.render_images,
        scalebar_um=args.scalebar_size,
        enable_stacking=args.enable_stacking,
        enable_registration=args.enable_registration,
        registration_method=args.registration_method,
        reference_frame=args.reference_frame,
        max_translation=args.max_translation,
        registration_padding=args.registration_padding,
        max_fails=args.max_fails,
        render_stacks=args.render_stacks,
        stack_video_fps=args.stack_video_fps,
    )

    # Copy input metadata to output directory for reference
    output_meta_path = args.output_dir / "meta.csv"
    df.to_csv(output_meta_path, index=False)
    print(f"Input metadata copied to: {output_meta_path}")

    results = []

    # Check if stacking mode is enabled
    if args.enable_stacking:
        # Time-lapse stacking mode
        print("\n=== TIME-LAPSE STACKING MODE ===")

        # Check for time/timestamp column
        time_col = None
        if "time" in df.columns:
            time_col = "time"
        elif "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            raise ValueError(
                "Stacking mode requires 'time' or 'timestamp' column in metadata. "
                f"Available columns: {list(df.columns)}"
            )

        print(f"Using '{time_col}' column for temporal ordering")

        # Make image paths absolute if they're relative
        if "image_file" in df.columns:
            df["image_file"] = df["image_file"].apply(
                lambda x: str(image_base_dir / x) if not Path(x).is_absolute() else x
            )

        # Group by roi_id and sort by timestamp
        print(f"\nGrouping by roi_id and sorting by {time_col}...")
        grouped = df.groupby("roi_id")
        n_rois_total = len(grouped)
        print(f"Found {n_rois_total} unique ROI(s)")

        # Limit ROIs if requested
        if args.max_rois is not None and args.max_rois < n_rois_total:
            # Convert to list to enable slicing
            grouped_list = list(grouped)[: args.max_rois]
            n_rois = args.max_rois
            print(f"Processing first {n_rois} ROI(s) (--max-rois)")
        else:
            grouped_list = list(grouped)
            n_rois = n_rois_total

        # Process each ROI stack
        for roi_idx, (roi_id, roi_df) in enumerate(grouped_list, 1):
            roi_id_str = str(roi_id).zfill(4)  # Pad with zeros

            # Sort by timestamp and assign timepoint indices
            roi_df = roi_df.sort_values(time_col).reset_index(drop=True)
            roi_df["timepoint"] = range(len(roi_df))

            print(f"\n[{roi_idx}/{n_rois}] Processing ROI {roi_id_str} ({len(roi_df)} frames)...")

            # Create ROI-specific output directory
            roi_output_dir = args.output_dir / f"roi_{roi_id_str}"

            # Check if ROI should be skipped
            if args.skip_existing and validate_roi_stack_complete(
                roi_output_dir,
                save_cropped=args.save_cropped,
                render_stacks=args.render_stacks,
                verbose=args.verbose,
            ):
                print("  -> SKIPPED (already processed successfully)")
                # Load existing result for summary
                try:
                    meta_df_existing = pd.read_csv(roi_output_dir / "meta.csv")
                    n_success = meta_df_existing["processing_success"].sum()
                    n_total = len(meta_df_existing)
                    n_cells = meta_df_existing["n_cells"].max()

                    result = ImageResult(
                        image_file=f"stack_{roi_id_str}",
                        roi_id=roi_id_str,
                    )
                    result.success = True
                    result.n_cells = int(n_cells) if not pd.isna(n_cells) else 0
                    result.output_path = str(roi_output_dir / "stack.tif")
                    results.append(result)

                    if args.verbose:
                        print(
                            f"  Loaded existing result: {n_success}/{n_total} frames, {result.n_cells} cells"
                        )
                except Exception as e:
                    if args.verbose:
                        print(f"  Warning: Failed to load existing metadata: {e}")
                continue

            roi_output_dir.mkdir(parents=True, exist_ok=True)

            # Process timelapse stack
            try:
                result, enhanced_df = processor.process_timelapse_stack(
                    roi_df.copy(),
                    roi_id_str,
                    roi_output_dir,
                )

                # Store result
                result.output_path = str(roi_output_dir / "stack.tif")
                results.append(result)

                # Report status
                if result.success:
                    n_success = enhanced_df["processing_success"].sum()
                    n_total = len(enhanced_df)
                    print(
                        f"  -> SUCCESS: {n_success}/{n_total} frames processed ({result.n_cells} cells)"
                    )
                else:
                    print(f"  -> FAILED at {result.failed_step}: {result.error_message}")

            except Exception as e:
                # Unexpected error during stack processing
                import traceback

                error_result = ImageResult(
                    image_file=f"stack_{roi_id_str}",
                    roi_id=roi_id_str,
                )
                error_result.failed_step = STEP_STACK_PROCESSING
                error_result.error_message = str(e)
                results.append(error_result)
                print(f"  -> ERROR: {e}")
                if args.verbose:
                    traceback.print_exc()

    else:
        # Single-image processing mode (original behavior)
        print(f"\nProcessing {len(df)} images...")

        for i, row in enumerate(df.itertuples(), 1):
            image_file = row.image_file
            roi_id = str(row.roi_id)

            # Construct paths
            image_path = image_base_dir / image_file
            output_name = Path(image_file).stem + ".tif"
            output_path = args.output_dir / output_name

            cropped_output_path = None
            if cropped_dir is not None:
                cropped_output_path = cropped_dir / output_name

            # Render output paths (PNG format)
            render_cropped_path = None
            render_uncropped_path = None
            if render_cropped_dir is not None:
                render_output_name = Path(image_file).stem + ".png"
                render_cropped_path = render_cropped_dir / render_output_name
            if render_uncropped_dir is not None:
                render_output_name = Path(image_file).stem + ".png"
                render_uncropped_path = render_uncropped_dir / render_output_name

            # Progress output
            if args.verbose:
                print(f"[{i}/{len(df)}] Processing {image_file} (roi_id={roi_id})...")
            else:
                print(f"[{i}/{len(df)}] {image_file}", end=" ")

            # Process image
            result = processor.process_image(
                image_path=image_path,
                roi_id=roi_id,
                output_path=output_path,
                cropped_output_path=cropped_output_path,
                render_cropped_path=render_cropped_path,
                render_uncropped_path=render_uncropped_path,
            )
            results.append(result)

            # Status output
            if result.success:
                if args.verbose:
                    print(f"  -> PASSED ({result.n_cells} cells)")
                else:
                    print(f"-> OK ({result.n_cells} cells)")
            else:
                if args.verbose:
                    print(f"  -> FAILED at {result.failed_step}: {result.error_message}")
                else:
                    print(f"-> FAILED [{result.failed_step}]")

    # Save debug visualizations for failed images
    if debug_dir is not None:
        failed_results = [r for r in results if not r.success]
        if failed_results:
            print(f"\nSaving debug visualizations for {len(failed_results)} failed images...")
            for r in failed_results:
                if r.image is None:
                    if args.verbose:
                        print(f"  Skipped (no image loaded): {r.image_file}")
                    continue
                stem = Path(r.image_file).stem
                debug_path = debug_dir / f"{stem}_{r.failed_step.lower()}.png"
                save_debug_visualization(r, debug_path)
                if args.verbose:
                    print(f"  Saved: {debug_path.name}")
            print(f"Debug images saved to: {debug_dir}")

    # Generate and print summary
    summary = generate_summary(results)
    print(f"\n{summary}")

    # Save summary to file
    summary_path = args.output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")

    # Export detailed results CSV
    results_csv_path = args.output_dir / "results.csv"
    export_results_csv(results, results_csv_path)
    print(f"Detailed results saved to: {results_csv_path}")


if __name__ == "__main__":
    main()

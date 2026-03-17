"""Time-lapse TIFF stack processing through the masking + segmentation pipeline.

Provides ``TimelapseProcessor`` which orchestrates the 4-phase pipeline:
1. Detection + rotation per frame
2. Optional registration (find reference frame, compute translations, apply)
3. Masking + segmentation + filtering per frame
4. Stack assembly with padding

Also provides ``create_segmenter()`` factory for creating segmentation backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from dart_mlci.mask import filter_segmentation_by_mask
from dart_mlci.pipeline import (
    ChamberPipelineCache,
    MarkerDetectionStep,
)
from dart_mlci.types import PipelineTimings
from dart_mlci.utils import normalize_image, to_hwc_numpy


@dataclass
class TimelapseResult:
    """Result of processing a time-lapse stack."""

    n_frames: int = 0
    n_success: int = 0
    n_cells_total: int = 0
    success: bool = False
    error: str = ""
    frame_timings: list[PipelineTimings] = field(default_factory=list)
    # Stack arrays (set after successful processing)
    cell_mask_stack: np.ndarray | None = None
    chamber_mask_stack: np.ndarray | None = None
    cropped_image_stack: np.ndarray | None = None
    # Per-frame metadata
    frame_data: list[dict] = field(default_factory=list)


def create_segmenter(name: str):
    """Create a segmenter instance by name.

    Args:
        name: Segmenter name. One of "cellpose-sam" or "omnipose".

    Returns:
        Segmenter instance.

    Raises:
        ImportError: If the required library is not installed.
        ValueError: If the name is not recognized.
    """
    if name == "cellpose-sam":
        try:
            from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter
        except ImportError as e:
            raise ImportError(
                "acia library not available. Install it with: pip install acia"
            ) from e
        return CellposeSAMSegmenter()
    elif name == "omnipose":
        try:
            from acia.segm.processor.omnipose import OmniposeSegmenter
        except ImportError as e:
            raise ImportError(
                "omnipose not available. Install it with: pip install cellpose_omni"
            ) from e
        return OmniposeSegmenter()
    else:
        raise ValueError(f"Unknown segmenter '{name}'. Choose from: cellpose-sam, omnipose")


class TimelapseProcessor:
    """Process time-lapse TIFF stacks through the full masking + segmentation pipeline.

    Usage:
        >>> processor = TimelapseProcessor(detection_step, pipeline_cache, segmenter=seg)
        >>> result = processor.process_stack(frames, "NormaleBox-inner", pixel_size=0.065789)
    """

    def __init__(
        self,
        detection_step: MarkerDetectionStep,
        pipeline_cache: ChamberPipelineCache,
        registration=None,
        segmenter=None,
        filter_threshold: float = 0.5,
    ):
        """Initialize the processor.

        Args:
            detection_step: Shared marker detection step.
            pipeline_cache: Cache for chamber-specific pipeline components.
            registration: Optional registration instance (PhaseCorrelationRegistration
                or TimelapseRegistration).
            segmenter: Optional segmenter (CellposeSAMSegmenter or OmniposeSegmenter).
            filter_threshold: Threshold for segmentation mask filtering.
        """
        self.detection_step = detection_step
        self.pipeline_cache = pipeline_cache
        self.registration = registration
        self.segmenter = segmenter
        self.filter_threshold = filter_threshold

    def process_stack(
        self,
        frames: np.ndarray,
        structure_name: str,
        pixel_size: float = 0.065789,
        normalize: bool = True,
        flip: bool = False,
        verbose: bool = False,
    ) -> TimelapseResult:
        """Process all frames through the 4-phase pipeline.

        Args:
            frames: TxHxW or TxHxWxC array of frames.
            structure_name: Chamber type name for pipeline lookup.
            pixel_size: Pixel size in microns (for cell area computation).
            normalize: Whether to normalize each frame to uint8.
            flip: Whether to flip frames vertically.
            verbose: Print progress information.

        Returns:
            TimelapseResult with stacks and per-frame metadata.
        """
        result = TimelapseResult()

        if frames.ndim == 2:
            frames = frames[None, :, :]  # Single frame -> (1, H, W)

        n_frames = frames.shape[0]
        result.n_frames = n_frames

        # Get pipeline components
        try:
            components = self.pipeline_cache.get(structure_name)
        except KeyError as e:
            result.error = str(e)
            return result

        matching_step = components["matching_step"]
        rotation_step = components["rotation_step"]
        masking_step = components["masking_step"]

        # Phase 1: Detection & Rotation (per-frame)
        if verbose:
            print("    Phase 1: Detection & rotation...")

        frame_data = []
        for t in range(n_frames):
            frame_raw = frames[t]
            if flip:
                frame_raw = frame_raw[::-1]

            # Prepare frame
            if normalize and frame_raw.dtype != np.uint8:
                frame_uint8 = normalize_image(frame_raw)
            else:
                frame_uint8 = (
                    frame_raw.astype(np.uint8) if frame_raw.dtype != np.uint8 else frame_raw
                )

            # Ensure RGB
            if frame_uint8.ndim == 2:
                frame_rgb = np.stack((frame_uint8,) * 3, axis=-1)
            else:
                frame_rgb = frame_uint8

            timings = PipelineTimings()
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
                det_result = self.detection_step(frame_rgb)
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
            result.frame_data = frame_data
            return result

        # Phase 2: Optional Registration
        if self.registration is not None and n_rotation_ok > 1:
            if verbose:
                print("    Phase 2: Registration...")
            self._apply_registration(frame_data, verbose)

        # Phase 3: Masking, Segmentation & Filtering
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

                # Ensure HWC
                cropped_image = to_hwc_numpy(cropped_image)

                # Segment cells
                if self.segmenter is not None:
                    t0 = time.perf_counter()
                    labeled_mask = self._segment_frame(cropped_image)
                    labeled_mask = filter_segmentation_by_mask(
                        labeled_mask, chamber_mask, threshold=self.filter_threshold
                    )
                    timings.segmentation = time.perf_counter() - t0
                else:
                    labeled_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint16)

                n_cells = int(labeled_mask.max())
                fd["n_cells"] = n_cells
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
        result.frame_data = frame_data
        result.frame_timings = [fd["timings"] for fd in frame_data]

        if n_success == 0:
            result.error = "All frames failed during masking/segmentation"
            return result

        # Phase 4: Assemble stacks
        if verbose:
            print("    Phase 4: Assembling stacks...")

        stacks = self._assemble_stacks(frame_data, cropped_images, cell_masks, chamber_masks)
        result.cell_mask_stack = stacks["cell_mask_stack"]
        result.chamber_mask_stack = stacks["chamber_mask_stack"]
        result.cropped_image_stack = stacks["image_stack"]

        result.success = True
        result.n_cells_total = sum(fd["n_cells"] for fd in frame_data)

        if verbose:
            print(f"    Done: {n_success}/{n_frames} frames, {result.n_cells_total} cells total")

        return result

    def _apply_registration(self, frame_data: list[dict], verbose: bool = False) -> None:
        """Apply registration to frame data in-place."""
        ref_idx = next(i for i, f in enumerate(frame_data) if f["success"])
        ref_rot = frame_data[ref_idx]["rotation_result"]
        ref_image = to_hwc_numpy(ref_rot["image"])

        frame_data[ref_idx]["dx"] = 0.0
        frame_data[ref_idx]["dy"] = 0.0
        frame_data[ref_idx]["reg_score"] = 1.0

        for i, fd in enumerate(frame_data):
            if not fd["success"] or i == ref_idx:
                continue
            try:
                target_image = to_hwc_numpy(fd["rotation_result"]["image"])
                t0 = time.perf_counter()
                dx, dy, score = self.registration.compute_translation(ref_image, target_image)
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

        # Apply translations
        for fd in frame_data:
            if not fd["success"]:
                continue
            dx, dy = fd["dx"], fd["dy"]
            if dx == 0.0 and dy == 0.0:
                continue

            rot_result = fd["rotation_result"]
            image = rot_result["image"]

            translated_image = self.registration.apply_translation(
                image if isinstance(image, np.ndarray) else to_hwc_numpy(image),
                -dx,
                -dy,
            )
            rot_result["image"] = translated_image

            for marker in rot_result["markers"]:
                center = marker["bbox_center"]
                marker["bbox_center"] = (center[0] - dx, center[1] - dy)

    def _segment_frame(self, cropped_image: np.ndarray) -> np.ndarray:
        """Segment a single frame. Requires acia."""
        import torch
        from acia.segm.local import THWCSequenceSource

        cropped_rgb = cv2.cvtColor(cropped_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        height, width = cropped_rgb.shape[:2]
        segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
        source = THWCSequenceSource(segm_input)

        with torch.no_grad():
            seg_result = self.segmenter(source.to_channel(0))

        masks = seg_result.toMasks(height, width, binary_mask=False)
        return masks[0]

    @staticmethod
    def _assemble_stacks(
        frame_data: list[dict],
        cropped_images: list[np.ndarray],
        cell_masks: list[np.ndarray],
        chamber_masks: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Assemble output stacks with padding for failed frames."""
        shapes = np.array([img.shape for img in cropped_images])
        max_h = int(np.max(shapes[:, 0]))
        max_w = int(np.max(shapes[:, 1]))
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

        return {
            "image_stack": np.stack(image_stack_list, axis=0),
            "cell_mask_stack": np.stack(mask_stack_list, axis=0),
            "chamber_mask_stack": np.stack(chamber_stack_list, axis=0),
        }

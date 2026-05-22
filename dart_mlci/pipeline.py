"""Step-based pipeline classes for marker detection, matching, rotation, and masking."""

from __future__ import annotations

import numpy as np
import torch

from dart_mlci.artifacts import ensure_artifact
from dart_mlci.constants import DEFAULT_MARKER_TOLERANCE_PX
from dart_mlci.detection import MarkerDetectionModel
from dart_mlci.mask import apply_mask
from dart_mlci.match import match_markers
from dart_mlci.rotation import compute_marker_group_angles, rotate_image_and_markers


class MarkerDetectionStep:
    """Detect markers in an image."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        verbose: bool = False,
        use_gpu_tensor: bool = False,
        conf_threshold: float = 0.5,
    ):
        """Initialize detection step.

        Args:
            model_path: Path to YOLO model weights. If None, uses DEFAULT_MODEL_PATH.
            device: Device to use for inference.
            verbose: Show YOLO inference output.
            use_gpu_tensor: Keep image on GPU for downstream steps to avoid redundant
                           transfers. Set True for performance, False for compatibility
                           with code expecting numpy arrays. (default: False)
            conf_threshold: Minimum confidence for detected markers (default: 0.5).
        """
        if model_path is None:
            model_path = ensure_artifact("models/v26_detect_s_imgsz1280.pt")
        self.mdm = MarkerDetectionModel(model_path, device=device, verbose=verbose)
        self.conf_threshold = conf_threshold
        self.use_gpu_tensor = use_gpu_tensor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image):
        # YOLO requires numpy input for proper preprocessing (resize, normalize, etc.)
        # After detection, convert to GPU tensor for downstream steps (rotation, etc.)
        markers = self.mdm.predict_markers(image)
        markers = [m for m in markers if m.get("conf", 0.0) >= self.conf_threshold]

        # Convert to GPU tensor for downstream steps to avoid redundant transfers
        # YOLO has already done its work; now we keep on GPU through rotation
        if self.use_gpu_tensor and self.device != "cpu" and torch.cuda.is_available():
            # Input image is HWC (height, width, channels)
            image_tensor = torch.from_numpy(image).float().to(self.device)
            # Convert HWC -> CHW for downstream pipeline (rotation expects CHW)
            if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3, 4):
                image_tensor = image_tensor.permute(2, 0, 1)
            data = {"image": image_tensor, "markers": markers}
        else:
            data = {"image": image, "markers": markers}

        return data


class MarkerMatchingStep:
    """Match markers into pairs."""

    def __init__(self, marker_group_pixel, tolerance=60, max_angle_deviation=5.0):
        self.marker_group_pixel = marker_group_pixel
        self.tolerance = tolerance
        self.max_angle_deviation = max_angle_deviation

    def __call__(self, data):
        markers = data["markers"]

        matched_marker_indices = match_markers(
            markers, marker_group=self.marker_group_pixel, tolerance=self.tolerance
        )

        data["matched_marker_indices"] = matched_marker_indices

        angles = compute_marker_group_angles(
            markers, matched_marker_indices, self.marker_group_pixel
        )

        if len(angles) >= 2:
            angle_range = max(angles) - min(angles)
            if angle_range > self.max_angle_deviation:
                raise ValueError(
                    f"Inconsistent rotation angles: range={angle_range:.2f}° exceeds {self.max_angle_deviation:.1f}°"
                )

        mean_angle = np.mean(angles)

        data["angle"] = mean_angle

        return data


class ImageRotationStep:
    """Rotate images and markers."""

    def __init__(self, use_gpu: bool = True):
        """Initialize rotation step.

        Args:
            use_gpu: Use GPU-accelerated kornia if available (default: True)
        """
        self.use_gpu = use_gpu

    def __call__(self, data):
        markers = data["markers"]
        mean_angle = data["angle"]
        image = data["image"]

        # Check if input is tensor (GPU) or numpy (CPU)
        is_tensor = isinstance(image, torch.Tensor)

        # Rotation functions expect CHW format
        # - Tensor path: image is already CHW from detection step
        # - Numpy path: image is HWC, need to convert to CHW
        if not is_tensor:
            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])  # HWC -> CHW

        rotated_image, rotated_markers = rotate_image_and_markers(
            image, markers, mean_angle, use_gpu=self.use_gpu, return_tensor=is_tensor
        )

        # Convert output back to original format
        # - Tensor path: keep as CHW for downstream steps
        # - Numpy path: convert back to HWC
        if not is_tensor:
            rotated_image = np.moveaxis(rotated_image, [0, 1, 2], [2, 0, 1])  # CHW -> HWC

        data["image"] = rotated_image
        data["markers"] = rotated_markers

        return data


class RoIMaskingStep:
    """Apply RoI mask to a rotated image."""

    def __init__(self, marker_group_pixels, roi_polygon, allow_truncation=False):
        super().__init__()

        self.marker_group_pixels = marker_group_pixels
        self.roi_polygon = roi_polygon
        self.allow_truncation = allow_truncation

    def __call__(self, data, cropped=True, return_bbox=False):
        image = data["image"]

        # Convert to numpy in CHW format for apply_mask
        # apply_mask expects CHW format (uses shape[-2:] for height, width)
        if isinstance(image, torch.Tensor):
            # Tensor path: already CHW, just convert to numpy
            image = image.cpu().numpy()
        else:
            # Numpy path: HWC from rotation, convert to CHW
            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])  # HWC -> CHW

        mask_result = apply_mask(
            matched_marker_indices=data["matched_marker_indices"],
            rotated_markers=data["markers"],
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.roi_polygon,
            rotated_image=image,
            return_uncropped=not cropped,
            return_bbox=return_bbox,
            allow_truncation=self.allow_truncation,
        )

        if return_bbox:
            cropped_image, cropped_mask, bbox = mask_result
            data["crop_bbox"] = bbox
        else:
            cropped_image, cropped_mask = mask_result

        cropped_image = np.moveaxis(cropped_image, [0, 1, 2], [2, 0, 1])

        data["image"] = cropped_image
        data["mask"] = cropped_mask

        return data


class ChamberPipelineCache:
    """Cache for chamber-specific pipeline components.

    Lazily creates and caches MarkerMatchingStep, ImageRotationStep, and
    RoIMaskingStep for each chamber type so they can be reused across frames.
    """

    def __init__(
        self,
        structure_library,
        tolerance: int = DEFAULT_MARKER_TOLERANCE_PX,
        allow_truncation: bool = False,
    ):
        """Initialize the cache.

        Args:
            structure_library: ChipStructureLibrary or SAKRoIStructureLibrary
                providing polygon_library and marker_group_configs.
            tolerance: Pixel tolerance for marker matching.
            allow_truncation: Allow ROI mask beyond image boundaries.
        """
        self.structure_library = structure_library
        self.tolerance = tolerance
        self.allow_truncation = allow_truncation
        self._cache: dict[str, dict] = {}

    def get(self, structure_name: str) -> dict:
        """Get or create pipeline components for a chamber type.

        Args:
            structure_name: Chamber structure name (e.g., "NormaleBox-inner").

        Returns:
            Dict with keys: roi_polygon, marker_group, matching_step,
            rotation_step, masking_step.

        Raises:
            KeyError: If structure_name is not in the structure library.
        """
        if structure_name not in self._cache:
            if structure_name not in self.structure_library.polygon_library:
                raise KeyError(
                    f"Unknown structure name: '{structure_name}'. "
                    f"Available: {list(self.structure_library.polygon_library.keys())}"
                )
            roi_polygon = self.structure_library.polygon_library[structure_name]
            marker_group = self.structure_library.marker_group_configs[structure_name]
            self._cache[structure_name] = {
                "roi_polygon": roi_polygon,
                "marker_group": marker_group,
                "matching_step": MarkerMatchingStep(marker_group, tolerance=self.tolerance),
                "rotation_step": ImageRotationStep(),
                "masking_step": RoIMaskingStep(
                    marker_group, roi_polygon, allow_truncation=self.allow_truncation
                ),
            }
        return self._cache[structure_name]

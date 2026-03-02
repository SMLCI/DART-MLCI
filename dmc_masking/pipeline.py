"""Step-based pipeline classes for marker detection, matching, rotation, and masking."""

import numpy as np
import torch

from dmc_masking.constants import DEFAULT_MODEL_PATH
from dmc_masking.detection import MarkerDetectionModel
from dmc_masking.mask import apply_mask
from dmc_masking.match import match_markers
from dmc_masking.rotation import compute_marker_group_angles, rotate_image_and_markers


class MarkerDetectionStep:
    """Detect markers in an image."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        verbose: bool = False,
        use_gpu_tensor: bool = False,
    ):
        """Initialize detection step.

        Args:
            model_path: Path to YOLO model weights. If None, uses DEFAULT_MODEL_PATH.
            device: Device to use for inference.
            verbose: Show YOLO inference output.
            use_gpu_tensor: Keep image on GPU for downstream steps to avoid redundant
                           transfers. Set True for performance, False for compatibility
                           with code expecting numpy arrays. (default: False)
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.mdm = MarkerDetectionModel(model_path, device=device, verbose=verbose)
        self.mdm.model.conf = 0.6
        self.use_gpu_tensor = use_gpu_tensor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image):
        # YOLO requires numpy input for proper preprocessing (resize, normalize, etc.)
        # After detection, convert to GPU tensor for downstream steps (rotation, etc.)
        markers = self.mdm.predict_markers(image)

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

    def __init__(self, marker_group_pixel, tolerance=60):
        self.marker_group_pixel = marker_group_pixel
        self.tolerance = tolerance

    def __call__(self, data):
        markers = data["markers"]

        matched_marker_indices = match_markers(
            markers, marker_group=self.marker_group_pixel, tolerance=self.tolerance
        )

        data["matched_marker_indices"] = matched_marker_indices

        angles = compute_marker_group_angles(
            markers, matched_marker_indices, self.marker_group_pixel
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

    def __init__(self, marker_group_pixels, roi_polygon):
        super().__init__()

        self.marker_group_pixels = marker_group_pixels
        self.roi_polygon = roi_polygon

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

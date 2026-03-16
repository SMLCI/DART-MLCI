"""Translation-based image registration using normalized cross-correlation on marker regions."""

import warnings

import cv2
import kornia.geometry.transform as KT
import numpy as np
import torch

from ._base import BaseRegistration


class TimelapseRegistration(BaseRegistration):
    """Translation-only registration using normalized cross-correlation on marker regions.

    This class provides GPU-accelerated translation-based registration for time-lapse
    microscopy images. It uses the marker region (where cross/circle markers are located)
    to compute optimal translation between frames.

    The registration process:
    1. Extract rectangular bounding box containing all matched markers
    2. Compute translation using normalized cross-correlation (grid search)
    3. Apply translation to images and masks using kornia's GPU-accelerated transforms

    Args:
        marker_group_pixel: Dict with marker positions in pixels,
            e.g. ``{"cross": np.array([x, y]), "circle": np.array([x, y])}``.
        max_translation: Maximum translation search range in pixels (default: 20).
        padding: Padding around marker region in pixels (default: 50).
        device: Device to use ('cuda', 'cpu', or None for auto-detect).
    """

    def __init__(
        self,
        marker_group_pixel: dict[str, np.ndarray],
        max_translation: int = 20,
        padding: int = 50,
        device: str | None = None,
    ):
        super().__init__(marker_group_pixel=marker_group_pixel, padding=padding)

        self.max_translation = max_translation

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def compute_translation(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray,
        reference_markers: list[dict] | None = None,
        target_markers: list[dict] | None = None,
    ) -> tuple[float, float, float]:
        """Compute translation (dx, dy) from reference to target using NCC.

        Uses normalized cross-correlation on the marker region to find the best
        translation that aligns the target image to the reference.

        Args:
            reference_image: Reference frame in HWC format (uint8 or float).
            target_image: Target frame to align in HWC format.
            reference_markers: Deprecated, unused. Kept for backward compatibility.
            target_markers: Deprecated, unused. Kept for backward compatibility.

        Returns:
            Tuple of ``(dx, dy, score)``:
                - dx: Translation in x direction (pixels)
                - dy: Translation in y direction (pixels)
                - score: Normalized cross-correlation score (0-1, higher is better)
        """
        # Extract marker regions
        ref_region = self.extract_marker_region(reference_image)
        target_region = self.extract_marker_region(target_image)

        # Convert to grayscale if needed
        if ref_region.ndim == 3 and ref_region.shape[2] > 1:
            ref_region = cv2.cvtColor(ref_region, cv2.COLOR_RGB2GRAY)
        if target_region.ndim == 3 and target_region.shape[2] > 1:
            target_region = cv2.cvtColor(target_region, cv2.COLOR_RGB2GRAY)

        # Ensure float32 for correlation
        ref_region = ref_region.astype(np.float32)
        target_region = target_region.astype(np.float32)

        # Use OpenCV template matching with normalized cross-correlation
        # TM_CCOEFF_NORMED gives correlation coefficient in [-1, 1]
        # We search in an expanded target region to allow for translation
        pad = self.max_translation
        target_padded = cv2.copyMakeBorder(
            target_region,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        # Match reference in padded target
        result = cv2.matchTemplate(target_padded, ref_region, cv2.TM_CCOEFF_NORMED)

        # Find best match
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)

        # Convert location to translation (relative to center of search region)
        # max_loc is top-left corner of best match in target_padded
        # If no translation, max_loc should be (pad, pad)
        dx = max_loc[0] - pad
        dy = max_loc[1] - pad

        # Score is the correlation coefficient (higher is better)
        score = max_val

        return float(dx), float(dy), float(score)

    def apply_translation(
        self,
        image: np.ndarray | torch.Tensor,
        dx: float,
        dy: float,
        return_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Apply translation to an image using kornia's GPU-accelerated transform.

        Args:
            image: Image in HWC format (numpy) or CHW format (tensor).
            dx: Translation in x direction (pixels).
            dy: Translation in y direction (pixels).
            return_tensor: If True, return torch.Tensor; else return numpy array.

        Returns:
            Translated image in same format as input (HWC for numpy, CHW for tensor).
        """
        # Handle numpy input
        if isinstance(image, np.ndarray):
            # Assume HWC format for numpy
            h, w = image.shape[:2]
            # Convert to CHW tensor
            if image.ndim == 2:
                # Grayscale: HxW -> 1xHxW
                image_tensor = torch.from_numpy(image[None, :, :]).float().to(self.device)
            else:
                # RGB: HxWxC -> CxHxW
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        else:
            # Already tensor, assume CHW format
            image_tensor = image.to(self.device).float()
            if image_tensor.dim() == 3:
                h, w = image_tensor.shape[-2:]
            else:
                raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

        # Add batch dimension if needed: CxHxW -> 1xCxHxW
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Create translation matrix (2x3 affine matrix)
        # [1  0  dx]
        # [0  1  dy]
        translation_matrix = torch.tensor(
            [[1.0, 0.0, dx], [0.0, 1.0, dy]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # Add batch dimension: 1x2x3

        # Apply translation using kornia
        translated = KT.warp_affine(
            image_tensor,
            translation_matrix,
            dsize=(h, w),
            mode="bilinear",
            padding_mode="zeros",
        )

        # Remove batch dimension: 1xCxHxW -> CxHxW
        translated = translated.squeeze(0)

        # Convert back to original format
        if return_tensor or isinstance(image, torch.Tensor):
            return translated
        else:
            # Convert back to numpy HWC
            translated_np = translated.cpu().numpy()
            if translated_np.shape[0] == 1:
                # Grayscale: 1xHxW -> HxW
                return translated_np[0]
            else:
                # RGB: CxHxW -> HxWxC
                return translated_np.transpose(1, 2, 0)

    def apply_translation_to_image(
        self,
        image: np.ndarray | torch.Tensor,
        dx: float,
        dy: float,
        return_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Deprecated: use ``apply_translation`` instead.

        Args:
            image: Image in HWC format (numpy) or CHW format (tensor).
            dx: Translation in x direction (pixels).
            dy: Translation in y direction (pixels).
            return_tensor: If True, return torch.Tensor; else return numpy array.

        Returns:
            Translated image.
        """
        warnings.warn(
            "apply_translation_to_image() is deprecated, use apply_translation() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.apply_translation(image, dx, dy, return_tensor=return_tensor)

    def apply_translation_to_mask(
        self,
        mask: np.ndarray | torch.Tensor,
        dx: float,
        dy: float,
        return_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Apply translation to a mask using nearest-neighbor interpolation.

        Args:
            mask: Binary or labeled mask in HW format (numpy) or HW format (tensor).
            dx: Translation in x direction (pixels).
            dy: Translation in y direction (pixels).
            return_tensor: If True, return torch.Tensor; else return numpy array.

        Returns:
            Translated mask in same format as input (HW).
        """
        # Handle numpy input
        if isinstance(mask, np.ndarray):
            h, w = mask.shape
            # Add channel dimension: HxW -> 1xHxW
            mask_tensor = torch.from_numpy(mask[None, :, :]).float().to(self.device)
        else:
            # Already tensor
            mask_tensor = mask.to(self.device).float()
            if mask_tensor.dim() == 2:
                h, w = mask_tensor.shape
                mask_tensor = mask_tensor.unsqueeze(0)  # HxW -> 1xHxW
            else:
                raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

        # Add batch dimension: 1xHxW -> 1x1xHxW
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Create translation matrix
        translation_matrix = torch.tensor(
            [[1.0, 0.0, dx], [0.0, 1.0, dy]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # Apply translation with nearest-neighbor interpolation (preserves labels)
        translated = KT.warp_affine(
            mask_tensor,
            translation_matrix,
            dsize=(h, w),
            mode="nearest",
            padding_mode="zeros",
        )

        # Remove batch and channel dimensions: 1x1xHxW -> HxW
        translated = translated.squeeze(0).squeeze(0)

        # Convert back to original format
        if return_tensor or isinstance(mask, torch.Tensor):
            return translated
        else:
            return translated.cpu().numpy()

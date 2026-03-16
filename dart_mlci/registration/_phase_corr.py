"""Phase correlation-based image registration for microscopy time-lapse data.

This module implements translation registration using phase correlation (FFT-based),
which is significantly faster and more robust to intensity variations than
normalized cross-correlation.
"""

import cv2
import numpy as np

from ._base import BaseRegistration
from .preprocessing import (
    create_hanning_window,
    preprocess_for_registration,
)


class PhaseCorrelationRegistration(BaseRegistration):
    """Translation registration using phase correlation (FFT-based).

    Phase correlation offers significant advantages over cross-correlation:
    - 10-100x faster using FFT (O(N log N) vs O(N^2))
    - More robust to intensity/contrast variations
    - Sharper correlation peaks for better accuracy
    - Subpixel accuracy via weighted centroid

    This method computes translation by finding the peak in the phase correlation
    between two images. It's particularly effective for low-contrast microscopy
    images where normalized cross-correlation struggles.

    Args:
        marker_group_pixel: Flat dict mapping marker IDs to pixel positions,
            e.g. ``{"cross": np.array([x, y]), "circle": np.array([x, y])}``.
        padding: Padding around marker region in pixels (default: 100).
            Larger than NCC (50) to provide better frequency representation.
        preprocess: Whether to apply CLAHE preprocessing (default: True).
        use_hanning: Apply Hanning window to reduce edge effects (default: True).
        clip_limit: CLAHE clip limit for contrast enhancement (default: 2.0).
        tile_grid_size: CLAHE tile grid size (default: (8, 8)).

    Example:
        >>> reg = PhaseCorrelationRegistration(marker_group_pixel, padding=100)
        >>> dx, dy, confidence = reg.compute_translation(frame0, frame1)
        >>> aligned = reg.apply_translation(frame1, -dx, -dy)
    """

    def __init__(
        self,
        marker_group_pixel: dict,
        padding: int = 100,
        preprocess: bool = True,
        use_hanning: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ):
        super().__init__(marker_group_pixel=marker_group_pixel, padding=padding)

        self.preprocess = preprocess
        self.use_hanning = use_hanning
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

        # Cache for Hanning window (created on first use)
        self._hanning_cache: np.ndarray | None = None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to image region.

        Args:
            image: Input image region (grayscale or RGB).

        Returns:
            Preprocessed image as float32 in [0, 1] range.
        """
        return preprocess_for_registration(
            image,
            use_clahe=self.preprocess,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            normalize=True,
        )

    def _get_hanning_window(self, shape: tuple[int, int]) -> np.ndarray:
        """Get or create Hanning window for given shape.

        Args:
            shape: (height, width) of window.

        Returns:
            2D Hanning window.
        """
        if self._hanning_cache is not None and self._hanning_cache.shape == shape:
            return self._hanning_cache

        self._hanning_cache = create_hanning_window(shape)
        return self._hanning_cache

    def compute_translation(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute translation from target to reference using phase correlation.

        Args:
            reference_image: Reference image (grayscale or RGB).
            target_image: Target image to register (grayscale or RGB).

        Returns:
            Tuple of ``(dx, dy, confidence)``:
                - dx: Translation in x direction (pixels)
                - dy: Translation in y direction (pixels)
                - confidence: Phase correlation peak confidence [0, 1]
        """
        # Extract marker regions
        ref_region = self.extract_marker_region(reference_image)
        target_region = self.extract_marker_region(target_image)

        # Preprocess
        ref_prep = self._preprocess_image(ref_region)
        target_prep = self._preprocess_image(target_region)

        # Ensure same size
        if ref_prep.shape != target_prep.shape:
            raise ValueError(f"Region size mismatch: {ref_prep.shape} vs {target_prep.shape}")

        # Apply Hanning window to reduce edge effects
        if self.use_hanning:
            window = self._get_hanning_window(ref_prep.shape)
            ref_prep = ref_prep * window
            target_prep = target_prep * window

        # Phase correlation using OpenCV
        shift, confidence = cv2.phaseCorrelate(ref_prep, target_prep)

        dx = float(shift[0])
        dy = float(shift[1])
        confidence = float(confidence)

        return dx, dy, confidence

    def apply_translation(
        self,
        image: np.ndarray,
        dx: float,
        dy: float,
        **kwargs,
    ) -> np.ndarray:
        """Apply translation to image using OpenCV warpAffine (forward mapping).

        Uses ``cv2.warpAffine`` with M = [[1,0,dx],[0,1,dy]], which shifts
        image content in the *positive* direction (right for +dx, down for +dy).

        To *undo* a detected shift and align the target back to the reference,
        pass the negated values::

            aligned = reg.apply_translation(target, -dx, -dy)

        Args:
            image: Image to translate.
            dx: Translation in x direction (pixels). Positive shifts content right.
            dy: Translation in y direction (pixels). Positive shifts content down.

        Returns:
            Translated image (same size and dtype as input).
        """
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        aligned = cv2.warpAffine(
            image,
            M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return aligned

    def apply_translation_to_mask(
        self,
        mask: np.ndarray,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        """Apply translation to a mask using nearest-neighbor interpolation.

        Args:
            mask: Binary or labeled mask in HW format (numpy).
            dx: Translation in x direction (pixels).
            dy: Translation in y direction (pixels).

        Returns:
            Translated mask (same size and dtype as input).
        """
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        translated = cv2.warpAffine(
            mask,
            M,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return translated

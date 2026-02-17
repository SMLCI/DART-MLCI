"""Abstract base class for translation-based image registration."""

from abc import ABC, abstractmethod

import numpy as np


class BaseRegistration(ABC):
    """Abstract base class for marker-region translation registration.

    Subclasses must implement ``compute_translation`` and ``apply_translation``.
    Common functionality (marker bbox computation, region extraction,
    batch registration) is provided here.

    Args:
        marker_group_pixel: Flat dict mapping marker IDs to pixel positions,
            e.g. ``{"cross": np.array([x, y]), "circle": np.array([x, y])}``.
        padding: Padding around marker bounding box in pixels.
    """

    def __init__(
        self,
        marker_group_pixel: dict[str, np.ndarray],
        padding: int = 50,
    ):
        self.marker_group_pixel = marker_group_pixel
        self.padding = padding

        # Compute marker region bbox once during init
        self.marker_bbox = self._compute_marker_bbox()

    def _compute_marker_bbox(self) -> tuple[int, int, int, int]:
        """Compute bounding box around all marker positions with padding.

        Returns:
            ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
        """
        if not self.marker_group_pixel:
            raise ValueError("No marker positions found in marker_group_pixel")

        positions = []
        for pos in self.marker_group_pixel.values():
            positions.append(np.asarray(pos))

        if not positions:
            raise ValueError("No marker positions found in marker_group_pixel")

        positions = np.array(positions)

        x_min = int(np.min(positions[:, 0]) - self.padding)
        y_min = int(np.min(positions[:, 1]) - self.padding)
        x_max = int(np.max(positions[:, 0]) + self.padding)
        y_max = int(np.max(positions[:, 1]) + self.padding)

        # Ensure non-negative
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        return (x_min, y_min, x_max, y_max)

    def extract_marker_region(self, image: np.ndarray) -> np.ndarray:
        """Extract marker region from image.

        Args:
            image: Input image (grayscale HxW or colour HxWxC).

        Returns:
            Cropped region containing the markers.
        """
        x_min, y_min, x_max, y_max = self.marker_bbox

        # Clip to image bounds
        h, w = image.shape[:2]
        x_min = max(0, min(x_min, w))
        y_min = max(0, min(y_min, h))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))

        return image[y_min:y_max, x_min:x_max]

    def get_marker_bbox(self) -> tuple[int, int, int, int]:
        """Return the computed marker bounding box.

        Returns:
            ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
        """
        return self.marker_bbox

    def get_marker_region_size(self) -> tuple[int, int]:
        """Return the size of the marker region.

        Returns:
            ``(width, height)`` in pixels.
        """
        x_min, y_min, x_max, y_max = self.marker_bbox
        return (x_max - x_min, y_max - y_min)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_translation(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute translation from reference to target.

        Args:
            reference_image: Reference frame.
            target_image: Target frame to align.

        Returns:
            ``(dx, dy, score)`` where *dx*/*dy* are pixel translations and
            *score* is a quality metric (higher is better).
        """

    @abstractmethod
    def apply_translation(
        self,
        image: np.ndarray,
        dx: float,
        dy: float,
        **kwargs,
    ) -> np.ndarray:
        """Apply a translation to an image.

        Args:
            image: Image to translate.
            dx: Translation in x direction (pixels).
            dy: Translation in y direction (pixels).

        Returns:
            Translated image (same size as input).
        """

    # ------------------------------------------------------------------
    # Batch registration
    # ------------------------------------------------------------------

    def register_to_reference(
        self,
        reference_image: np.ndarray,
        target_images: list,
    ) -> list:
        """Register multiple target images to a reference.

        For each target the detected translation is *negated* before being
        applied so that the target is aligned back to the reference.

        Args:
            reference_image: Reference image.
            target_images: List of images to register.

        Returns:
            List of ``(aligned_image, dx, dy, score)`` tuples.
        """
        results = []
        for target in target_images:
            dx, dy, score = self.compute_translation(reference_image, target)
            aligned = self.apply_translation(target, -dx, -dy)
            results.append((aligned, dx, dy, score))
        return results

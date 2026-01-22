"""Coordinate system transforms for calibration.

This module provides classes for transforming between different coordinate systems
used in the DMC masking pipeline:
- Blueprint coordinates (design, microns)
- Image pixel coordinates
- Stage/microscope coordinates (microns)

The key coordinate system conventions:
- Blueprint: Cartesian convention where +Y points UP
- Image: Standard image convention where +Y points DOWN
- Stage: Hardware-dependent, typically +Y DOWN

The Y-inversion between blueprint and image coordinates is handled explicitly
in the offset calculations (using + instead of - for Y).
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import numpy.typing as npt


class CoordinateSystem(Enum):
    """Enumeration of coordinate systems used in the pipeline."""

    BLUEPRINT = auto()  # Design coordinates (microns, Cartesian Y-up)
    IMAGE_PIXEL = auto()  # Pixel coordinates (origin top-left, Y-down)
    IMAGE_MICRON = auto()  # Image coordinates in microns
    STAGE = auto()  # Microscope stage (microns, hardware-dependent)


@dataclass
class TransformFitResult:
    """Result of fitting a transform to point correspondences.

    Attributes:
        residuals: Per-point residual distances after fitting
        rmse: Root mean square error
        max_error: Maximum residual
    """

    residuals: np.ndarray
    rmse: float
    max_error: float


@dataclass
class PixelToMicronTransform:
    """Scale transform from pixels to microns.

    This is a simple scaling transform that converts pixel coordinates
    to physical units (microns).

    Attributes:
        pixel_size: Size of one pixel in microns

    Example:
        >>> transform = PixelToMicronTransform(pixel_size=0.065789)
        >>> pixels = np.array([100, 200])
        >>> microns = transform(pixels)
        >>> print(microns)  # [6.5789, 13.1578]
    """

    pixel_size: float

    def __call__(self, pixels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform pixel coordinates to microns.

        Args:
            pixels: Coordinates in pixels (single point or Nx2 array)

        Returns:
            Coordinates in microns
        """
        return pixels * self.pixel_size

    def inverse(self, microns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform micron coordinates to pixels.

        Args:
            microns: Coordinates in microns (single point or Nx2 array)

        Returns:
            Coordinates in pixels
        """
        return microns / self.pixel_size


@dataclass
class ImageToStageTransform:
    """Translate from image-relative to stage-absolute coordinates.

    When an image is captured, the microscope stage is at a known position.
    This position represents the top-left corner of the image in stage coordinates.
    To convert any point in the image (in microns) to stage coordinates,
    we add the stage position.

    Attributes:
        stage_position: (x, y) position of image top-left in stage coords

    Example:
        >>> transform = ImageToStageTransform(stage_position=np.array([6802.4, -4272.9]))
        >>> image_pos = np.array([32.89, 19.74])  # microns from top-left
        >>> stage_pos = transform(image_pos)
        >>> print(stage_pos)  # [6835.29, -4253.16]
    """

    stage_position: npt.NDArray[np.float64]

    def __call__(self, image_microns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform image-relative coordinates to stage coordinates.

        Args:
            image_microns: Position in image coordinates (microns from top-left)

        Returns:
            Position in stage coordinates
        """
        return image_microns + self.stage_position

    def inverse(self, stage_microns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform stage coordinates to image-relative coordinates.

        Args:
            stage_microns: Position in stage coordinates

        Returns:
            Position in image coordinates (microns from top-left)
        """
        return stage_microns - self.stage_position


@dataclass
class AffineTransform2D:
    """General 2D affine transform with homogeneous matrix representation.

    Handles translation, rotation, scale, and mirror/flip operations.
    Uses a 3x3 homogeneous matrix internally.

    The matrix format is:
        [[a, b, tx],
         [c, d, ty],
         [0, 0, 1]]

    Where:
        - (a, b, c, d) form the 2x2 linear transform (rotation, scale, shear)
        - (tx, ty) is the translation vector

    Attributes:
        matrix: 3x3 homogeneous transformation matrix

    Example:
        >>> # Create identity transform
        >>> t = AffineTransform2D.identity()
        >>> point = np.array([10.0, 20.0])
        >>> result = t(point)
        >>> np.allclose(result, point)  # True

        >>> # Create translation
        >>> t = AffineTransform2D.translation(50, 100)
        >>> result = t(np.array([0.0, 0.0]))
        >>> print(result)  # [50.0, 100.0]
    """

    matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def __post_init__(self):
        """Validate matrix shape."""
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3, got shape {self.matrix.shape}")

    def __call__(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply transform to points.

        Args:
            points: Single point (2,) or array of points (Nx2)

        Returns:
            Transformed points (same shape as input)
        """
        points = np.asarray(points, dtype=np.float64)
        single_point = points.ndim == 1

        if single_point:
            points = points.reshape(1, -1)

        # Add homogeneous coordinate
        n_points = points.shape[0]
        ones = np.ones((n_points, 1), dtype=np.float64)
        homogeneous = np.hstack([points, ones])

        # Apply transform
        result = (self.matrix @ homogeneous.T).T

        # Extract x, y (discard homogeneous coordinate)
        result = result[:, :2]

        if single_point:
            return result.squeeze()
        return result

    @property
    def inverse(self) -> "AffineTransform2D":
        """Compute the inverse transform.

        Returns:
            AffineTransform2D that undoes this transform
        """
        return AffineTransform2D(np.linalg.inv(self.matrix))

    def __matmul__(self, other: "AffineTransform2D") -> "AffineTransform2D":
        """Compose transforms: self @ other means apply other first, then self.

        Args:
            other: Transform to apply first

        Returns:
            Composed transform
        """
        return AffineTransform2D(self.matrix @ other.matrix)

    @classmethod
    def identity(cls) -> "AffineTransform2D":
        """Create an identity transform (no-op).

        Returns:
            Identity transform
        """
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineTransform2D":
        """Create a translation transform.

        Args:
            tx: Translation in x
            ty: Translation in y

        Returns:
            Translation transform
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = tx
        matrix[1, 2] = ty
        return cls(matrix)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> "AffineTransform2D":
        """Create a scale transform.

        Args:
            sx: Scale factor in x
            sy: Scale factor in y (defaults to sx for uniform scale)

        Returns:
            Scale transform
        """
        if sy is None:
            sy = sx
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        return cls(matrix)

    @classmethod
    def rotation(cls, angle: float, degrees: bool = True) -> "AffineTransform2D":
        """Create a rotation transform around the origin.

        Args:
            angle: Rotation angle (positive = counter-clockwise)
            degrees: If True, angle is in degrees; otherwise radians

        Returns:
            Rotation transform
        """
        if degrees:
            angle = np.radians(angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
        return cls(matrix)

    @classmethod
    def mirror_x(cls) -> "AffineTransform2D":
        """Create a transform that mirrors around the Y-axis (flips X).

        Returns:
            Mirror transform
        """
        return cls.scale(-1, 1)

    @classmethod
    def mirror_y(cls) -> "AffineTransform2D":
        """Create a transform that mirrors around the X-axis (flips Y).

        Returns:
            Mirror transform
        """
        return cls.scale(1, -1)

    @classmethod
    def from_point_pairs(
        cls,
        source: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> tuple["AffineTransform2D", TransformFitResult]:
        """Compute affine transform from corresponding point pairs.

        Uses least squares fitting to find the best affine transform
        that maps source points to target points.

        Args:
            source: Nx2 array of source points
            target: Nx2 array of corresponding target points

        Returns:
            Tuple of (transform, fit_result) where fit_result contains
            residuals and error metrics

        Raises:
            ValueError: If fewer than 3 point pairs are provided
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        n_points = source.shape[0]
        if n_points < 3:
            raise ValueError(f"Need at least 3 points for affine transform, got {n_points}")

        # Build design matrix: [x, y, 1] for each point
        design = np.hstack((source, np.ones((n_points, 1))))

        # Solve using least squares: design @ Ab = target
        # Ab is 3x2: [[a, c], [b, d], [tx, ty]]
        Ab, residuals, rank, s = np.linalg.lstsq(design, target, rcond=None)

        # Construct 3x3 homogeneous matrix
        # Ab format: [[a, c], [b, d], [tx, ty]]
        # We want: [[a, b, tx], [c, d, ty], [0, 0, 1]]
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = Ab[0, 0]  # a
        matrix[0, 1] = Ab[1, 0]  # b
        matrix[0, 2] = Ab[2, 0]  # tx
        matrix[1, 0] = Ab[0, 1]  # c
        matrix[1, 1] = Ab[1, 1]  # d
        matrix[1, 2] = Ab[2, 1]  # ty

        transform = cls(matrix)

        # Compute per-point residuals
        transformed = transform(source)
        point_residuals = np.linalg.norm(transformed - target, axis=1)

        fit_result = TransformFitResult(
            residuals=point_residuals,
            rmse=float(np.sqrt(np.mean(point_residuals**2))),
            max_error=float(np.max(point_residuals)),
        )

        return transform, fit_result

    def to_matrix_2x3(self) -> npt.NDArray[np.float64]:
        """Extract the 2x3 matrix representation.

        Returns:
            2x3 matrix [[a, b, tx], [c, d, ty]]
        """
        return self.matrix[:2, :]

    @classmethod
    def from_matrix_2x3(cls, matrix_2x3: npt.NDArray[np.float64]) -> "AffineTransform2D":
        """Create transform from a 2x3 matrix.

        Args:
            matrix_2x3: 2x3 matrix [[a, b, tx], [c, d, ty]]

        Returns:
            AffineTransform2D
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = matrix_2x3
        return cls(matrix)


def compute_blueprint_to_image_offset(
    polygon_center: npt.NDArray[np.float64],
    marker_position: npt.NDArray[np.float64],
    invert_y: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute the offset from marker to polygon center in image coordinates.

    This handles the Y-axis inversion between blueprint (Cartesian Y-up)
    and image (Y-down) coordinate systems.

    In blueprint coordinates:
        marker_position[1] = 8 means the marker is 8 units ABOVE the origin

    In image coordinates:
        We need to ADD this value (not subtract) to get the correct position
        because image Y increases downward.

    Args:
        polygon_center: Center of the polygon in local coordinates
        marker_position: Position of the marker (e.g., cross) in local coordinates
        invert_y: Whether to invert Y (True for blueprint->image)

    Returns:
        Offset vector that can be added to detected marker position

    Example:
        >>> center = np.array([50.0, 50.0])
        >>> marker = np.array([14.0, 8.0])
        >>> offset = compute_blueprint_to_image_offset(center, marker)
        >>> # offset[0] = 50 - 14 = 36
        >>> # offset[1] = 50 + 8 = 58  (note: + because of Y inversion)
    """
    offset_x = polygon_center[0] - marker_position[0]

    if invert_y:
        # Blueprint uses Y-up, image uses Y-down
        # So we ADD the marker Y instead of subtracting
        offset_y = polygon_center[1] + marker_position[1]
    else:
        # Same coordinate system
        offset_y = polygon_center[1] - marker_position[1]

    return np.array([offset_x, offset_y])


def apply_rotation_to_offset(
    offset: npt.NDArray[np.float64],
    rotation_angle: float,
    degrees: bool = True,
) -> npt.NDArray[np.float64]:
    """Apply rotation to an offset vector.

    Used to correct for image rotation when computing chamber center
    from marker positions.

    Args:
        offset: Offset vector to rotate
        rotation_angle: Rotation angle (positive = counter-clockwise)
        degrees: If True, angle is in degrees; otherwise radians

    Returns:
        Rotated offset vector
    """
    if degrees:
        angle_rad = np.radians(rotation_angle)
    else:
        angle_rad = rotation_angle

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotated = np.array(
        [
            offset[0] * cos_a - offset[1] * sin_a,
            offset[0] * sin_a + offset[1] * cos_a,
        ]
    )

    return rotated

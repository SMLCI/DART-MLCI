"""Calibration module for DMC masking.

This module provides functionality for calibrating microscope maps from
calibration images with known stage positions.

Coordinate System Overview
--------------------------
The calibration pipeline works with three coordinate systems:

1. Blueprint (Design) Coordinates
   - Units: microns
   - Origin: Design-defined (typically top-left of chip)
   - Convention: Cartesian (+Y points UP)

2. Image Pixel Coordinates
   - Units: pixels
   - Origin: Top-left of image
   - Convention: Standard image (+Y points DOWN)

3. Stage (Microscope) Coordinates
   - Units: microns
   - Origin: Hardware reference
   - Convention: Hardware-dependent (typically +Y DOWN)

The key coordinate handling:
- Blueprint uses Y-up (Cartesian), image uses Y-down
- This Y-inversion is handled explicitly with + instead of -
- The affine transform captures any additional rotations/flips

Example Usage
-------------
>>> from dart_mlci.calibration import (
...     AffineTransform2D,
...     PixelToMicronTransform,
...     ImageToStageTransform,
... )

>>> # Convert pixels to microns
>>> pixel_transform = PixelToMicronTransform(pixel_size=0.065789)
>>> center_microns = pixel_transform(center_pixels)

>>> # Convert image coordinates to stage coordinates
>>> stage_transform = ImageToStageTransform(stage_position=np.array([6802.4, -4272.9]))
>>> stage_pos = stage_transform(center_microns)

>>> # Compute affine transform from point correspondences
>>> transform, fit_result = AffineTransform2D.from_point_pairs(blueprint_pts, measured_pts)
>>> calibrated_map = transform(blueprint_map)
"""

from .coordinates import (
    AffineTransform2D,
    CoordinateSystem,
    ImageToStageTransform,
    PixelToMicronTransform,
    TransformFitResult,
    apply_rotation_to_offset,
    compute_blueprint_to_image_offset,
)
from .core import (
    CalibrationError,
    CalibrationResult,
    ImageCalibrationResult,
    ImageDebugData,
    compute_chamber_center,
    compute_microscope_position,
    filter_matched_pairs_by_bounds,
    process_calibration_image,
    run_calibration,
)
from .validation import (
    ValidationDebugData,
    ValidationResult,
    ValidationSummary,
    process_validation_image,
    run_validation,
)

__all__ = [
    "AffineTransform2D",
    "CalibrationError",
    "CalibrationResult",
    "CoordinateSystem",
    "ImageCalibrationResult",
    "ImageDebugData",
    "ImageToStageTransform",
    "PixelToMicronTransform",
    "TransformFitResult",
    "ValidationDebugData",
    "ValidationResult",
    "ValidationSummary",
    "apply_rotation_to_offset",
    "compute_blueprint_to_image_offset",
    "compute_chamber_center",
    "compute_microscope_position",
    "filter_matched_pairs_by_bounds",
    "process_calibration_image",
    "process_validation_image",
    "run_calibration",
    "run_validation",
]

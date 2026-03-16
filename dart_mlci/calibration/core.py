"""Core calibration dataclasses and functions.

This module provides the core data structures and computation functions
for the calibration pipeline.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from dart_mlci.map import AffineTransformResult, Map
from dart_mlci.mask import RoIPolygon


@dataclass
class ImageDebugData:
    """Debug data for per-image visualization.

    Stores intermediate results from processing a single calibration image,
    useful for debugging and generating visualizations.

    Attributes:
        image: The loaded image array (HxWx3 uint8)
        markers: List of detected markers with bbox_center and label
        matched_indices: List of (cross_idx, circle_idx) tuples
        chamber_center_pixels: Computed chamber center in pixel coordinates
        chamber_center_microns: Chamber center in microscope coordinates
        stage_position: Stage position dict with x, y, and optionally z
        pixel_size: Pixel size in microns
        structure_name: Name of the chamber structure
        roi_polygon: The RoI polygon for this chamber type
        marker_group_pixels: Expected marker positions in pixels
        rotation_angle: Detected rotation angle in degrees
    """

    image: np.ndarray | None = None
    markers: list[dict] | None = None
    matched_indices: list[tuple[int, int]] | None = None
    chamber_center_pixels: np.ndarray | None = None
    chamber_center_microns: np.ndarray | None = None
    stage_position: dict[str, float] | None = None
    pixel_size: float | None = None
    structure_name: str | None = None
    roi_polygon: RoIPolygon | None = None
    marker_group_pixels: dict[str, np.ndarray] | None = None
    rotation_angle: float | None = None


@dataclass
class ImageCalibrationResult:
    """Result of processing a single calibration image.

    Attributes:
        roi_id: RoI identifier (e.g., "0050")
        success: Whether processing was successful
        microscope_position: (x, y) position in microscope coordinates (microns)
        z_position: Z position (from stage_position)
        error_message: Error message if processing failed
        debug_data: Optional debug data for visualization
    """

    roi_id: str
    success: bool
    microscope_position: np.ndarray | None
    z_position: float | None
    error_message: str | None = None
    debug_data: ImageDebugData | None = None


@dataclass
class CalibrationResult:
    """Result of the full calibration process.

    Attributes:
        measured_map: Map of measured positions from successful calibrations
        transform_result: Affine transform and error metrics
        calibrated_map: Full blueprint map transformed to microscope coordinates
        image_results: Per-image processing results
        z_positions: Dictionary mapping roi_id to z position
    """

    measured_map: Map
    transform_result: AffineTransformResult
    calibrated_map: Map
    image_results: list[ImageCalibrationResult]
    z_positions: dict[str, float] = field(default_factory=dict)


def compute_chamber_center(
    markers: list[dict],
    matched_indices: list[tuple[int, int]],
    marker_group_pixels: dict[str, npt.NDArray[np.float64]],
    roi_polygon: RoIPolygon,
    rotation_angle: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Compute the chamber center in pixel coordinates.

    The chamber center is computed by finding the offset from the cross marker
    to the polygon center in blueprint coordinates, then rotating that offset
    by the detected rotation angle to account for image orientation.

    IMPORTANT: The Y-coordinate offset uses ADDITION (not subtraction) because
    the blueprint uses Cartesian Y-up convention while images use Y-down.
    This is the key coordinate system handling in the calibration pipeline.

    Args:
        markers: List of detected markers with bbox_center
        matched_indices: List of (cross_idx, circle_idx) tuples
        marker_group_pixels: Expected marker positions in pixels
        roi_polygon: RoI polygon for getting centroid
        rotation_angle: Rotation angle in degrees from markers (default: 0.0)

    Returns:
        Chamber center position in pixels as (x, y) array

    Raises:
        ValueError: If no matched marker pairs are provided
    """
    if not matched_indices:
        raise ValueError("No matched marker pairs found")

    # Get polygon centroid in the polygon's local coordinate system
    polygon_center = roi_polygon.center

    # The cross_local from marker_group_pixels is in the ORIGINAL coordinate system
    cross_local = marker_group_pixels["cross"]

    # Compute offset from cross to polygon center
    # CRITICAL: Y uses + not - due to Y-axis inversion between blueprint and image
    center_offset = np.array(
        [
            polygon_center[0] - cross_local[0],
            polygon_center[1] + cross_local[1],  # Uses + for Y-inversion
        ]
    )

    # Apply rotation to the offset to account for image orientation
    angle_rad = np.radians(rotation_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotated_offset = np.array(
        [
            center_offset[0] * cos_a - center_offset[1] * sin_a,
            center_offset[0] * sin_a + center_offset[1] * cos_a,
        ]
    )

    # Use first matched pair (the one with best distance match)
    cross_idx, _circle_idx = matched_indices[0]
    cross_detected = markers[cross_idx]["bbox_center"]

    # Chamber center = detected cross + rotated offset to center
    return cross_detected + rotated_offset


def filter_matched_pairs_by_bounds(
    markers: list[dict],
    matched_indices: list[tuple[int, int]],
    marker_group_pixels: dict[str, npt.NDArray[np.float64]],
    roi_polygon: RoIPolygon,
    image_shape: tuple[int, int],
    rotation_angle: float = 0.0,
) -> list[tuple[int, int]]:
    """Filter matched marker pairs to keep only those with RoI fully within image bounds.

    This function positions the RoI polygon using the same rotation-aware logic as
    apply_mask_rotation_free() to ensure consistent bounds checking.

    Args:
        markers: List of detected markers with bbox_center
        matched_indices: List of (cross_idx, circle_idx) tuples
        marker_group_pixels: Expected marker positions in pixels
        roi_polygon: RoI polygon template
        image_shape: (height, width) of the image
        rotation_angle: Rotation angle in degrees from markers

    Returns:
        Filtered list of matched indices, sorted by margin to image boundary (largest first)
    """
    im_height, im_width = image_shape
    valid_pairs = []

    # Get the cross marker position in the polygon's local coordinate system
    cross_local = marker_group_pixels["cross"]

    for cross_idx, circle_idx in matched_indices:
        cross_marker = markers[cross_idx]
        circle_marker = markers[circle_idx]

        # Compute scaling correction using Euclidean distance
        # This works correctly for any rotation angle (unlike X-only distance)
        detected_dist = np.linalg.norm(cross_marker["bbox_center"] - circle_marker["bbox_center"])
        expected_dist = np.linalg.norm(marker_group_pixels["cross"] - marker_group_pixels["circle"])
        diff = detected_dist - expected_dist

        # Position the polygon using the same logic as apply_mask_rotation_free:
        # 1. Compute rotation origin in polygon's local coordinates
        rotation_origin = (cross_local[0], -cross_local[1])

        # 2. Rotate the polygon around the cross marker position
        rp = roi_polygon.rotate(rotation_angle, origin=rotation_origin)

        # 3. Translate so the cross marker aligns with detection
        rp = rp.translate(
            x=cross_marker["bbox_center"][0] - rotation_origin[0] + diff,
            y=cross_marker["bbox_center"][1] - rotation_origin[1],
        )

        # Check if polygon is within image bounds
        xmin, ymin, xmax, ymax = rp.roi_polygon.bounds

        if xmin < 0 or xmax > im_width or ymin < 0 or ymax > im_height:
            continue

        # Compute minimum margin to boundary
        min_margin = min(xmin, ymin, im_width - xmax, im_height - ymax)
        valid_pairs.append(((cross_idx, circle_idx), min_margin))

    # Sort by margin (largest first) and return just the indices
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    return [pair for pair, margin in valid_pairs]


def compute_microscope_position(
    chamber_center_pixels: npt.NDArray[np.float64],
    stage_position: dict[str, float],
    pixel_size: float,
) -> tuple[npt.NDArray[np.float64], float | None]:
    """Compute microscope position from chamber center and stage position.

    The microscope position is computed by:
    1. Converting chamber center from pixels to microns
    2. Adding the stage position (which is the top-left of the image)

    Args:
        chamber_center_pixels: Chamber center in pixel coordinates
        stage_position: Stage position dict with x, y, and optionally z
        pixel_size: Pixel size in microns

    Returns:
        Tuple of (microscope_position_xy, z_position) where:
        - microscope_position_xy is (x, y) in microns
        - z_position is z coordinate or None if not provided
    """
    chamber_center_microns = chamber_center_pixels * pixel_size

    microscope_x = stage_position["x"] + chamber_center_microns[0]
    microscope_y = stage_position["y"] + chamber_center_microns[1]
    z_position = stage_position.get("z")

    return np.array([microscope_x, microscope_y]), z_position

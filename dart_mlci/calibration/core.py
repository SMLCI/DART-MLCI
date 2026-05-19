"""Core calibration dataclasses and functions.

This module provides the core data structures and computation functions
for the calibration pipeline, including per-image processing and the
full calibration orchestration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from dart_mlci.map import AffineTransformResult, Map, RoIPosition
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

    def save_stats(self, stats_path: Path) -> None:
        """Write calibration statistics (RMSE, residuals, failures) to JSON."""
        successful_results = [r for r in self.image_results if r.success]
        residuals = {
            r.roi_id: float(self.transform_result.residuals[i])
            for i, r in enumerate(successful_results)
        }
        failed_images = [
            {"roi_id": r.roi_id, "error": r.error_message}
            for r in self.image_results
            if not r.success
        ]
        stats = {
            "transform_stats": {
                "rmse": float(self.transform_result.rmse),
                "max_error": float(self.transform_result.max_error),
                "n_calibration_points": len(successful_results),
                "residuals": residuals,
            },
            "failed_images": failed_images,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)


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


def process_calibration_image(
    image: np.ndarray,
    roi_id: str,
    stage_position: dict[str, float],
    detection_step,
    structure_library,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> ImageCalibrationResult:
    """Process a single calibration image from a numpy array.

    Args:
        image: HxWx3 uint8 image array (already loaded into memory)
        roi_id: RoI identifier (e.g., "0050")
        stage_position: Stage position dict with x, y, and optionally z
        detection_step: MarkerDetectionStep instance
        structure_library: ChipStructureLibrary or SAKRoIStructureLibrary
        pixel_size: Pixel size in microns
        verbose: Print progress information
        collect_debug: Collect debug data for visualization
        conf_threshold: Minimum confidence for detected markers (default: 0.5)
        max_angle_deviation: Maximum allowed range (in degrees) between rotation
            angles from different marker pairs. If exceeded, the image is
            rejected. (default: 5.0)

    Returns:
        ImageCalibrationResult with microscope position or error
    """
    from dart_mlci import MarkerMatchingStep
    from dart_mlci.rotation import compute_marker_group_angles

    debug_data = ImageDebugData(stage_position=stage_position) if collect_debug else None

    try:
        # 1. Auto-detect chamber type from roi_id
        structure_name, roi_polygon, marker_group_pixels = structure_library(roi_id)

        if verbose:
            print(f"    - Chamber type: {structure_name}")

        if debug_data:
            debug_data.structure_name = structure_name
            debug_data.roi_polygon = roi_polygon
            debug_data.marker_group_pixels = marker_group_pixels
            debug_data.image = image

        # 2. Create matching step for this chamber type
        matching_step = MarkerMatchingStep(marker_group_pixels, tolerance=60)

        # 3. Detect markers
        detection_result = detection_step(image)
        markers = detection_result["markers"]

        # Filter markers by confidence threshold
        markers = [m for m in markers if m.get("conf", 0.0) >= conf_threshold]
        detection_result["markers"] = markers

        if verbose:
            print(f"    - Markers detected: {len(markers)} (conf >= {conf_threshold})")

        if debug_data:
            debug_data.markers = markers

        if not markers:
            return ImageCalibrationResult(
                roi_id=roi_id,
                success=False,
                microscope_position=None,
                z_position=None,
                error_message="DETECTION: No markers found",
                debug_data=debug_data,
            )

        # 4. Match markers
        matching_result = matching_step(detection_result)
        matched_indices = matching_result["matched_marker_indices"]

        if verbose:
            print(f"    - Pairs matched: {len(matched_indices)}")

        if debug_data:
            debug_data.matched_indices = matched_indices

        if not matched_indices:
            return ImageCalibrationResult(
                roi_id=roi_id,
                success=False,
                microscope_position=None,
                z_position=None,
                error_message="MATCHING: No marker pairs matched",
                debug_data=debug_data,
            )

        # 5. Compute rotation angle from detected markers
        angles = compute_marker_group_angles(
            markers, matched_indices, marker_group_pixels, signed=True
        )

        # Check angle consistency across pairs
        if len(angles) >= 2:
            angle_range = max(angles) - min(angles)
            if angle_range > max_angle_deviation:
                return ImageCalibrationResult(
                    roi_id=roi_id,
                    success=False,
                    microscope_position=None,
                    z_position=None,
                    error_message=(
                        f"ANGLES: Inconsistent rotation angles "
                        f"(range={angle_range:.2f}° > {max_angle_deviation:.1f}°)"
                    ),
                    debug_data=debug_data,
                )

        rotation_angle = np.mean(angles)

        if verbose:
            print(f"    - Rotation angle: {rotation_angle:.2f}°")

        if debug_data:
            debug_data.rotation_angle = rotation_angle

        # 6. Filter matched pairs to keep only those with RoI fully in image bounds
        matched_indices = filter_matched_pairs_by_bounds(
            markers=markers,
            matched_indices=matched_indices,
            marker_group_pixels=marker_group_pixels,
            roi_polygon=roi_polygon,
            image_shape=image.shape[:2],
            rotation_angle=rotation_angle,
        )

        if verbose:
            print(f"    - Valid pairs (in bounds): {len(matched_indices)}")

        if not matched_indices:
            return ImageCalibrationResult(
                roi_id=roi_id,
                success=False,
                microscope_position=None,
                z_position=None,
                error_message="BOUNDS: No marker pairs with RoI fully in image bounds",
                debug_data=debug_data,
            )

        # Update debug data with filtered indices
        if debug_data:
            debug_data.matched_indices = matched_indices

        # 7. Compute chamber center in pixels (with rotation correction)
        chamber_center_pixels = compute_chamber_center(
            markers, matched_indices, marker_group_pixels, roi_polygon, rotation_angle
        )

        # 8. Convert to microns and compute microscope position
        chamber_center_microns = chamber_center_pixels * pixel_size
        microscope_x = stage_position["x"] + chamber_center_microns[0]
        microscope_y = stage_position["y"] + chamber_center_microns[1]
        z_position = stage_position.get("z", 0.0)

        microscope_position = np.array([microscope_x, microscope_y])

        if verbose:
            print(
                f"    - Chamber center (px): "
                f"({chamber_center_pixels[0]:.1f}, {chamber_center_pixels[1]:.1f})"
            )
            print(
                f"    - Stage position: "
                f"({stage_position['x']:.2f}, {stage_position['y']:.2f}, {z_position:.2f})"
            )
            print(f"    - Microscope position: ({microscope_x:.2f}, {microscope_y:.2f})")
            print("    - Status: SUCCESS")

        if debug_data:
            debug_data.chamber_center_pixels = chamber_center_pixels
            debug_data.chamber_center_microns = np.array([microscope_x, microscope_y])
            debug_data.pixel_size = pixel_size

        return ImageCalibrationResult(
            roi_id=roi_id,
            success=True,
            microscope_position=microscope_position,
            z_position=z_position,
            error_message=None,
            debug_data=debug_data,
        )

    except Exception as e:
        return ImageCalibrationResult(
            roi_id=roi_id,
            success=False,
            microscope_position=None,
            z_position=None,
            error_message=f"ERROR: {e!s}",
            debug_data=debug_data,
        )


class CalibrationError(ValueError):
    """Raised when calibration fails, with per-image results attached."""

    def __init__(self, message: str, image_results: list[ImageCalibrationResult]):
        super().__init__(message)
        self.image_results = image_results


def run_calibration(
    images: list[np.ndarray],
    roi_ids: list[str],
    stage_positions: list[dict[str, float]],
    detection_step,
    structure_library,
    blueprint_map: Map,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> CalibrationResult:
    """Run the full calibration pipeline on in-memory images.

    Args:
        images: List of HxWx3 uint8 image arrays
        roi_ids: List of RoI identifiers (e.g., ["0050", "0100", "7000"])
        stage_positions: List of stage position dicts with x, y, and optionally z
        detection_step: MarkerDetectionStep instance
        structure_library: ChipStructureLibrary or SAKRoIStructureLibrary
        blueprint_map: Blueprint map with design-coordinate positions
        pixel_size: Pixel size in microns
        verbose: Print progress information
        collect_debug: Collect debug data for visualizations
        conf_threshold: Minimum confidence for detected markers (default: 0.5)
        max_angle_deviation: Maximum allowed angle range in degrees (default: 5.0)

    Returns:
        CalibrationResult with calibrated map and statistics

    Raises:
        CalibrationError: If fewer than 3 images succeed (includes image_results)
    """
    if verbose:
        print("[Step 1/3] Processing calibration images")

    image_results: list[ImageCalibrationResult] = []

    for i, (image, roi_id, stage_position) in enumerate(
        zip(images, roi_ids, stage_positions, strict=True)
    ):
        roi_id = str(roi_id).zfill(4)  # Ensure 4-digit format

        if verbose:
            print(f"  Image {i + 1}/{len(images)}")
            print(f"    - RoI ID: {roi_id}")

        result = process_calibration_image(
            image=image,
            roi_id=roi_id,
            stage_position=stage_position,
            detection_step=detection_step,
            structure_library=structure_library,
            pixel_size=pixel_size,
            verbose=verbose,
            collect_debug=collect_debug,
            conf_threshold=conf_threshold,
            max_angle_deviation=max_angle_deviation,
        )

        image_results.append(result)

        if not result.success and verbose:
            print(f"    - Status: FAILED - {result.error_message}")

    if verbose:
        print()

    # Build measured map from successful calibrations
    successful_results = [r for r in image_results if r.success]

    if len(successful_results) < 3:
        raise CalibrationError(
            f"Need at least 3 successful calibration images for affine transform, "
            f"got {len(successful_results)}",
            image_results=image_results,
        )

    measured_positions = [
        RoIPosition(roi_id=r.roi_id, position=r.microscope_position) for r in successful_results
    ]
    measured_map = Map(measured_positions)

    # Store z positions for later
    z_positions = {r.roi_id: r.z_position for r in successful_results if r.z_position is not None}

    # Compute affine transform from blueprint to measured
    if verbose:
        print("[Step 2/3] Computing affine transform")
        print(f"  Calibration points: {len(successful_results)}")

    transform_result = blueprint_map.compute_affine_transform(measured_map)

    if verbose:
        print(f"  RMSE: {transform_result.rmse:.3f} microns")
        print(f"  Max error: {transform_result.max_error:.3f} microns")
        print()

    # Apply transform to full blueprint map
    if verbose:
        print("[Step 3/3] Applying transform")
        print(f"  Transformed {len(blueprint_map.roi_positions)} chamber positions")
        print()

    calibrated_map = blueprint_map.apply_transform(transform_result)

    if verbose:
        print("=== Calibration Complete ===")
        print()

    return CalibrationResult(
        measured_map=measured_map,
        transform_result=transform_result,
        calibrated_map=calibrated_map,
        image_results=image_results,
        z_positions=z_positions,
    )

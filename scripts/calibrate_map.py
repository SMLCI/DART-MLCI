#!/usr/bin/env python
"""Calibration script for map estimation from images.

Takes N calibration images with known microscope stage positions, detects markers,
computes chamber centers, and produces a Map in microscope coordinates using
the improved compute_affine_transform.

Example usage:
    python scripts/calibrate_map.py --config calibration.json --output calibrated_map.csv
    python scripts/calibrate_map.py --config calibration.json --output map.csv --stats stats.json --verbose

JSON config format:
    {
        "chip_name": "SAK",
        "calibration_images": [
            {
                "image_path": "/path/to/image1.tif",
                "roi_id": "0050",
                "stage_position": {"x": 5278.0, "y": -37408.0, "z": 100.0}
            }
        ],
        "pixel_size": 0.065789,
        "chip_config_path": "artifacts/chips/sak.json",
        "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt"
    }

    Note: Provide either "chip_config_path" (unified chip JSON with blueprint_map)
    or "blueprint_map_path" (CSV). chip_config_path is preferred as it also provides
    the structure library for chamber type lookup.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity

import dmc_masking
from dmc_masking import ChipStructureLibrary, MarkerDetectionStep, MarkerMatchingStep
from dmc_masking.io import load_image
from dmc_masking.map import AffineTransformResult, Map, RoIPosition
from dmc_masking.mask import RoIPolygon, SAKRoIStructureLibrary, apply_mask_rotation_free
from dmc_masking.rotation import compute_marker_group_angles


@dataclass
class ImageDebugData:
    """Debug data for per-image visualization."""

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
    rotation_angle: float | None = None  # Angle from markers (in degrees)


@dataclass
class ImageCalibrationResult:
    """Result of processing a single calibration image."""

    roi_id: str
    success: bool
    microscope_position: np.ndarray | None  # (x, y) in microns, z stored separately
    z_position: float | None
    error_message: str | None = None
    debug_data: ImageDebugData | None = None


@dataclass
class CalibrationResult:
    """Result of the full calibration process."""

    measured_map: Map
    transform_result: AffineTransformResult
    calibrated_map: Map
    image_results: list[ImageCalibrationResult]
    z_positions: dict[str, float] = field(default_factory=dict)


def load_config(path: Path) -> dict:
    """Load calibration configuration from JSON file.

    Args:
        path: Path to the JSON config file

    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        return json.load(f)


def validate_config(config: dict, config_path: Path | None = None) -> None:
    """Validate calibration configuration and raise helpful errors.

    Args:
        config: Configuration dictionary to validate
        config_path: Optional path to config file (for error messages)

    Raises:
        ValueError: If required fields are missing or invalid
    """
    source = f" in '{config_path}'" if config_path else ""

    # Check required top-level fields
    required_fields = ["calibration_images", "pixel_size"]
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(
            f"Missing required field(s){source}: {', '.join(missing_fields)}\n"
            f"Required fields are:\n"
            f"  - calibration_images: List of calibration image configurations\n"
            f"  - pixel_size: Pixel size in microns (e.g., 0.065789)\n"
            f"  - blueprint_map_path OR chip_config_path: Source for the blueprint map"
        )

    # Ensure at least one blueprint map source is provided
    if "blueprint_map_path" not in config and "chip_config_path" not in config:
        raise ValueError(
            f"Must provide either 'blueprint_map_path' (CSV) or 'chip_config_path' "
            f"(chip JSON with blueprint_map){source}"
        )

    # Validate calibration_images
    cal_images = config["calibration_images"]
    if not isinstance(cal_images, list):
        raise ValueError(
            f"'calibration_images'{source} must be a list, got {type(cal_images).__name__}"
        )

    if len(cal_images) < 3:
        raise ValueError(
            f"Need at least 3 calibration images for affine transform, "
            f"got {len(cal_images)}{source}"
        )

    # Validate each calibration image entry
    for i, img_config in enumerate(cal_images):
        prefix = f"calibration_images[{i}]{source}"

        if not isinstance(img_config, dict):
            raise ValueError(f"{prefix} must be a dictionary, got {type(img_config).__name__}")

        # Check required fields in each image config
        img_required = ["image_path", "roi_id", "stage_position"]
        img_missing = [f for f in img_required if f not in img_config]
        if img_missing:
            raise ValueError(
                f"{prefix} is missing required field(s): {', '.join(img_missing)}\n"
                f"Each calibration image entry must have:\n"
                f"  - image_path: Path to the calibration image file\n"
                f"  - roi_id: RoI identifier (e.g., '0050')\n"
                f"  - stage_position: Dict with 'x', 'y', and optionally 'z' coordinates"
            )

        # Validate stage_position
        stage_pos = img_config["stage_position"]
        if not isinstance(stage_pos, dict):
            raise ValueError(
                f"{prefix}.stage_position must be a dictionary with 'x' and 'y' keys, "
                f"got {type(stage_pos).__name__}"
            )

        stage_required = ["x", "y"]
        stage_missing = [f for f in stage_required if f not in stage_pos]
        if stage_missing:
            raise ValueError(
                f"{prefix}.stage_position is missing required field(s): {', '.join(stage_missing)}\n"
                f"stage_position must have 'x' and 'y' keys (and optionally 'z')"
            )

        # Check that image file exists
        image_path = Path(img_config["image_path"])
        if not image_path.exists():
            raise ValueError(f"{prefix}.image_path: File not found: {image_path}")

    # Validate pixel_size
    pixel_size = config["pixel_size"]
    if not isinstance(pixel_size, int | float) or pixel_size <= 0:
        raise ValueError(f"'pixel_size'{source} must be a positive number, got {pixel_size}")

    # Validate blueprint_map_path exists (if provided)
    if "blueprint_map_path" in config:
        blueprint_path = Path(config["blueprint_map_path"])
        if not blueprint_path.exists():
            raise ValueError(f"'blueprint_map_path'{source}: File not found: {blueprint_path}")

    # Validate chip_config_path exists (if provided)
    if "chip_config_path" in config and config["chip_config_path"] is not None:
        chip_config_path = Path(config["chip_config_path"])
        if not chip_config_path.exists():
            raise ValueError(f"'chip_config_path'{source}: File not found: {chip_config_path}")

    # Validate optional model_path if provided
    if "model_path" in config and config["model_path"] is not None:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise ValueError(f"'model_path'{source}: File not found: {model_path}")


def compute_chamber_center(
    markers: list[dict],
    matched_indices: list[tuple[int, int]],
    marker_group_pixels: dict[str, np.ndarray],
    roi_polygon: RoIPolygon,
    rotation_angle: float = 0.0,
) -> np.ndarray:
    """Compute the chamber center in pixel coordinates.

    The chamber center is computed by finding the offset from the cross marker
    to the polygon center in blueprint coordinates, then rotating that offset
    by the inverse of the detected rotation angle to account for the fact that
    calibration images are not rotated (unlike production images).

    Args:
        markers: List of detected markers with bbox_center
        matched_indices: List of (cross_idx, circle_idx) tuples
        marker_group_pixels: Expected marker positions in pixels
        roi_polygon: RoI polygon for getting centroid
        rotation_angle: Rotation angle in degrees from markers (default: 0.0)

    Returns:
        Chamber center position in pixels as (x, y) array
    """
    if not matched_indices:
        raise ValueError("No matched marker pairs found")

    # Get polygon centroid in the polygon's local coordinate system (after translation to 0,0)
    polygon_center = roi_polygon.center  # np.array([cx, cy])

    # The cross_local from marker_group_pixels is in the ORIGINAL coordinate system
    # (only scaled, not translated). But the polygon has been translated so its
    # bounds start at (0,0). The marker positions are defined relative to the
    # polygon's original origin, so we need to use them directly as the offset
    # from (0,0) in the polygon's local coordinate system.
    cross_local = marker_group_pixels["cross"]

    # The offset from cross to polygon center in the polygon's local coordinate system.
    # Since the polygon is translated to start at (0,0), and cross_local represents
    # the cross position relative to the polygon's origin, the offset is simply:
    # center_offset = polygon_center - cross_local
    center_offset = np.array(
        [
            polygon_center[0] - cross_local[0],
            polygon_center[1] + cross_local[1],  # Changed: - → +
        ]
    )

    # Apply rotation to the offset to account for image orientation.
    # The calibration images are NOT rotated, so we need to rotate the offset
    # by the same angle that the markers have been rotated from blueprint.
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
    cross_idx, circle_idx = matched_indices[0]
    cross_detected = markers[cross_idx]["bbox_center"]

    # Chamber center = detected cross + rotated offset to center
    return cross_detected + rotated_offset


def filter_matched_pairs_by_bounds(
    markers: list[dict],
    matched_indices: list[tuple[int, int]],
    marker_group_pixels: dict[str, np.ndarray],
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


def crop_calibration_image(
    image: np.ndarray,
    markers: list[dict],
    matched_indices: list[tuple[int, int]],
    marker_group_pixels: dict[str, np.ndarray],
    roi_polygon: RoIPolygon,
    rotation_angle: float,
    return_uncropped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a calibration image using the rotation-free masking approach.

    This function applies masking and cropping without rotating the entire image.
    Instead, it rotates the RoI polygon to match the detected orientation and
    applies it directly to the unrotated image.

    Args:
        image: The unrotated image, shape (H, W) or (C, H, W)
        markers: List of detected marker dicts with bbox_center
        matched_indices: List of (cross_idx, circle_idx) tuples
        marker_group_pixels: Expected marker positions in pixels
        roi_polygon: RoI polygon template
        rotation_angle: Detected rotation angle in degrees
        return_uncropped: If True, return full image and mask without cropping

    Returns:
        Tuple of (cropped_image, cropped_mask)

    Raises:
        ValueError: If no RoI polygon fits within image bounds
    """
    return apply_mask_rotation_free(
        matched_marker_indices=matched_indices,
        markers=markers,
        marker_group_pixels=marker_group_pixels,
        roi_polygon=roi_polygon,
        image=image,
        rotation_angle=rotation_angle,
        return_uncropped=return_uncropped,
    )


def process_calibration_image(
    image_path: Path,
    roi_id: str,
    stage_position: dict[str, float],
    detection_step: MarkerDetectionStep,
    structure_library: SAKRoIStructureLibrary,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> ImageCalibrationResult:
    """Process a single calibration image.

    Args:
        image_path: Path to the image file
        roi_id: RoI identifier (e.g., "0050")
        stage_position: Stage position dict with x, y, and optionally z
        detection_step: Marker detection step
        structure_library: SAK structure library for chamber type lookup
        pixel_size: Pixel size in microns
        verbose: Print progress information
        collect_debug: Collect debug data for visualization
        conf_threshold: Minimum confidence for detected markers (default: 0.5)
        max_angle_deviation: Maximum allowed range (in degrees) between rotation angles
            from different marker pairs. If exceeded, the image is rejected. (default: 5.0)

    Returns:
        ImageCalibrationResult with microscope position or error
    """
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

        # 2. Create matching step for this chamber type
        matching_step = MarkerMatchingStep(marker_group_pixels, tolerance=60)

        # 3. Load and process image
        image = load_image(image_path)

        if debug_data:
            debug_data.image = image

        # 4. Detect markers
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

        # 5. Match markers
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

        # 5b. Compute rotation angle from detected markers (needed for bounds check)
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
                    error_message=f"ANGLES: Inconsistent rotation angles (range={angle_range:.2f}° > {max_angle_deviation:.1f}°)",
                    debug_data=debug_data,
                )

        rotation_angle = np.mean(angles)

        if verbose:
            print(f"    - Rotation angle: {rotation_angle:.2f}°")

        if debug_data:
            debug_data.rotation_angle = rotation_angle

        # 5c. Filter matched pairs to keep only those with RoI fully in image bounds
        # Pass rotation_angle so the polygon is positioned correctly (matching apply_mask_rotation_free)
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

        # 6. Compute chamber center in pixels (with rotation correction)
        chamber_center_pixels = compute_chamber_center(
            markers, matched_indices, marker_group_pixels, roi_polygon, rotation_angle
        )

        # 8. Convert to microns
        chamber_center_microns = chamber_center_pixels * pixel_size

        # 9. Compute microscope position (stage is at top-left corner of image)
        microscope_x = stage_position["x"] + chamber_center_microns[0]
        microscope_y = stage_position["y"] + chamber_center_microns[1]
        z_position = stage_position.get("z", 0.0)

        microscope_position = np.array([microscope_x, microscope_y])

        if verbose:
            print(
                f"    - Chamber center (px): ({chamber_center_pixels[0]:.1f}, {chamber_center_pixels[1]:.1f})"
            )
            print(
                f"    - Stage position: ({stage_position['x']:.2f}, {stage_position['y']:.2f}, {z_position:.2f})"
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


def run_calibration(
    config: dict,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> tuple[CalibrationResult, Map]:
    """Run the full calibration pipeline.

    Args:
        config: Configuration dictionary with calibration_images, pixel_size, etc.
        verbose: Print progress information
        collect_debug: Collect debug data for visualizations

    Returns:
        Tuple of (CalibrationResult with calibrated map and statistics, blueprint_map)
    """
    pixel_size = config["pixel_size"]
    blueprint_map_path = config.get("blueprint_map_path")
    model_path = config.get("model_path")

    # Set default model path if not specified
    if model_path is None:
        model_path = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/v26_detect_s_imgsz1280.pt"
        )
    else:
        model_path = Path(model_path)

    # Set default structure library or chip config path
    chip_config_path = config.get("chip_config_path")
    structure_library_path = config.get("structure_library_path")
    if chip_config_path is None and structure_library_path is None:
        structure_library_path = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/chamber_structure.json"
        )
    elif structure_library_path is not None:
        structure_library_path = Path(structure_library_path)

    device = config.get("device")
    calibration_images = config["calibration_images"]

    if verbose:
        print("=== Calibration Pipeline ===")
        print()
        print("[Step 1/4] Loading configuration")

    # Load blueprint map: prefer chip_config_path, fall back to blueprint_map_path CSV
    if chip_config_path is not None:
        structure_library = ChipStructureLibrary.from_file(chip_config_path, pixel_size=pixel_size)
        blueprint_map = structure_library.get_blueprint_map()
        blueprint_map_source = chip_config_path
    elif blueprint_map_path is not None:
        blueprint_map = Map.from_csv(Path(blueprint_map_path))
        blueprint_map_source = blueprint_map_path
    else:
        raise ValueError("Must provide either 'chip_config_path' or 'blueprint_map_path'")

    if verbose:
        print(f"  Blueprint map: {blueprint_map_source} ", end="")

    if verbose:
        print(f"({len(blueprint_map.roi_positions)} chambers)")
        print(f"  Model: {model_path}")
        print(f"  Pixel size: {pixel_size} microns")
        print(f"  Calibration images: {len(calibration_images)}")
        print()

    # Initialize structure library if not already loaded from chip config
    if chip_config_path is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            structure_library = SAKRoIStructureLibrary(
                lookup_path=structure_library_path,
                pixel_size=pixel_size,
            )

    detection_step = MarkerDetectionStep(str(model_path), device=device, verbose=False)

    # Process each calibration image
    if verbose:
        print("[Step 2/4] Processing calibration images")

    image_results: list[ImageCalibrationResult] = []

    for i, img_config in enumerate(calibration_images):
        image_path = Path(img_config["image_path"])
        roi_id = str(img_config["roi_id"]).zfill(4)  # Ensure 4-digit format
        stage_position = img_config["stage_position"]

        if verbose:
            print(f"  Image {i + 1}/{len(calibration_images)}: {image_path.name}")
            print(f"    - RoI ID: {roi_id}")

        result = process_calibration_image(
            image_path=image_path,
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
        raise ValueError(
            f"Need at least 3 successful calibration images for affine transform, "
            f"got {len(successful_results)}"
        )

    measured_positions = [
        RoIPosition(roi_id=r.roi_id, position=r.microscope_position) for r in successful_results
    ]
    measured_map = Map(measured_positions)

    # Store z positions for later
    z_positions = {r.roi_id: r.z_position for r in successful_results if r.z_position is not None}

    # Compute affine transform from blueprint to measured
    if verbose:
        print("[Step 3/4] Computing affine transform")
        print(f"  Calibration points: {len(successful_results)}")

    transform_result = blueprint_map.compute_affine_transform(measured_map)

    if verbose:
        print(f"  RMSE: {transform_result.rmse:.3f} microns")
        print(f"  Max error: {transform_result.max_error:.3f} microns")
        print()
        print("  Per-point residuals:")
        for i, r in enumerate(successful_results):
            print(f"    {r.roi_id}: {transform_result.residuals[i]:.2f} microns")
        print()

    # Apply transform to full blueprint map
    if verbose:
        print("[Step 4/4] Applying transform")
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
    ), blueprint_map


def save_calibrated_map(calibrated_map: Map, z_positions: dict[str, float], output_path: Path):
    """Save calibrated map to CSV file.

    Args:
        calibrated_map: The calibrated Map object
        z_positions: Dictionary mapping roi_id to z position
        output_path: Path to output CSV file
    """
    df = calibrated_map.to_df()

    # Add z column - use average z from calibration points, or 0 if none
    if z_positions:
        avg_z = np.mean(list(z_positions.values()))
        df["z"] = df["roi_id"].map(lambda rid: z_positions.get(rid, avg_z))
    else:
        df["z"] = 0.0

    df.to_csv(output_path, index=False)


def save_stats(
    result: CalibrationResult,
    stats_path: Path,
):
    """Save calibration statistics to JSON file.

    Args:
        result: CalibrationResult with transform stats and image results
        stats_path: Path to output JSON file
    """
    # Build residuals dict
    successful_results = [r for r in result.image_results if r.success]
    residuals = {
        r.roi_id: float(result.transform_result.residuals[i])
        for i, r in enumerate(successful_results)
    }

    # Build failed images list
    failed_images = [
        {"roi_id": r.roi_id, "error": r.error_message}
        for r in result.image_results
        if not r.success
    ]

    stats = {
        "transform_stats": {
            "rmse": float(result.transform_result.rmse),
            "max_error": float(result.transform_result.max_error),
            "n_calibration_points": len(successful_results),
            "residuals": residuals,
        },
        "failed_images": failed_images,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def save_intermediate_outputs(
    result: CalibrationResult,
    blueprint_map: Map,
    output_dir: Path,
) -> None:
    """Save measured positions CSV and transform parameters JSON.

    Args:
        result: CalibrationResult with transform and image results
        blueprint_map: Original blueprint map
        output_dir: Directory for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get successful results for residual lookup
    successful_results = [r for r in result.image_results if r.success]

    # Save measured_positions.csv
    rows = []
    for i, img_result in enumerate(successful_results):
        blueprint_pos = blueprint_map.roi_positions[img_result.roi_id].position
        measured_pos = img_result.microscope_position
        residual = result.transform_result.residuals[i]
        rows.append(
            {
                "roi_id": img_result.roi_id,
                "blueprint_x": float(blueprint_pos[0]),
                "blueprint_y": float(blueprint_pos[1]),
                "measured_x": float(measured_pos[0]),
                "measured_y": float(measured_pos[1]),
                "residual": float(residual),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "measured_positions.csv", index=False)

    # Save transform_params.json
    # We need to reconstruct the affine matrix from the transform function
    # The transform function uses: x @ Ab[:2, :] + Ab[2:3, :]
    # So we need to get Ab by applying the transform to basis vectors
    transform_fn = result.transform_result.transform

    # Get the affine matrix coefficients by transforming basis points
    origin = np.array([0.0, 0.0])
    x_unit = np.array([1.0, 0.0])
    y_unit = np.array([0.0, 1.0])

    t_origin = transform_fn(origin)
    t_x = transform_fn(x_unit)
    t_y = transform_fn(y_unit)

    # Reconstruct affine matrix: [[a, b, tx], [c, d, ty]]
    # where [a, c] is the transformed x-unit minus origin
    # and [b, d] is the transformed y-unit minus origin
    a, c = t_x - t_origin
    b, d = t_y - t_origin
    tx, ty = t_origin

    affine_matrix = [[float(a), float(b), float(tx)], [float(c), float(d), float(ty)]]

    params = {
        "affine_matrix": affine_matrix,
        "rmse": float(result.transform_result.rmse),
        "max_error": float(result.transform_result.max_error),
        "n_points": len(successful_results),
    }

    with open(output_dir / "transform_params.json", "w") as f:
        json.dump(params, f, indent=2)


def plot_calibration_result(
    blueprint_map: Map,
    calibrated_map: Map,
    measured_map: Map,
    transform_result: AffineTransformResult,
    output_path: Path,
) -> None:
    """Generate side-by-side visualization of calibration.

    Args:
        blueprint_map: Original blueprint map
        calibrated_map: Transformed map in microscope coordinates
        measured_map: Map of measured calibration points
        transform_result: Affine transform result with residuals
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get calibration point IDs
    calibration_ids = set(measured_map.roi_positions.keys())

    # Left panel: Blueprint map
    ax1.set_title("Blueprint Map (Design Coordinates)", fontsize=14)

    # Plot all chambers as gray + markers
    blueprint_df = blueprint_map.to_df()
    ax1.scatter(
        blueprint_df["x"],
        blueprint_df["y"],
        c="gray",
        s=20,
        alpha=0.5,
        marker="+",
        label="All chambers",
    )

    # Highlight calibration chambers with distinct colors per ROI
    cal_blueprint = blueprint_df[blueprint_df["roi_id"].isin(calibration_ids)]
    cal_colors = ["#377eb8", "#ff7f00", "#984ea3", "#e41a1c", "#4daf4a"]
    roi_color_map = {
        roi_id: cal_colors[i % len(cal_colors)] for i, roi_id in enumerate(sorted(calibration_ids))
    }
    for _, row in cal_blueprint.iterrows():
        color = roi_color_map[row["roi_id"]]
        ax1.scatter(
            row["x"],
            row["y"],
            c=color,
            s=100,
            marker="+",
            linewidths=2,
            label=row["roi_id"],
            zorder=5,
        )
        ax1.annotate(
            row["roi_id"],
            (row["x"], row["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax1.set_xlabel(r"X ($\mu m$)")
    ax1.set_ylabel(r"Y ($\mu m$)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right panel: Calibrated map
    ax2.set_title("Calibrated Map (Microscope Coordinates)", fontsize=14)

    # Plot all chambers as gray + markers
    calibrated_df = calibrated_map.to_df()
    ax2.scatter(
        calibrated_df["x"],
        calibrated_df["y"],
        c="gray",
        s=20,
        alpha=0.5,
        marker="+",
        label="All chambers",
    )

    # Plot calibration points (measured positions) with distinct colors per ROI
    measured_df = measured_map.to_df()
    for _, row in measured_df.iterrows():
        color = roi_color_map[row["roi_id"]]
        ax2.scatter(
            row["x"],
            row["y"],
            c=color,
            s=100,
            marker="+",
            linewidths=2,
            label=row["roi_id"],
            zorder=5,
        )

    # Draw residual arrows from transformed blueprint to measured
    for roi_id in measured_map.roi_positions:
        # Get transformed blueprint position
        blueprint_pos = blueprint_map.roi_positions[roi_id].position
        transformed_pos = transform_result.transform(blueprint_pos)
        measured_pos = measured_map.roi_positions[roi_id].position

        # Draw arrow from transformed to measured (residual)
        dx = measured_pos[0] - transformed_pos[0]
        dy = measured_pos[1] - transformed_pos[1]
        if np.sqrt(dx**2 + dy**2) > 0.1:  # Only draw if residual is visible
            ax2.annotate(
                "",
                xy=(measured_pos[0], measured_pos[1]),
                xytext=(transformed_pos[0], transformed_pos[1]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
                zorder=4,
            )

    # Add labels for calibration points
    for _, row in measured_df.iterrows():
        color = roi_color_map[row["roi_id"]]
        ax2.annotate(
            row["roi_id"],
            (row["x"], row["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax2.set_xlabel(r"X ($\mu m$)")
    ax2.set_ylabel(r"Y ($\mu m$)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Decompose affine transform parameters
    origin = transform_result.transform(np.array([0.0, 0.0]))
    e1 = transform_result.transform(np.array([1.0, 0.0])) - origin
    e2 = transform_result.transform(np.array([0.0, 1.0])) - origin
    A = np.column_stack([e1, e2])  # 2x2 linear part
    translation = origin

    rotation_rad = np.arctan2(A[1, 0], A[0, 0])
    rotation_deg = np.degrees(rotation_rad)
    sx = np.linalg.norm(A[:, 0])
    sy = np.linalg.norm(A[:, 1])

    fig.suptitle(
        f"Rotation: {rotation_deg:.2f}° | Scale: ({sx:.4f}, {sy:.4f})"
        f" | Translation: ({translation[0]:.1f}, {translation[1]:.1f}) $\\mu m$\n"
        f"Matrix: [[{A[0,0]:.4f}, {A[0,1]:.4f}], [{A[1,0]:.4f}, {A[1,1]:.4f}]]"
        f" | RMSE = {transform_result.rmse:.3f} $\\mu m$,"
        f" Max Error = {transform_result.max_error:.3f} $\\mu m$",
        fontsize=11,
        y=1.02,
    )

    # Single shared legend below both panels (deduplicated)
    handles, labels = ax1.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, lbl in zip(handles, labels, strict=False):
        if lbl not in seen:
            seen[lbl] = True
            unique_handles.append(h)
            unique_labels.append(lbl)
    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        ncol=min(len(unique_labels), 6),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_image_debug(
    debug_data: ImageDebugData,
    roi_id: str,
    output_path: Path,
) -> None:
    """Generate per-image debug visualization with markers and positions.

    Args:
        debug_data: ImageDebugData containing image, markers, and positions
        roi_id: RoI identifier for the title
        output_path: Path to save the debug image
    """
    if debug_data.image is None:
        return

    # Convert BGR to RGB for matplotlib
    image = debug_data.image
    if image.ndim == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image_rgb)

    # Draw markers if available
    if debug_data.markers:
        colors = {"cross": "red", "circle": "blue"}
        marker_symbols = {"cross": "x", "circle": "o"}

        for i, marker in enumerate(debug_data.markers):
            center = marker["bbox_center"]
            label = marker["label"]
            conf = marker.get("conf", 0.0)

            color = colors.get(label, "green")
            symbol = marker_symbols.get(label, "s")

            ax.scatter(center[0], center[1], c=color, marker=symbol, s=200, linewidths=3, zorder=5)
            ax.annotate(
                f"{i}: {label} ({conf:.2f})",
                (center[0], center[1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color=color,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            )

    # Draw matched pairs
    if debug_data.matched_indices and debug_data.markers:
        for cross_idx, circle_idx in debug_data.matched_indices:
            cross_center = debug_data.markers[cross_idx]["bbox_center"]
            circle_center = debug_data.markers[circle_idx]["bbox_center"]
            ax.plot(
                [cross_center[0], circle_center[0]],
                [cross_center[1], circle_center[1]],
                "g-",
                linewidth=2,
                alpha=0.7,
            )

    # Draw rotated RoI polygon
    if (
        debug_data.roi_polygon is not None
        and debug_data.marker_group_pixels is not None
        and debug_data.rotation_angle is not None
        and debug_data.matched_indices
        and debug_data.markers
        and debug_data.chamber_center_pixels is not None
    ):
        # Get the polygon from the RoIPolygon wrapper
        polygon = debug_data.roi_polygon.roi_polygon

        # Get the polygon centroid in local coordinates
        polygon_center = debug_data.roi_polygon.center

        # Step 1: Rotate the polygon around its centroid
        # This aligns the blueprint polygon with the actual image orientation
        rotated_polygon = affinity.rotate(
            polygon, debug_data.rotation_angle, origin=(polygon_center[0], polygon_center[1])
        )

        # Step 2: Translate so the polygon centroid aligns with the computed chamber center
        # The chamber_center_pixels was computed with the correct rotated offset
        dx = debug_data.chamber_center_pixels[0] - polygon_center[0]
        dy = debug_data.chamber_center_pixels[1] - polygon_center[1]
        translated_polygon = affinity.translate(rotated_polygon, xoff=dx, yoff=dy)

        # Extract vertices and draw as matplotlib Polygon
        coords = np.array(translated_polygon.exterior.coords)
        poly_patch = MplPolygon(
            coords,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label="RoI polygon",
        )
        ax.add_patch(poly_patch)

    # Draw chamber center
    if debug_data.chamber_center_pixels is not None:
        ax.scatter(
            debug_data.chamber_center_pixels[0],
            debug_data.chamber_center_pixels[1],
            c="gold",
            marker="*",
            s=400,
            edgecolors="black",
            linewidths=1,
            zorder=10,
            label="Chamber center",
        )

    # Add microscope position label near the star
    # Microscope position = stage_position + center_pixels * pixel_size
    if (
        debug_data.stage_position is not None
        and debug_data.chamber_center_pixels is not None
        and debug_data.pixel_size is not None
    ):
        microscope_x = (
            debug_data.stage_position["x"]
            + debug_data.chamber_center_pixels[0] * debug_data.pixel_size
        )
        microscope_y = (
            debug_data.stage_position["y"]
            + debug_data.chamber_center_pixels[1] * debug_data.pixel_size
        )
        ax.annotate(
            f"({microscope_x:.1f}, {microscope_y:.1f}) um",
            xy=(debug_data.chamber_center_pixels[0], debug_data.chamber_center_pixels[1]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=9,
            color="gold",
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.7, pad=2),
            zorder=11,
        )

    # Add text annotations in a single box
    info_lines = []
    if debug_data.stage_position:
        info_lines.append(
            f"Stage: ({debug_data.stage_position['x']:.2f}, {debug_data.stage_position['y']:.2f}) µm"
        )
    if debug_data.chamber_center_pixels is not None:
        info_lines.append(
            f"Center (px): ({debug_data.chamber_center_pixels[0]:.1f}, {debug_data.chamber_center_pixels[1]:.1f})"
        )
    if debug_data.rotation_angle is not None:
        info_lines.append(f"Rotation: {debug_data.rotation_angle:.2f}°")

    if info_lines:
        ax.text(
            10,
            30,
            "\n".join(info_lines),
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 5},
            verticalalignment="top",
        )

    # Title
    title = f"RoI {roi_id}"
    if debug_data.structure_name:
        title += f" - {debug_data.structure_name}"
    ax.set_title(title, fontsize=14)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def calibrate_map(
    config: dict | Path | str,
    output_path: Path | str | None = None,
    stats_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[CalibrationResult, Map]:
    """Calibrate a map from calibration images with known stage positions.

    This is the main programmatic entry point for the calibration pipeline.

    Args:
        config: Either a configuration dictionary, or a path to a JSON config file.
            Required keys in the dict:
            - calibration_images: List of dicts with image_path, roi_id, stage_position
            - pixel_size: Pixel size in microns
            - blueprint_map_path: Path to the blueprint map CSV
            Optional keys:
            - model_path: Path to the detection model
            - structure_library_path: Path to the structure library JSON
            - device: Device to run on (can also be passed as argument)
        output_path: Path to save the calibrated map CSV. If None, map is not saved.
        stats_path: Path to save calibration statistics JSON. If None, stats not saved.
        output_dir: Directory for intermediate outputs and visualizations.
            If provided, saves measured_positions.csv, transform_params.json,
            calibration_plot.png, and per-image debug images.
        device: Device to run on (e.g., 'cuda:0', 'cpu'). Overrides config if provided.
        verbose: Print detailed progress information.

    Returns:
        Tuple of (CalibrationResult, blueprint_map)

    Example:
        >>> config = {
        ...     "calibration_images": [
        ...         {"image_path": "img1.tif", "roi_id": "0050",
        ...          "stage_position": {"x": 100.0, "y": 200.0, "z": 0.0}},
        ...         # ... more images (at least 3 required)
        ...     ],
        ...     "pixel_size": 0.065789,
        ...     "blueprint_map_path": "artifacts/sak_blueprint_map.csv",
        ... }
        >>> result, blueprint = calibrate_map(config, output_path="calibrated.csv", verbose=True)
        >>> print(f"RMSE: {result.transform_result.rmse:.3f} microns")
    """
    # Load config if path is provided
    config_path = None
    if isinstance(config, str | Path):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_config(config_path)

    # Resolve relative image paths against config file directory
    if config_path is not None:
        config_dir = config_path.resolve().parent
        for img_config in config.get("calibration_images", []):
            img_path = Path(img_config["image_path"])
            if not img_path.is_absolute():
                img_config["image_path"] = str(config_dir / img_path)

    # Validate configuration
    validate_config(config, config_path)

    # Override device if specified
    if device:
        config["device"] = device

    # Convert paths to Path objects
    if output_path is not None:
        output_path = Path(output_path)
    if stats_path is not None:
        stats_path = Path(stats_path)
    if output_dir is not None:
        output_dir = Path(output_dir)

    # Get conf_threshold and max_angle_deviation from config, can be overridden by caller
    conf_threshold = config.get("conf_threshold", 0.5)
    max_angle_deviation = config.get("max_angle_deviation", 5.0)

    # Collect debug data if output_dir is specified
    collect_debug = output_dir is not None

    # Run calibration
    result, blueprint_map = run_calibration(
        config,
        verbose=verbose,
        collect_debug=collect_debug,
        conf_threshold=conf_threshold,
        max_angle_deviation=max_angle_deviation,
    )

    # Save calibrated map if requested
    if output_path is not None:
        save_calibrated_map(result.calibrated_map, result.z_positions, output_path)
        if verbose:
            print(f"Calibrated map saved to: {output_path}")

    # Save stats if requested
    if stats_path is not None:
        save_stats(result, stats_path)
        if verbose:
            print(f"Calibration stats saved to: {stats_path}")

    # Save intermediate outputs and visualizations if output_dir is specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save intermediate outputs (CSV and JSON)
        save_intermediate_outputs(result, blueprint_map, output_dir)
        if verbose:
            print(f"Intermediate outputs saved to: {output_dir}")

        # Generate calibration plot
        plot_calibration_result(
            blueprint_map=blueprint_map,
            calibrated_map=result.calibrated_map,
            measured_map=result.measured_map,
            transform_result=result.transform_result,
            output_path=output_dir / "calibration_plot.png",
        )
        if verbose:
            print(f"Calibration plot saved to: {output_dir / 'calibration_plot.png'}")

        # Generate per-image debug visualizations
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for img_result in result.image_results:
            if img_result.debug_data:
                plot_image_debug(
                    debug_data=img_result.debug_data,
                    roi_id=img_result.roi_id,
                    output_path=images_dir / f"{img_result.roi_id}_debug.png",
                )
        if verbose:
            print(f"Per-image debug plots saved to: {images_dir}")

    return result, blueprint_map


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a map from calibration images with known stage positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate_map.py --config calibration.json --output calibrated_map.csv
  python scripts/calibrate_map.py --config calibration.json --output map.csv --stats stats.json --verbose
  python scripts/calibrate_map.py --config calibration.json --output map.csv --output-dir ./debug --verbose

JSON config format:
  {
      "chip_name": "SAK",
      "calibration_images": [
          {
              "image_path": "/path/to/image1.tif",
              "roi_id": "0050",
              "stage_position": {"x": 5278.0, "y": -37408.0, "z": 100.0}
          }
      ],
      "pixel_size": 0.065789,
      "chip_config_path": "artifacts/chips/sak.json",
      "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt"
  }

  Note: Provide either "chip_config_path" (unified chip JSON) or
  "blueprint_map_path" (CSV). chip_config_path is preferred.
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--chip-config",
        type=Path,
        default=None,
        help="Path to unified chip config JSON file (overrides structure_library_path in config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file for calibrated map",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to output JSON file for calibration statistics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate outputs and visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default: auto",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Minimum detection confidence threshold (default: 0.5, overrides config)",
    )
    parser.add_argument(
        "--max-angle-deviation",
        type=float,
        default=None,
        help="Maximum allowed rotation angle range across marker pairs in degrees (default: 5.0, overrides config)",
    )

    args = parser.parse_args()

    try:
        # If CLI overrides are provided, inject into the config
        config_input = args.config
        if (
            args.chip_config is not None
            or args.conf_threshold is not None
            or args.max_angle_deviation is not None
        ):
            # Load the config file, add overrides, pass as dict
            config_path = Path(args.config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if isinstance(config_input, Path):
                config_input = load_config(config_path)
            if args.chip_config is not None:
                config_input["chip_config_path"] = str(args.chip_config)
            if args.conf_threshold is not None:
                config_input["conf_threshold"] = args.conf_threshold
            if args.max_angle_deviation is not None:
                config_input["max_angle_deviation"] = args.max_angle_deviation

        # Run calibration using the function API
        result, _ = calibrate_map(
            config=config_input,
            output_path=args.output,
            stats_path=args.stats,
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
        )

        # Print summary (always print, even if not verbose)
        n_success = sum(1 for r in result.image_results if r.success)
        n_total = len(result.image_results)
        print(f"\nCalibration complete: {n_success}/{n_total} images successful")
        print(f"RMSE: {result.transform_result.rmse:.3f} microns")
        print(f"Max error: {result.transform_result.max_error:.3f} microns")

        # Report failures
        failed = [r for r in result.image_results if not r.success]
        if failed:
            print(f"\nFailed images ({len(failed)}):")
            for r in failed:
                print(f"  {r.roi_id}: {r.error_message}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

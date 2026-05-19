"""Validation pipeline for comparing computed chamber positions to calibrated map.

Provides ``process_validation_image()`` and ``run_validation()`` that reuse the
same marker detection, matching, bounds filtering, and center computation logic
as the calibration pipeline but add error-metric computation against expected
positions from a calibrated map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dart_mlci.io import load_image
from dart_mlci.map import Map
from dart_mlci.mask import RoIPolygon
from dart_mlci.pipeline import MarkerDetectionStep, MarkerMatchingStep
from dart_mlci.rotation import compute_marker_group_angles

from .core import compute_chamber_center, filter_matched_pairs_by_bounds


@dataclass
class ValidationDebugData:
    """Debug data for per-image validation visualization."""

    image: np.ndarray | None = None
    markers: list[dict] | None = None
    matched_indices: list[tuple[int, int]] | None = None
    chamber_center_pixels: np.ndarray | None = None
    expected_center_pixels: np.ndarray | None = None
    stage_position: dict[str, float] | None = None
    pixel_size: float | None = None
    structure_name: str | None = None
    roi_polygon: RoIPolygon | None = None
    marker_group_pixels: dict[str, np.ndarray] | None = None
    rotation_angle: float | None = None
    error_microns: float | None = None


@dataclass
class ValidationResult:
    """Result of validating a single image."""

    roi_id: str
    success: bool
    map_x: float | None
    map_y: float | None
    measured_x: float | None
    measured_y: float | None
    error: float | None
    error_message: str | None = None
    debug_data: ValidationDebugData | None = None
    error_px: float | None = None


@dataclass
class ValidationSummary:
    """Summary of validation results."""

    results: list[ValidationResult]
    mean_error: float
    median_error: float
    std_error: float
    max_error: float
    min_error: float
    p90_error: float
    n_success: int
    n_failed: int

    def to_csv(self, output_path: Path, pixel_size: float) -> None:
        """Write per-image validation results to CSV (computes error_px from pixel_size)."""
        rows = []
        for r in self.results:
            error_px = r.error / pixel_size if r.error is not None else None
            r.error_px = error_px
            rows.append(
                {
                    "roi_id": r.roi_id,
                    "map_x": r.map_x,
                    "map_y": r.map_y,
                    "measured_x": r.measured_x,
                    "measured_y": r.measured_y,
                    "error": r.error,
                    "error_px": error_px,
                    "success": r.success,
                    "error_message": r.error_message,
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False)

    @staticmethod
    def from_csv(csv_path: Path) -> ValidationSummary:
        """Reconstruct a ValidationSummary from a previously saved CSV."""
        df = pd.read_csv(csv_path)
        results = []
        for _, row in df.iterrows():
            results.append(
                ValidationResult(
                    roi_id=str(row["roi_id"]),
                    success=bool(row["success"]),
                    map_x=row["map_x"] if pd.notna(row["map_x"]) else None,
                    map_y=row["map_y"] if pd.notna(row["map_y"]) else None,
                    measured_x=row["measured_x"] if pd.notna(row["measured_x"]) else None,
                    measured_y=row["measured_y"] if pd.notna(row["measured_y"]) else None,
                    error=row["error"] if pd.notna(row["error"]) else None,
                    error_message=(
                        row["error_message"] if pd.notna(row["error_message"]) else None
                    ),
                    error_px=(
                        row["error_px"]
                        if "error_px" in df.columns and pd.notna(row["error_px"])
                        else None
                    ),
                )
            )

        successful = [r for r in results if r.success and r.error is not None]
        errors = [r.error for r in successful]

        return ValidationSummary(
            results=results,
            mean_error=float(np.mean(errors)) if errors else 0.0,
            median_error=float(np.median(errors)) if errors else 0.0,
            std_error=float(np.std(errors)) if errors else 0.0,
            max_error=float(np.max(errors)) if errors else 0.0,
            min_error=float(np.min(errors)) if errors else 0.0,
            p90_error=float(np.percentile(errors, 90)) if errors else 0.0,
            n_success=len(successful),
            n_failed=len(results) - len(successful),
        )


def process_validation_image(
    image_path: Path,
    roi_id: str,
    stage_position: dict[str, float],
    expected_position: np.ndarray,
    detection_step: MarkerDetectionStep,
    structure_library,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> ValidationResult:
    """Process a single validation image and compute error against expected position.

    Args:
        image_path: Path to the image file.
        roi_id: RoI identifier (e.g., "0050").
        stage_position: Stage position dict with x, y (and optionally z).
        expected_position: Expected position from calibrated map (x, y) in microns.
        detection_step: Marker detection step.
        structure_library: ChipStructureLibrary or SAKRoIStructureLibrary.
        pixel_size: Pixel size in microns.
        verbose: Print progress information.
        collect_debug: Collect debug data for visualization.
        conf_threshold: Minimum confidence for detected markers.
        max_angle_deviation: Maximum allowable angle range across marker pairs.

    Returns:
        ValidationResult with error metrics or failure reason.
    """
    debug_data = ValidationDebugData() if collect_debug else None

    try:
        # 1. Auto-detect chamber type from roi_id
        structure_name, roi_polygon, marker_group_pixels = structure_library(roi_id)

        if verbose:
            print(f"    - Chamber type: {structure_name}")

        # 2. Create matching step for this chamber type
        matching_step = MarkerMatchingStep(marker_group_pixels, tolerance=60)

        # 3. Load and process image
        image = load_image(image_path)

        if collect_debug:
            debug_data.image = image
            debug_data.stage_position = stage_position
            debug_data.pixel_size = pixel_size
            debug_data.structure_name = structure_name
            debug_data.roi_polygon = roi_polygon
            debug_data.marker_group_pixels = marker_group_pixels

        # 4. Detect markers
        detection_result = detection_step(image)
        markers = detection_result["markers"]
        markers = [m for m in markers if m.get("conf", 0.0) >= conf_threshold]
        detection_result["markers"] = markers

        if verbose:
            print(f"    - Markers detected: {len(markers)} (conf >= {conf_threshold})")

        if collect_debug:
            debug_data.markers = markers

        if not markers:
            return ValidationResult(
                roi_id=roi_id,
                success=False,
                map_x=expected_position[0],
                map_y=expected_position[1],
                measured_x=None,
                measured_y=None,
                error=None,
                error_message="DETECTION: No markers found",
                debug_data=debug_data,
            )

        # 5. Match markers
        matching_result = matching_step(detection_result)
        matched_indices = matching_result["matched_marker_indices"]

        if verbose:
            print(f"    - Pairs matched: {len(matched_indices)}")

        if not matched_indices:
            return ValidationResult(
                roi_id=roi_id,
                success=False,
                map_x=expected_position[0],
                map_y=expected_position[1],
                measured_x=None,
                measured_y=None,
                error=None,
                error_message="MATCHING: No marker pairs matched",
                debug_data=debug_data,
            )

        # 5b. Compute rotation angle
        angles = compute_marker_group_angles(
            markers, matched_indices, marker_group_pixels, signed=True
        )

        if len(angles) >= 2:
            angle_range = max(angles) - min(angles)
            if angle_range > max_angle_deviation:
                return ValidationResult(
                    roi_id=roi_id,
                    success=False,
                    map_x=expected_position[0],
                    map_y=expected_position[1],
                    measured_x=None,
                    measured_y=None,
                    error=None,
                    error_message=(
                        f"ANGLES: Inconsistent rotation angles "
                        f"(range={angle_range:.2f}° > {max_angle_deviation:.1f}°)"
                    ),
                    debug_data=debug_data,
                )

        rotation_angle = np.mean(angles)

        if verbose:
            print(f"    - Rotation angle: {rotation_angle:.2f} deg")

        # 5c. Filter matched pairs by image bounds
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

        if collect_debug:
            debug_data.matched_indices = matched_indices
            debug_data.rotation_angle = rotation_angle

        if not matched_indices:
            return ValidationResult(
                roi_id=roi_id,
                success=False,
                map_x=expected_position[0],
                map_y=expected_position[1],
                measured_x=None,
                measured_y=None,
                error=None,
                error_message="BOUNDS: No marker pairs with RoI fully in image bounds",
                debug_data=debug_data,
            )

        # 6. Compute chamber center in pixels
        chamber_center_pixels = compute_chamber_center(
            markers, matched_indices, marker_group_pixels, roi_polygon, rotation_angle
        )

        if collect_debug:
            debug_data.chamber_center_pixels = chamber_center_pixels
            expected_offset_microns = expected_position - np.array(
                [stage_position["x"], stage_position["y"]]
            )
            debug_data.expected_center_pixels = expected_offset_microns / pixel_size

        # 7. Convert to microns
        chamber_center_microns = chamber_center_pixels * pixel_size

        # 8. Compute measured microscope position
        measured_x = stage_position["x"] + chamber_center_microns[0]
        measured_y = stage_position["y"] + chamber_center_microns[1]

        # 9. Compute L2 error
        error = float(
            np.sqrt(
                (measured_x - expected_position[0]) ** 2 + (measured_y - expected_position[1]) ** 2
            )
        )

        if collect_debug:
            debug_data.error_microns = error

        if verbose:
            print(f"    - Measured: ({measured_x:.2f}, {measured_y:.2f})")
            print(f"    - Expected: ({expected_position[0]:.2f}, {expected_position[1]:.2f})")
            print(f"    - L2 Error: {error:.3f} microns")
            print("    - Status: SUCCESS")

        return ValidationResult(
            roi_id=roi_id,
            success=True,
            map_x=expected_position[0],
            map_y=expected_position[1],
            measured_x=measured_x,
            measured_y=measured_y,
            error=error,
            error_message=None,
            debug_data=debug_data,
        )

    except Exception as e:
        return ValidationResult(
            roi_id=roi_id,
            success=False,
            map_x=expected_position[0] if expected_position is not None else None,
            map_y=expected_position[1] if expected_position is not None else None,
            measured_x=None,
            measured_y=None,
            error=None,
            error_message=f"ERROR: {e!s}",
            debug_data=debug_data,
        )


def run_validation(
    config: dict,
    detection_step: MarkerDetectionStep,
    structure_library,
    verbose: bool = False,
    max_images: int | None = None,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> ValidationSummary:
    """Run the full validation pipeline.

    Args:
        config: Configuration dictionary with calibrated_map_path, meta_csv_path,
            pixel_size, and optionally images_dir.
        detection_step: Initialized MarkerDetectionStep.
        structure_library: Initialized structure library.
        verbose: Print progress information.
        max_images: Maximum number of images to process.
        collect_debug: Collect debug data for visualization.
        conf_threshold: Minimum marker detection confidence.
        max_angle_deviation: Maximum angle range across pairs.

    Returns:
        ValidationSummary with all results and statistics.
    """
    pixel_size = config["pixel_size"]
    calibrated_map_path = Path(config["calibrated_map_path"])
    meta_csv_path = Path(config["meta_csv_path"])

    if verbose:
        print("=== Validation Pipeline ===")
        print()
        print("[Step 1/3] Loading data")
        print(f"  Calibrated map: {calibrated_map_path} ", end="")

    # Load calibrated map
    calibrated_map = Map.from_csv(calibrated_map_path)

    if verbose:
        print(f"({len(calibrated_map.roi_positions)} chambers)")

    # Load meta.csv
    meta_df = pd.read_csv(meta_csv_path)
    meta_df["roi_id"] = meta_df["roi_id"].apply(lambda rid: f"{int(rid):04d}")

    images_dir = config.get("images_dir")
    meta_dir = Path(images_dir) if images_dir is not None else meta_csv_path.parent

    if max_images is not None:
        meta_df = meta_df.head(max_images)

    if verbose:
        print(f"  Meta CSV: {meta_csv_path} ({len(meta_df)} images)")
        print(f"  Pixel size: {pixel_size} microns")
        print()

    # Process each validation image
    if verbose:
        print("[Step 2/3] Processing validation images")

    results: list[ValidationResult] = []

    pbar = tqdm(
        meta_df.iterrows(),
        total=len(meta_df),
        desc="Validating images",
        disable=verbose,
        unit="img",
    )

    for i, row in pbar:
        roi_id = row["roi_id"]
        pbar.set_postfix(roi=roi_id)

        stage_position = {
            "x": row["position_x"],
            "y": row["position_y"],
            "z": row.get("position_z", 0.0),
        }

        image_path = meta_dir / row["image_file"]

        if verbose:
            print(f"  Image {i + 1}/{len(meta_df)}: {image_path.name}")
            print(f"    - RoI ID: {roi_id}")

        # Check if roi_id exists in calibrated map
        if roi_id not in calibrated_map.roi_positions:
            if verbose:
                print("    - Status: SKIPPED - RoI not in calibrated map")
            results.append(
                ValidationResult(
                    roi_id=roi_id,
                    success=False,
                    map_x=None,
                    map_y=None,
                    measured_x=None,
                    measured_y=None,
                    error=None,
                    error_message="MISSING: RoI not found in calibrated map",
                )
            )
            continue

        expected_position = np.array(
            [
                calibrated_map.roi_positions[roi_id].x,
                calibrated_map.roi_positions[roi_id].y,
            ]
        )

        result = process_validation_image(
            image_path=image_path,
            roi_id=roi_id,
            stage_position=stage_position,
            expected_position=expected_position,
            detection_step=detection_step,
            structure_library=structure_library,
            pixel_size=pixel_size,
            verbose=verbose,
            collect_debug=collect_debug,
            conf_threshold=conf_threshold,
            max_angle_deviation=max_angle_deviation,
        )

        results.append(result)

    # Compute summary statistics
    errors = [r.error for r in results if r.success and r.error is not None]
    errors_arr = np.array(errors) if errors else np.array([0.0])

    n_success = sum(1 for r in results if r.success)
    n_failed = len(results) - n_success

    summary = ValidationSummary(
        results=results,
        mean_error=float(np.mean(errors_arr)),
        median_error=float(np.median(errors_arr)),
        std_error=float(np.std(errors_arr)),
        max_error=float(np.max(errors_arr)),
        min_error=float(np.min(errors_arr)),
        p90_error=float(np.percentile(errors_arr, 90)),
        n_success=n_success,
        n_failed=n_failed,
    )

    if verbose:
        print()
        print("[Step 3/3] Summary")
        print(f"  Successful: {n_success}/{len(results)}")
        if errors:
            print(f"  Mean error: {summary.mean_error:.3f} µm")
            print(f"  Median error: {summary.median_error:.3f} µm")
            print(f"  Std error: {summary.std_error:.3f} µm")
            print(f"  P90 error: {summary.p90_error:.3f} µm")

    return summary

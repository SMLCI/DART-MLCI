#!/usr/bin/env python
"""Validation script for comparing computed chamber positions to calibrated map.

Takes validation images with known stage positions, detects markers, computes chamber
centers, and compares against expected positions from a calibrated map.

Example usage:
    python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output
    python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output --verbose
    python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output --debug --max-images 5

JSON config format:
    {
        "calibrated_map_path": "calibrated_map.csv",
        "meta_csv_path": "/path/to/meta.csv",
        "pixel_size": 0.065789,
        "model_path": "artifacts/models/v8_detect_s_imgsz640.pt",
        "structure_library_path": "artifacts/chamber_structure.json"
    }
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calibrate_map import (
    compute_chamber_center,
    filter_matched_pairs_by_bounds,
)
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity
from tqdm import tqdm

import dmc_masking
from dmc_masking import MarkerDetectionStep, MarkerMatchingStep
from dmc_masking.io import load_image
from dmc_masking.map import Map
from dmc_masking.mask import RoIPolygon, SAKRoIStructureLibrary
from dmc_masking.rotation import compute_marker_group_angles


@dataclass
class ValidationDebugData:
    """Debug data for per-image validation visualization."""

    image: np.ndarray | None = None
    markers: list[dict] | None = None
    matched_indices: list[tuple[int, int]] | None = None
    chamber_center_pixels: np.ndarray | None = None  # Measured center in pixels
    expected_center_pixels: np.ndarray | None = None  # Expected center from map (in pixels)
    stage_position: dict[str, float] | None = None
    pixel_size: float | None = None
    structure_name: str | None = None
    roi_polygon: RoIPolygon | None = None
    marker_group_pixels: dict[str, np.ndarray] | None = None
    rotation_angle: float | None = None
    error_microns: float | None = None  # L2 error in microns


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


@dataclass
class ValidationSummary:
    """Summary of validation results."""

    results: list[ValidationResult]
    mean_error: float
    median_error: float
    std_error: float
    max_error: float
    min_error: float
    n_success: int
    n_failed: int


def load_config(path: Path) -> dict:
    """Load validation configuration from JSON file."""
    with open(path) as f:
        return json.load(f)


def validate_config(config: dict, config_path: Path | None = None) -> None:
    """Validate configuration and raise helpful errors."""
    source = f" in '{config_path}'" if config_path else ""

    required_fields = ["calibrated_map_path", "meta_csv_path", "pixel_size"]
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(
            f"Missing required field(s){source}: {', '.join(missing_fields)}\n"
            f"Required fields are:\n"
            f"  - calibrated_map_path: Path to calibrated map CSV\n"
            f"  - meta_csv_path: Path to meta.csv with validation images\n"
            f"  - pixel_size: Pixel size in microns (e.g., 0.065789)"
        )

    # Validate paths exist
    calibrated_map_path = Path(config["calibrated_map_path"])
    if not calibrated_map_path.exists():
        raise ValueError(f"'calibrated_map_path'{source}: File not found: {calibrated_map_path}")

    meta_csv_path = Path(config["meta_csv_path"])
    if not meta_csv_path.exists():
        raise ValueError(f"'meta_csv_path'{source}: File not found: {meta_csv_path}")

    # Validate pixel_size
    pixel_size = config["pixel_size"]
    if not isinstance(pixel_size, int | float) or pixel_size <= 0:
        raise ValueError(f"'pixel_size'{source} must be a positive number, got {pixel_size}")

    # Validate optional model_path if provided
    if "model_path" in config and config["model_path"] is not None:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise ValueError(f"'model_path'{source}: File not found: {model_path}")


def process_validation_image(
    image_path: Path,
    roi_id: str,
    stage_position: dict[str, float],
    expected_position: np.ndarray,
    detection_step: MarkerDetectionStep,
    structure_library: SAKRoIStructureLibrary,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
) -> ValidationResult:
    """Process a single validation image and compute error.

    Args:
        image_path: Path to the image file
        roi_id: RoI identifier (e.g., "0050")
        stage_position: Stage position dict with x, y, z
        expected_position: Expected position from calibrated map (x, y)
        detection_step: Marker detection step
        structure_library: SAK structure library for chamber type lookup
        pixel_size: Pixel size in microns
        verbose: Print progress information
        collect_debug: Collect debug data for visualization

    Returns:
        ValidationResult with error or failure reason
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

        # Store debug data
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

        if verbose:
            print(f"    - Markers detected: {len(markers)}")

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

        # 5b. Compute rotation angle from detected markers (needed for bounds check)
        angles = compute_marker_group_angles(
            markers, matched_indices, marker_group_pixels, signed=True
        )
        rotation_angle = np.mean(angles)

        if verbose:
            print(f"    - Rotation angle: {rotation_angle:.2f} deg")

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

        # 6. Compute chamber center in pixels (with rotation correction)
        chamber_center_pixels = compute_chamber_center(
            markers, matched_indices, marker_group_pixels, roi_polygon, rotation_angle
        )

        if collect_debug:
            debug_data.chamber_center_pixels = chamber_center_pixels
            # Compute expected center in pixels: expected_position is in microns,
            # convert to pixels relative to image origin (stage position)
            expected_offset_microns = expected_position - np.array(
                [stage_position["x"], stage_position["y"]]
            )
            debug_data.expected_center_pixels = expected_offset_microns / pixel_size

        # 8. Convert to microns
        chamber_center_microns = chamber_center_pixels * pixel_size

        # 9. Compute measured microscope position (stage is at top-left corner of image)
        measured_x = stage_position["x"] + chamber_center_microns[0]
        measured_y = stage_position["y"] + chamber_center_microns[1]

        # 10. Compute L2 error
        error = np.sqrt(
            (measured_x - expected_position[0]) ** 2 + (measured_y - expected_position[1]) ** 2
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
    verbose: bool = False,
    max_images: int | None = None,
    collect_debug: bool = False,
    debug_output_dir: Path | None = None,
) -> ValidationSummary:
    """Run the full validation pipeline.

    Args:
        config: Configuration dictionary
        verbose: Print progress information
        max_images: Maximum number of images to process (for testing)
        collect_debug: Collect debug data for visualization
        debug_output_dir: Directory to save debug images (saves incrementally)

    Returns:
        ValidationSummary with all results and statistics
    """
    pixel_size = config["pixel_size"]
    calibrated_map_path = Path(config["calibrated_map_path"])
    meta_csv_path = Path(config["meta_csv_path"])
    model_path = config.get("model_path")

    # Set default model path if not specified
    if model_path is None:
        model_path = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/v8_detect_s_imgsz640.pt"
        )
    else:
        model_path = Path(model_path)

    # Set default structure library path
    structure_library_path = config.get("structure_library_path")
    if structure_library_path is None:
        structure_library_path = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/chamber_structure.json"
        )
    else:
        structure_library_path = Path(structure_library_path)

    device = config.get("device")

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

    # Handle roi_id formatting (ensure 4-digit format)
    meta_df["roi_id"] = meta_df["roi_id"].apply(lambda rid: f"{int(rid):04d}")

    # Get the directory containing meta.csv for resolving relative image paths
    meta_dir = meta_csv_path.parent

    # Limit number of images if specified
    if max_images is not None:
        meta_df = meta_df.head(max_images)

    if verbose:
        print(f"  Meta CSV: {meta_csv_path} ({len(meta_df)} images)")
        print(f"  Model: {model_path}")
        print(f"  Pixel size: {pixel_size} microns")
        print()

    # Initialize structure library and detection step
    structure_library = SAKRoIStructureLibrary(
        lookup_path=structure_library_path,
        pixel_size=pixel_size,
    )

    detection_step = MarkerDetectionStep(str(model_path), device=device, verbose=False)

    # Process each validation image
    if verbose:
        print("[Step 2/3] Processing validation images")

    results: list[ValidationResult] = []

    # Use tqdm progress bar (disabled in verbose mode since it prints detailed info)
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

        # Get stage position
        stage_position = {
            "x": row["position_x"],
            "y": row["position_y"],
            "z": row.get("position_z", 0.0),
        }

        # Resolve image path (relative to meta.csv directory)
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

        # Get expected position from calibrated map
        expected_position = calibrated_map.roi_positions[roi_id].position

        # Check if image file exists
        if not image_path.exists():
            if verbose:
                print("    - Status: FAILED - Image file not found")
            results.append(
                ValidationResult(
                    roi_id=roi_id,
                    success=False,
                    map_x=expected_position[0],
                    map_y=expected_position[1],
                    measured_x=None,
                    measured_y=None,
                    error=None,
                    error_message=f"FILE: Image not found: {image_path}",
                )
            )
            continue

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
        )

        results.append(result)

        # Save debug image immediately after processing
        if debug_output_dir is not None and result.debug_data is not None:
            debug_path = debug_output_dir / f"{result.roi_id}_debug.png"
            plot_validation_debug(result.debug_data, result, debug_path)
            # Clear debug data to free memory
            result.debug_data = None

        if not result.success and verbose:
            print(f"    - Status: FAILED - {result.error_message}")

    if verbose:
        print()

    # Compute summary statistics
    successful_results = [r for r in results if r.success]
    errors = [r.error for r in successful_results]

    if errors:
        summary = ValidationSummary(
            results=results,
            mean_error=float(np.mean(errors)),
            median_error=float(np.median(errors)),
            std_error=float(np.std(errors)),
            max_error=float(np.max(errors)),
            min_error=float(np.min(errors)),
            n_success=len(successful_results),
            n_failed=len(results) - len(successful_results),
        )
    else:
        summary = ValidationSummary(
            results=results,
            mean_error=0.0,
            median_error=0.0,
            std_error=0.0,
            max_error=0.0,
            min_error=0.0,
            n_success=0,
            n_failed=len(results),
        )

    if verbose:
        print("[Step 3/3] Computing statistics")
        print(f"  Successful: {summary.n_success}/{len(results)}")
        print(f"  Failed: {summary.n_failed}/{len(results)}")
        if summary.n_success > 0:
            print(f"  Mean error: {summary.mean_error:.3f} microns")
            print(f"  Median error: {summary.median_error:.3f} microns")
            print(f"  Std error: {summary.std_error:.3f} microns")
            print(f"  Max error: {summary.max_error:.3f} microns")
            print(f"  Min error: {summary.min_error:.3f} microns")
        print()

    return summary


def plot_error_histogram(
    summary: ValidationSummary,
    output_path: Path,
) -> None:
    """Generate histogram of L2 errors.

    Args:
        summary: ValidationSummary with results
        output_path: Path to save the plot
    """
    errors = [r.error for r in summary.results if r.success and r.error is not None]

    if not errors:
        print("Warning: No successful results to plot histogram")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n_bins = min(30, len(errors) // 2 + 1)
    n_bins = max(10, n_bins)

    ax.hist(errors, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue")

    # Add vertical lines for mean and median
    ax.axvline(
        summary.mean_error,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {summary.mean_error:.3f} um",
    )
    ax.axvline(
        summary.median_error,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {summary.median_error:.3f} um",
    )

    ax.set_xlabel("L2 Error (microns)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Position Errors", fontsize=14)

    # Add stats annotation
    stats_text = (
        f"N = {summary.n_success}\n"
        f"Mean = {summary.mean_error:.3f} um\n"
        f"Median = {summary.median_error:.3f} um\n"
        f"Std = {summary.std_error:.3f} um\n"
        f"Max = {summary.max_error:.3f} um"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.75))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_map(
    summary: ValidationSummary,
    output_path: Path,
) -> None:
    """Generate map visualization with crosses colored by error.

    Args:
        summary: ValidationSummary with results
        output_path: Path to save the plot
    """
    successful_results = [r for r in summary.results if r.success]

    if not successful_results:
        print("Warning: No successful results to plot error map")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract positions and errors
    x_positions = [r.map_x for r in successful_results]
    y_positions = [r.map_y for r in successful_results]
    errors = [r.error for r in successful_results]

    # Create scatter plot with cross markers colored by error
    scatter = ax.scatter(
        x_positions,
        y_positions,
        c=errors,
        cmap="RdYlGn_r",
        marker="+",
        s=100,
        linewidths=2,
    )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="L2 Error (microns)")

    # Also plot failed results as gray X markers
    failed_results = [r for r in summary.results if not r.success and r.map_x is not None]
    if failed_results:
        failed_x = [r.map_x for r in failed_results]
        failed_y = [r.map_y for r in failed_results]
        ax.scatter(
            failed_x,
            failed_y,
            c="gray",
            marker="x",
            s=50,
            linewidths=1,
            alpha=0.5,
            label=f"Failed ({len(failed_results)})",
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("X (microns)", fontsize=12)
    ax.set_ylabel("Y (microns)", fontsize=12)
    ax.set_title(
        f"Validation Error Map (N={summary.n_success}, Mean Error={summary.mean_error:.3f} um)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_validation_debug(
    debug_data: ValidationDebugData,
    result: ValidationResult,
    output_path: Path,
) -> None:
    """Generate per-image debug visualization with markers, positions, and error.

    Args:
        debug_data: ValidationDebugData containing image, markers, and positions
        result: ValidationResult with measured/expected positions
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
        rotated_polygon = affinity.rotate(
            polygon, debug_data.rotation_angle, origin=(polygon_center[0], polygon_center[1])
        )

        # Step 2: Translate so the polygon centroid aligns with the computed chamber center
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

    # Draw expected center (from calibrated map) - magenta diamond
    if debug_data.expected_center_pixels is not None:
        ax.scatter(
            debug_data.expected_center_pixels[0],
            debug_data.expected_center_pixels[1],
            c="magenta",
            marker="D",
            s=300,
            edgecolors="black",
            linewidths=1,
            zorder=9,
            label="Expected (map)",
        )

    # Draw measured chamber center - gold star
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
            label="Measured center",
        )

    # Draw error arrow from expected to measured
    if (
        debug_data.expected_center_pixels is not None
        and debug_data.chamber_center_pixels is not None
    ):
        ax.annotate(
            "",
            xy=(debug_data.chamber_center_pixels[0], debug_data.chamber_center_pixels[1]),
            xytext=(debug_data.expected_center_pixels[0], debug_data.expected_center_pixels[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=3,
                mutation_scale=20,
            ),
            zorder=8,
        )

    # Add microscope position label near the measured center
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

    # Add text annotations in a single info box
    info_lines = []
    if debug_data.stage_position:
        info_lines.append(
            f"Stage: ({debug_data.stage_position['x']:.2f}, {debug_data.stage_position['y']:.2f}) µm"
        )
    if debug_data.chamber_center_pixels is not None:
        info_lines.append(
            f"Measured (px): ({debug_data.chamber_center_pixels[0]:.1f}, {debug_data.chamber_center_pixels[1]:.1f})"
        )
    if result.measured_x is not None and result.measured_y is not None:
        info_lines.append(f"Measured (µm): ({result.measured_x:.2f}, {result.measured_y:.2f})")
    if result.map_x is not None and result.map_y is not None:
        info_lines.append(f"Expected (µm): ({result.map_x:.2f}, {result.map_y:.2f})")
    if debug_data.error_microns is not None:
        info_lines.append(f"Error: {debug_data.error_microns:.3f} µm")
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
    title = f"RoI {result.roi_id}"
    if debug_data.structure_name:
        title += f" - {debug_data.structure_name}"
    if result.success:
        title += f" (Error: {debug_data.error_microns:.3f} µm)"
    else:
        title += f" - FAILED: {result.error_message}"
    ax.set_title(title, fontsize=14)

    # Add legend
    ax.legend(loc="upper right")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_validation_results(
    summary: ValidationSummary,
    output_path: Path,
) -> None:
    """Save validation results to CSV file.

    Args:
        summary: ValidationSummary with results
        output_path: Path to output CSV file
    """
    rows = []
    for r in summary.results:
        rows.append(
            {
                "roi_id": r.roi_id,
                "map_x": r.map_x,
                "map_y": r.map_y,
                "measured_x": r.measured_x,
                "measured_y": r.measured_y,
                "error": r.error,
                "success": r.success,
                "error_message": r.error_message,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def validate_map(
    config: dict | Path | str,
    output_dir: Path | str,
    device: str | None = None,
    verbose: bool = False,
    max_images: int | None = None,
    debug: bool = False,
) -> ValidationSummary:
    """Validate a calibrated map against validation images.

    This is the main programmatic entry point for the validation pipeline.

    Args:
        config: Either a configuration dictionary, or a path to a JSON config file.
            Required keys:
            - calibrated_map_path: Path to calibrated map CSV
            - meta_csv_path: Path to meta.csv with validation images
            - pixel_size: Pixel size in microns
            Optional keys:
            - model_path: Path to the detection model
            - structure_library_path: Path to the structure library JSON
            - device: Device to run on (can also be passed as argument)
        output_dir: Directory for outputs (histogram, map, CSV)
        device: Device to run on (e.g., 'cuda:0', 'cpu'). Overrides config if provided.
        verbose: Print detailed progress information.
        max_images: Maximum number of images to process (for testing).
        debug: Generate per-image debug visualizations.

    Returns:
        ValidationSummary with all results and statistics
    """
    # Load config if path is provided
    config_path = None
    if isinstance(config, str | Path):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_config(config_path)

    # Validate configuration
    validate_config(config, config_path)

    # Override device if specified
    if device:
        config["device"] = device

    # Convert output_dir to Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create debug images directory if debug mode is enabled
    debug_output_dir = None
    if debug:
        debug_output_dir = output_dir / "images"
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Debug images will be saved to: {debug_output_dir}")
            print()

    # Run validation
    summary = run_validation(
        config,
        verbose=verbose,
        max_images=max_images,
        collect_debug=debug,
        debug_output_dir=debug_output_dir,
    )

    # Generate outputs
    if verbose:
        print("=== Generating Outputs ===")
        print()

    # Save validation results CSV
    results_path = output_dir / "validation_results.csv"
    save_validation_results(summary, results_path)
    if verbose:
        print(f"  Results saved to: {results_path}")

    # Generate error histogram
    histogram_path = output_dir / "error_histogram.png"
    plot_error_histogram(summary, histogram_path)
    if verbose:
        print(f"  Histogram saved to: {histogram_path}")

    # Generate error map
    map_path = output_dir / "error_map.png"
    plot_error_map(summary, map_path)
    if verbose:
        print(f"  Error map saved to: {map_path}")

    if verbose:
        print()
        print("=== Validation Complete ===")
        print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate calibrated map against validation images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output
  python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output --verbose
  python scripts/validate_map.py --config validation_config.json --output-dir ./validation_output --debug --max-images 5

JSON config format:
  {
      "calibrated_map_path": "calibrated_map.csv",
      "meta_csv_path": "/path/to/meta.csv",
      "pixel_size": 0.065789,
      "model_path": "artifacts/models/v8_detect_s_imgsz640.pt",
      "structure_library_path": "artifacts/chamber_structure.json"
  }
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files (histogram, map, CSV)",
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
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate per-image debug visualizations in output_dir/images/",
    )

    args = parser.parse_args()

    try:
        # Run validation using the function API
        summary = validate_map(
            config=args.config,
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
            max_images=args.max_images,
            debug=args.debug,
        )

        # Print summary (always print, even if not verbose)
        print(
            f"\nValidation complete: {summary.n_success}/{len(summary.results)} images successful"
        )
        if summary.n_success > 0:
            print(f"Mean error: {summary.mean_error:.3f} microns")
            print(f"Median error: {summary.median_error:.3f} microns")
            print(f"Max error: {summary.max_error:.3f} microns")

        # Report failures
        failed = [r for r in summary.results if not r.success]
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

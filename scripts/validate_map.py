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
        "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt",
        "structure_library_path": "artifacts/chamber_structure.json"
    }
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity
from tqdm import tqdm

from dart_mlci import MarkerDetectionStep
from dart_mlci.calibration.validation import (
    ValidationDebugData,
    ValidationResult,
    ValidationSummary,
    process_validation_image,
)
from dart_mlci.constants import DEFAULT_MODEL_PATH, DEFAULT_STRUCTURE_LIBRARY_PATH
from dart_mlci.map import Map
from dart_mlci.mask import SAKRoIStructureLibrary
from dart_mlci.script_utils import load_json_config, validate_validation_config


def run_validation_cli(
    config: dict,
    verbose: bool = False,
    max_images: int | None = None,
    collect_debug: bool = False,
    debug_output_dir: Path | None = None,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
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
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)

    # Set default structure library path
    structure_library_path = config.get("structure_library_path")
    if structure_library_path is None:
        structure_library_path = DEFAULT_STRUCTURE_LIBRARY_PATH
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

    # Get the directory for resolving relative image paths
    # Use images_dir from config if provided, otherwise fall back to meta.csv parent
    images_dir = config.get("images_dir")
    if images_dir is not None:
        meta_dir = Path(images_dir)
    else:
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
            conf_threshold=conf_threshold,
            max_angle_deviation=max_angle_deviation,
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
            p90_error=float(np.percentile(errors, 90)),
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
            p90_error=0.0,
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
            print(f"  P90 error: {summary.p90_error:.3f} microns")
            print(f"  Max error: {summary.max_error:.3f} microns")
            print(f"  Min error: {summary.min_error:.3f} microns")
        print()

    return summary


def plot_error_histogram(
    summary: ValidationSummary,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
    label_fontsize: float = 12,
    title_fontsize: float = 14,
    tick_fontsize: float = 10,
    legend_fontsize: float = 10,
    stats_fontsize: float = 10,
    font_family: str | None = None,
    dpi: int = 150,
) -> None:
    """Generate histogram of L2 errors.

    Args:
        summary: ValidationSummary with results
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        label_fontsize: Font size for axis labels
        title_fontsize: Font size for the title
        tick_fontsize: Font size for axis tick labels
        legend_fontsize: Font size for legend text
        stats_fontsize: Font size for the stats annotation box
        font_family: Font family for the plot
        dpi: DPI for PNG output
    """
    errors = [r.error for r in summary.results if r.success and r.error is not None]

    if not errors:
        print("Warning: No successful results to plot histogram")
        return

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    _fig, ax = plt.subplots(figsize=figsize)

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
        label=rf"Mean: {summary.mean_error:.3f} $\mu$m",
    )
    ax.axvline(
        summary.median_error,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=rf"Median: {summary.median_error:.3f} $\mu$m",
    )
    ax.axvline(
        summary.p90_error,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=rf"P90: {summary.p90_error:.3f} $\mu$m",
    )

    ax.set_xlabel(r"L2 Error [$\mu$m]", fontsize=label_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("Distribution of Position Errors", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Add stats annotation
    stats_text = (
        f"N = {summary.n_success}\n"
        rf"Mean = {summary.mean_error:.3f} $\mu$m" + "\n"
        rf"Median = {summary.median_error:.3f} $\mu$m" + "\n"
        rf"Std = {summary.std_error:.3f} $\mu$m" + "\n"
        rf"P90 = {summary.p90_error:.3f} $\mu$m" + "\n"
        rf"Max = {summary.max_error:.3f} $\mu$m"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=stats_fontsize,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.70), fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches=None)
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches=None)
    plt.close()


def plot_error_histogram_pixels(
    summary: ValidationSummary,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
    label_fontsize: float = 12,
    title_fontsize: float = 14,
    tick_fontsize: float = 10,
    legend_fontsize: float = 10,
    stats_fontsize: float = 10,
    font_family: str | None = None,
    dpi: int = 150,
) -> None:
    """Generate histogram of L2 errors expressed in pixels.

    Reads ``error_px`` from the results (populated by ValidationSummary.to_csv
    or ValidationSummary.from_csv). Skips if no pixel errors are available.
    """
    errors_px = [r.error_px for r in summary.results if r.success and r.error_px is not None]

    if not errors_px:
        print("Warning: No successful results with error_px to plot pixel histogram")
        return

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    mean_px = float(np.mean(errors_px))
    median_px = float(np.median(errors_px))
    std_px = float(np.std(errors_px))
    p90_px = float(np.percentile(errors_px, 90))
    max_px = float(np.max(errors_px))

    _fig, ax = plt.subplots(figsize=figsize)
    n_bins = max(10, min(30, len(errors_px) // 2 + 1))
    ax.hist(errors_px, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue")

    ax.axvline(
        mean_px,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_px:.2f} px",
    )
    ax.axvline(
        median_px,
        color="purple",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_px:.2f} px",
    )
    ax.axvline(
        p90_px,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"P90: {p90_px:.2f} px",
    )

    ax.set_xlabel("L2 Error [pixels]", fontsize=label_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("Distribution of Position Errors (pixels)", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    stats_text = (
        f"N = {len(errors_px)}\n"
        f"Mean = {mean_px:.2f} px\n"
        f"Median = {median_px:.2f} px\n"
        f"Std = {std_px:.2f} px\n"
        f"P90 = {p90_px:.2f} px\n"
        f"Max = {max_px:.2f} px"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=stats_fontsize,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.70), fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches=None)
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches=None)
    plt.close()


def plot_error_map(
    summary: ValidationSummary,
    output_path: Path,
    figsize: tuple[float, float] | None = None,
    label_fontsize: float = 12,
    title_fontsize: float = 14,
    colorbar_fontsize: float = 12,
    marker_size: float = 100,
    marker_linewidth: float = 2,
    tick_fontsize: float = 10,
    colorbar_tick_fontsize: float = 10,
    font_family: str | None = None,
    dpi: int = 150,
    invert_xaxis: bool = True,
    invert_yaxis: bool = True,
) -> None:
    """Generate map visualization with crosses colored by error.

    Args:
        summary: ValidationSummary with results
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        label_fontsize: Font size for axis labels
        title_fontsize: Font size for the title
        colorbar_fontsize: Font size for the colorbar label
        marker_size: Size of the scatter markers
        marker_linewidth: Line width of the scatter markers
        dpi: DPI for PNG output
    """
    successful_results = [r for r in summary.results if r.success]

    if not successful_results:
        print("Warning: No successful results to plot error map")
        return

    # Extract positions and errors
    x_positions = [r.map_x for r in successful_results]
    y_positions = [r.map_y for r in successful_results]
    errors = [r.error for r in successful_results]

    # Auto-calculate figure size from data aspect ratio if not overridden
    if figsize is None:
        x_range = max(x_positions) - min(x_positions)
        y_range = max(y_positions) - min(y_positions)
        data_aspect = x_range / y_range if y_range > 0 else 1.0
        fig_height = 10
        # Extra width for colorbar
        fig_width = fig_height * data_aspect + 2
        figsize = (fig_width, fig_height)

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    _fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot with cross markers colored by error
    scatter = ax.scatter(
        x_positions,
        y_positions,
        c=errors,
        cmap="viridis",
        marker="+",
        s=marker_size,
        linewidths=marker_linewidth,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r"L2 Error [$\mu$m]", fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

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

    ax.set_xlabel(r"X [$\mu$m]", fontsize=label_fontsize)
    ax.set_ylabel(r"Y [$\mu$m]", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_title(
        rf"Validation Error Map (N={summary.n_success}, Mean Error={summary.mean_error:.3f} $\mu$m)",
        fontsize=title_fontsize,
    )
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches=None)
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches=None)
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

    _fig, ax = plt.subplots(figsize=(12, 10))
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
    plt.savefig(output_path, dpi=150, bbox_inches=None)
    plt.close()


def validate_map(
    config: dict | Path | str,
    output_dir: Path | str,
    device: str | None = None,
    verbose: bool = False,
    max_images: int | None = None,
    debug: bool = False,
    plot_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
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
        config = load_json_config(config_path)

    # Validate configuration
    validate_validation_config(config, config_path)

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

    # Get conf_threshold and max_angle_deviation from config
    conf_threshold = config.get("conf_threshold", 0.5)
    max_angle_deviation = config.get("max_angle_deviation", 5.0)

    # Run validation
    summary = run_validation_cli(
        config,
        verbose=verbose,
        max_images=max_images,
        collect_debug=debug,
        debug_output_dir=debug_output_dir,
        conf_threshold=conf_threshold,
        max_angle_deviation=max_angle_deviation,
    )

    # Generate outputs
    if verbose:
        print("=== Generating Outputs ===")
        print()

    # Save validation results CSV (populates error_px on each result)
    pixel_size = config["pixel_size"]
    results_path = output_dir / "validation_results.csv"
    summary.to_csv(results_path, pixel_size)
    if verbose:
        print(f"  Results saved to: {results_path}")

    # Generate error histograms (microns + pixels)
    histogram_path = output_dir / "error_histogram.png"
    plot_error_histogram(summary, histogram_path, **(hist_kwargs or {}))
    if verbose:
        print(f"  Histogram saved to: {histogram_path}")

    histogram_px_path = output_dir / "error_histogram_pixels.png"
    plot_error_histogram_pixels(summary, histogram_px_path, **(hist_kwargs or {}))
    if verbose:
        print(f"  Pixel histogram saved to: {histogram_px_path}")

    # Generate error map
    map_path = output_dir / "error_map.png"
    plot_error_map(summary, map_path, **(plot_kwargs or {}))
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
      "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt",
      "structure_library_path": "artifacts/chamber_structure.json"
  }
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON configuration file (not required with --replot)",
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
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate figures from existing validation_results.csv without re-running validation",
    )
    parser.add_argument(
        "--map-figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Error map figure size in inches (default: auto from data)",
    )
    parser.add_argument(
        "--hist-figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Histogram figure size in inches (default: 10 6)",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=12,
        help="Font size for axis labels (default: 12)",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=14,
        help="Font size for the title (default: 14)",
    )
    parser.add_argument(
        "--colorbar-fontsize",
        type=float,
        default=12,
        help="Font size for colorbar label (default: 12)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=100,
        help="Scatter marker size (default: 100)",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=float,
        default=10,
        help="Font size for axis tick labels (default: 10)",
    )
    parser.add_argument(
        "--colorbar-tick-fontsize",
        type=float,
        default=10,
        help="Font size for colorbar tick labels (default: 10)",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        default=None,
        help="Font family for the plot (e.g., 'serif', 'sans-serif', 'Times New Roman', 'Arial')",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG output (default: 150)",
    )
    parser.add_argument(
        "--no-invert-xaxis",
        action="store_true",
        help="Do not invert the error-map x-axis (default: inverted)",
    )
    parser.add_argument(
        "--no-invert-yaxis",
        action="store_true",
        help="Do not invert the error-map y-axis (default: inverted)",
    )

    args = parser.parse_args()

    if not args.replot and args.config is None:
        parser.error("--config is required unless --replot is used")

    shared_kwargs = dict(
        label_fontsize=args.label_fontsize,
        title_fontsize=args.title_fontsize,
        tick_fontsize=args.tick_fontsize,
        font_family=args.font_family,
        dpi=args.dpi,
    )
    map_kwargs = dict(
        **shared_kwargs,
        figsize=tuple(args.map_figsize) if args.map_figsize else None,
        colorbar_fontsize=args.colorbar_fontsize,
        colorbar_tick_fontsize=args.colorbar_tick_fontsize,
        marker_size=args.marker_size,
        invert_xaxis=not args.no_invert_xaxis,
        invert_yaxis=not args.no_invert_yaxis,
    )
    hist_kwargs = dict(
        **shared_kwargs,
        figsize=tuple(args.hist_figsize) if args.hist_figsize else (10, 6),
    )

    try:
        output_dir = Path(args.output_dir)

        if args.replot:
            # Regenerate figures from existing CSV
            csv_path = output_dir / "validation_results.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"No validation results found at: {csv_path}")

            summary = ValidationSummary.from_csv(csv_path)
            print(f"Loaded {len(summary.results)} results from {csv_path}")

            histogram_path = output_dir / "error_histogram.png"
            plot_error_histogram(summary, histogram_path, **hist_kwargs)
            print(f"  Histogram saved to: {histogram_path}")

            histogram_px_path = output_dir / "error_histogram_pixels.png"
            plot_error_histogram_pixels(summary, histogram_px_path, **hist_kwargs)
            print(f"  Pixel histogram saved to: {histogram_px_path}")

            map_path = output_dir / "error_map.png"
            plot_error_map(summary, map_path, **map_kwargs)
            print(f"  Error map saved to: {map_path}")
            print(f"  Error map saved to: {map_path.with_suffix('.svg')}")
        else:
            # Override config from CLI if provided
            config_input = args.config
            if args.conf_threshold is not None or args.max_angle_deviation is not None:
                config_path = Path(args.config)
                if not config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                config_input = load_json_config(config_path)
                if args.conf_threshold is not None:
                    config_input["conf_threshold"] = args.conf_threshold
                if args.max_angle_deviation is not None:
                    config_input["max_angle_deviation"] = args.max_angle_deviation

            # Run validation using the function API
            summary = validate_map(
                config=config_input,
                output_dir=args.output_dir,
                device=args.device,
                verbose=args.verbose,
                max_images=args.max_images,
                debug=args.debug,
                plot_kwargs=map_kwargs,
                hist_kwargs=hist_kwargs,
            )

        # Print summary (always print, even if not verbose)
        print(
            f"\nValidation complete: {summary.n_success}/{len(summary.results)} images successful"
        )
        if summary.n_success > 0:
            print(f"Mean error: {summary.mean_error:.3f} microns")
            print(f"Median error: {summary.median_error:.3f} microns")
            print(f"P90 error: {summary.p90_error:.3f} microns")
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

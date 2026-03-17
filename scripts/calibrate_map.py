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
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity

import dart_mlci
from dart_mlci import ChipStructureLibrary, MarkerDetectionStep
from dart_mlci.calibration import (
    CalibrationResult,
    ImageCalibrationResult,
    ImageDebugData,
)
from dart_mlci.calibration import (
    process_calibration_image as _process_calibration_image,
)
from dart_mlci.calibration import (
    run_calibration as _run_calibration,
)
from dart_mlci.io import load_image
from dart_mlci.map import AffineTransformResult, Map
from dart_mlci.mask import RoIPolygon, SAKRoIStructureLibrary, apply_mask_rotation_free


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
    structure_library,
    pixel_size: float,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> ImageCalibrationResult:
    """Process a single calibration image from a file path.

    Thin wrapper that loads the image and delegates to the core function.
    """
    image = load_image(image_path)
    return _process_calibration_image(
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


def run_calibration(
    config: dict,
    verbose: bool = False,
    collect_debug: bool = False,
    conf_threshold: float = 0.5,
    max_angle_deviation: float = 5.0,
) -> tuple[CalibrationResult, Map]:
    """Run the full calibration pipeline from a config dict with file paths.

    This is a CLI-oriented wrapper that resolves file paths, loads the model,
    loads images from disk, then delegates to the core run_calibration.

    Args:
        config: Configuration dictionary with calibration_images, pixel_size, etc.
        verbose: Print progress information
        collect_debug: Collect debug data for visualizations
        conf_threshold: Minimum confidence for detected markers (default: 0.5)
        max_angle_deviation: Maximum allowed angle range in degrees (default: 5.0)

    Returns:
        Tuple of (CalibrationResult with calibrated map and statistics, blueprint_map)
    """
    pixel_size = config["pixel_size"]
    blueprint_map_path = config.get("blueprint_map_path")
    model_path = config.get("model_path")

    # Set default model path if not specified
    if model_path is None:
        model_path = (
            Path(dart_mlci.__file__).parent.parent / "artifacts/models/v26_detect_s_imgsz1280.pt"
        )
    else:
        model_path = Path(model_path)

    # Set default structure library or chip config path
    chip_config_path = config.get("chip_config_path")
    structure_library_path = config.get("structure_library_path")
    if chip_config_path is None and structure_library_path is None:
        structure_library_path = (
            Path(dart_mlci.__file__).parent.parent / "artifacts/chamber_structure.json"
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

    # Load all images from disk
    images = []
    roi_ids = []
    stage_positions = []
    for img_config in calibration_images:
        image_path = Path(img_config["image_path"])
        images.append(load_image(image_path))
        roi_ids.append(str(img_config["roi_id"]))
        stage_positions.append(img_config["stage_position"])

    if verbose:
        print(f"  Loaded {len(images)} images from disk")
        print()

    # Delegate to core run_calibration
    result = _run_calibration(
        images=images,
        roi_ids=roi_ids,
        stage_positions=stage_positions,
        detection_step=detection_step,
        structure_library=structure_library,
        blueprint_map=blueprint_map,
        pixel_size=pixel_size,
        verbose=verbose,
        collect_debug=collect_debug,
        conf_threshold=conf_threshold,
        max_angle_deviation=max_angle_deviation,
    )

    return result, blueprint_map


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

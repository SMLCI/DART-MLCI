#!/usr/bin/env python
"""
Generate golden files for regression testing.
Run this ONCE before refactoring to capture expected behavior.

Usage:
    conda activate dmc-masking-claude
    python scripts/generate_golden_files.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_calibration_golden():
    """Generate golden files from current calibration output."""
    from scripts.calibrate_map import calibrate_map

    output_dir = Path("tests/golden")
    output_dir.mkdir(exist_ok=True)

    config_path = Path("scripts/calibration_test.json")
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Cannot generate golden files without test config.")
        sys.exit(1)

    print("Running calibration to capture golden values...")
    print(f"Config: {config_path}")
    print()

    # Run calibration
    result, blueprint_map = calibrate_map(
        config=config_path,
        verbose=True,
    )

    # Save calibrated positions
    positions = {}
    for roi_id, roi_pos in result.calibrated_map.roi_positions.items():
        positions[roi_id] = {"position": roi_pos.position.tolist()}

    with open(output_dir / "calibration_positions.json", "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Saved calibrated positions ({len(positions)} RoIs)")

    # Get the affine matrix by transforming basis vectors
    transform_fn = result.transform_result.transform
    origin = np.array([0.0, 0.0])
    x_unit = np.array([1.0, 0.0])
    y_unit = np.array([0.0, 1.0])

    t_origin = transform_fn(origin)
    t_x = transform_fn(x_unit)
    t_y = transform_fn(y_unit)

    # Reconstruct affine matrix: [[a, b, tx], [c, d, ty]]
    a, c = t_x - t_origin
    b, d = t_y - t_origin
    tx, ty = t_origin

    affine_matrix = [[float(a), float(b), float(tx)], [float(c), float(d), float(ty)]]

    # Save transform parameters
    transform = {
        "matrix": affine_matrix,
        "rmse": float(result.transform_result.rmse),
        "max_error": float(result.transform_result.max_error),
        "residuals": result.transform_result.residuals.tolist(),
    }

    with open(output_dir / "transform_params.json", "w") as f:
        json.dump(transform, f, indent=2)
    print(f"Saved transform parameters (RMSE: {transform['rmse']:.3f} microns)")

    # Save successful image results for regression testing
    image_results = {}
    for img_result in result.image_results:
        if img_result.success:
            image_results[img_result.roi_id] = {
                "microscope_position": img_result.microscope_position.tolist(),
                "z_position": img_result.z_position,
            }
        else:
            image_results[img_result.roi_id] = {
                "success": False,
                "error": img_result.error_message,
            }

    with open(output_dir / "image_results.json", "w") as f:
        json.dump(image_results, f, indent=2)
    print(f"Saved image results ({len(image_results)} images)")

    # Save blueprint map positions for reference
    blueprint_positions = {}
    for roi_id, roi_pos in blueprint_map.roi_positions.items():
        blueprint_positions[roi_id] = {"position": roi_pos.position.tolist()}

    with open(output_dir / "blueprint_positions.json", "w") as f:
        json.dump(blueprint_positions, f, indent=2)
    print(f"Saved blueprint positions ({len(blueprint_positions)} RoIs)")

    print()
    print(f"Golden files saved to {output_dir.absolute()}")
    print()
    print("Summary:")
    print(f"  - RMSE: {result.transform_result.rmse:.3f} microns")
    print(f"  - Max error: {result.transform_result.max_error:.3f} microns")
    print(f"  - Calibration points: {sum(1 for r in result.image_results if r.success)}")
    print(f"  - Total RoIs: {len(result.calibrated_map.roi_positions)}")


if __name__ == "__main__":
    generate_calibration_golden()

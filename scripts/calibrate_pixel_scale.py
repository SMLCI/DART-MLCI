#!/usr/bin/env python
"""Verify pixel calibration by comparing detected marker distances to physical distances.

Loads a folder config JSON (same format as process_folder.py), detects markers in
the first frame of the first TIFF in each subfolder, and compares the detected
marker distance to the expected physical distance to compute the actual pixel size.

Example usage:
    python scripts/calibrate_pixel_scale.py --config dart_experiment/folder_config_cellpose_sam.json
    python scripts/calibrate_pixel_scale.py --config dart_experiment/folder_config.json --confidence 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

import dart_mlci
from dart_mlci.chip import ChipStructureLibrary
from dart_mlci.detection import MarkerDetectionModel
from dart_mlci.utils import normalize_image


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    required = ["input_dir", "folders"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Verify pixel calibration from detected marker distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to folder config JSON")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum marker detection confidence (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(str(args.config))

    base_dir = Path(dart_mlci.__file__).parent.parent
    input_dir = base_dir / config["input_dir"]
    chip_config_path = base_dir / config.get("chip_config", "artifacts/chips/sak.json")
    model_path = base_dir / config.get("model_path", "artifacts/models/v26_detect_s_imgsz1280.pt")
    config_pixel_size = config.get("pixel_size", 0.065789)
    flip = config.get("flip", False)
    folders = config["folders"]

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    # Load chip config at pixel_size=1.0 to get physical distances in µm
    chip_lib = ChipStructureLibrary.from_file(chip_config_path, pixel_size=1.0)
    mdm = MarkerDetectionModel(str(model_path), verbose=False)

    header = (
        f"{'Folder':<30s} {'ChamberType':<25s} {'PhysDist(um)':>13s} "
        f"{'DetDist(px)':>13s} {'ExpDist(px)':>13s} {'NeededPxSz':>13s} "
        f"{'CfgPxSz':>13s} {'Error(%)':>10s}"
    )
    print(header)
    print("-" * len(header))

    all_needed = []

    for folder_name, ctype in folders.items():
        folder_path = input_dir / folder_name
        if not folder_path.exists():
            print(f"{folder_name:<30s} FOLDER NOT FOUND")
            continue

        tifs = sorted(folder_path.glob("*.tif")) + sorted(folder_path.glob("*.tiff"))
        if not tifs:
            print(f"{folder_name:<30s} NO TIF FOUND")
            continue
        tif_path = tifs[0]

        # Load first frame
        stack = tifffile.imread(str(tif_path))
        frame = stack[0] if stack.ndim == 3 else stack
        if flip:
            frame = frame[::-1]
        frame = normalize_image(frame)

        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Detect markers
        markers = mdm.predict_markers(frame)
        markers = [m for m in markers if m.get("conf", 0) >= args.confidence]

        crosses = [m for m in markers if m["label"] == "cross"]
        circles = [m for m in markers if m["label"] == "circle"]

        if not crosses or not circles:
            print(
                f"{folder_name:<30s} {ctype:<25s} "
                f"No marker pairs (crosses={len(crosses)}, circles={len(circles)})"
            )
            continue

        # Physical distance from chip config
        mg_phys = chip_lib.marker_group_configs[ctype]
        phys_dist = np.linalg.norm(np.array(mg_phys["cross"]) - np.array(mg_phys["circle"]))

        # Expected distance at configured pixel size
        expected_dist = phys_dist / config_pixel_size

        # Find best matching cross-circle pair
        best_det_dist = None
        best_err = float("inf")
        for c in crosses:
            for ci in circles:
                d = np.linalg.norm(np.array(c["bbox_center"]) - np.array(ci["bbox_center"]))
                err = abs(d - expected_dist)
                if err < best_err:
                    best_err = err
                    best_det_dist = d

        needed_px = phys_dist / best_det_dist
        error_pct = (needed_px - config_pixel_size) / config_pixel_size * 100
        all_needed.append(needed_px)

        print(
            f"{folder_name:<30s} {ctype:<25s} {phys_dist:>13.2f} "
            f"{best_det_dist:>13.1f} {expected_dist:>13.1f} {needed_px:>13.6f} "
            f"{config_pixel_size:>13.6f} {error_pct:>+10.2f}"
        )

    if all_needed:
        print()
        mean_needed = np.mean(all_needed)
        std_needed = np.std(all_needed)
        print(f"Mean needed pixel size: {mean_needed:.6f} +/- {std_needed:.6f}")
        print(f"Current config pixel size: {config_pixel_size:.6f}")
        print(f"Difference: {(mean_needed - config_pixel_size) / config_pixel_size * 100:+.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare calibration data zip files for reproducibility.

Creates two zip files from the full mapping dataset:
  - Subset zip (~250 MB): 3 calibration images + 20 spatially distributed
    validation images + configs
  - Full zip (~9 GB): all 1164 images + configs

Each zip contains:
    calibration_data/
    ├── calibration_config.json    # For calibrate_map.py
    ├── validation_config.json     # For validate_map.py
    ├── meta.csv                   # Image metadata (full or subset)
    └── images/                    # TIF files

Usage:
    python scripts/prepare_calibration_data.py --output-dir ./calibration_zips
    python scripts/prepare_calibration_data.py --output-dir ./calibration_zips --subset-only
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Source data
DATA_DIR = Path("/mnt/ibg-1_omerostorage/2025_02_21/mapping/data/output-2025-02-25_16:10:00")
META_CSV = DATA_DIR / "meta.csv"

# Calibration roi_ids — image_file and stage_position are resolved from
# meta.csv at runtime to guarantee the association stays correct.
CALIBRATION_ROI_IDS = ["0000", "7000", "7129"]


def resolve_calibration_images(meta_df: pd.DataFrame) -> list[dict]:
    """Look up image_file and stage_position for each calibration roi_id.

    Raises ValueError if a roi_id is missing or matches multiple rows.
    """
    resolved = []
    for roi_id in CALIBRATION_ROI_IDS:
        rows = meta_df[meta_df["roi_id"] == roi_id]
        if len(rows) == 0:
            raise ValueError(f"Calibration roi_id {roi_id!r} not found in meta.csv")
        if len(rows) > 1:
            raise ValueError(
                f"Calibration roi_id {roi_id!r} matches {len(rows)} rows in meta.csv; "
                f"expected exactly one"
            )
        row = rows.iloc[0]
        resolved.append(
            {
                "image_file": row["image_file"],
                "roi_id": roi_id,
                "stage_position": {
                    "x": float(row["position_x"]),
                    "y": float(row["position_y"]),
                    "z": float(row["position_z"]),
                },
            }
        )
    return resolved


PIXEL_SIZE = 0.065789
N_VALIDATION = 20


def select_validation_images(
    meta_df: pd.DataFrame,
    calibration_files: set[str],
    n: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Select spatially distributed validation images.

    Uses a greedy farthest-point sampling strategy to pick images that are
    well spread across the chip.

    Args:
        meta_df: Full meta.csv DataFrame
        calibration_files: Set of calibration image filenames to exclude
        n: Number of validation images to select
        seed: Random seed for the initial point

    Returns:
        DataFrame with the selected validation rows
    """
    # Exclude calibration images
    candidates = meta_df[~meta_df["image_file"].isin(calibration_files)].copy()
    candidates = candidates.reset_index(drop=True)

    if len(candidates) <= n:
        return candidates

    # Build position array
    positions = candidates[["position_x", "position_y"]].values

    # Greedy farthest-point sampling
    rng = np.random.RandomState(seed)
    selected_indices = [rng.randint(len(candidates))]

    for _ in range(n - 1):
        selected_positions = positions[selected_indices]
        # Compute min distance from each candidate to any selected point
        dists = np.min(
            np.linalg.norm(positions[:, None, :] - selected_positions[None, :, :], axis=2),
            axis=1,
        )
        # Set already-selected to 0 so they aren't re-picked
        dists[selected_indices] = 0
        selected_indices.append(int(np.argmax(dists)))

    return candidates.iloc[selected_indices]


def build_calibration_config(calibration_images: list[dict], image_dir: str = "images") -> dict:
    """Build calibration_config.json content."""
    cal_images = []
    for img in calibration_images:
        cal_images.append(
            {
                "image_path": f"{image_dir}/{img['image_file']}",
                "roi_id": img["roi_id"],
                "stage_position": img["stage_position"],
            }
        )

    return {
        "chip_name": "SAK",
        "calibration_images": cal_images,
        "pixel_size": PIXEL_SIZE,
        "chip_config_path": "artifacts/chips/sak.json",
        "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt",
    }


def build_validation_config() -> dict:
    """Build validation_config.json content."""
    return {
        "calibrated_map_path": "calibrated_map.csv",
        "meta_csv_path": "meta.csv",
        "pixel_size": PIXEL_SIZE,
        "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt",
        "chip_config_path": "artifacts/chips/sak.json",
    }


def create_zip(
    output_path: Path,
    meta_df: pd.DataFrame,
    data_dir: Path,
    calibration_images: list[dict],
    verbose: bool = False,
) -> None:
    """Create a zip file with calibration data.

    Args:
        output_path: Path for the output zip file
        meta_df: DataFrame with rows to include (must have 'image_file' column)
        data_dir: Source directory containing the TIF files
        verbose: Print progress
    """
    prefix = "calibration_data"

    # Map each source filename -> roi-id-prefixed filename (e.g. "7030.tif").
    # meta.csv is rewritten so image_file matches the new names in images/.
    meta_df = meta_df.copy()
    rename_map: dict[str, str] = {}
    for _, row in meta_df.iterrows():
        src_name = row["image_file"]
        ext = Path(src_name).suffix or ".tif"
        new_name = f"{row['roi_id']}{ext}"
        if src_name in rename_map and rename_map[src_name] != new_name:
            raise ValueError(
                f"Image {src_name} is referenced by multiple roi_ids; cannot rename uniquely"
            )
        rename_map[src_name] = new_name
    meta_df["image_file"] = meta_df["image_file"].map(rename_map)

    # Mirror the rename into the calibration_images list used by the config
    renamed_calibration_images = [
        {**img, "image_file": rename_map[img["image_file"]]} for img in calibration_images
    ]

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
        cal_config = build_calibration_config(renamed_calibration_images, image_dir="images")
        zf.writestr(
            f"{prefix}/calibration_config.json",
            json.dumps(cal_config, indent=2) + "\n",
        )

        val_config = build_validation_config()
        zf.writestr(
            f"{prefix}/validation_config.json",
            json.dumps(val_config, indent=2) + "\n",
        )

        zf.writestr(f"{prefix}/meta.csv", meta_df.to_csv(index=False))

        items = list(rename_map.items())
        for i, (src_name, new_name) in enumerate(items):
            src = data_dir / src_name
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                continue
            if verbose and (i % 50 == 0 or i == len(items) - 1):
                print(f"  Adding image {i + 1}/{len(items)}: {new_name}")
            zf.write(src, f"{prefix}/images/{new_name}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Created {output_path} ({size_mb:.1f} MB, {len(rename_map)} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare calibration data zip files for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output zip files",
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only create the subset zip (skip full zip)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load meta.csv
    print("Loading meta.csv...")
    meta_df = pd.read_csv(META_CSV)
    meta_df["roi_id"] = meta_df["roi_id"].apply(lambda rid: f"{int(rid):04d}")
    print(f"  {len(meta_df)} images total")

    # Resolve calibration image_file + stage_position from meta.csv
    calibration_images = resolve_calibration_images(meta_df)
    cal_files = {img["image_file"] for img in calibration_images}
    for img in calibration_images:
        print(f"  roi {img['roi_id']} -> {img['image_file']}")

    # Select validation images for subset
    print("Selecting spatially distributed validation images...")
    val_df = select_validation_images(meta_df, cal_files, n=N_VALIDATION)
    print(f"  Selected {len(val_df)} validation images")

    # Build subset meta.csv (calibration + validation images)
    cal_df = meta_df[meta_df["image_file"].isin(cal_files)]
    subset_df = pd.concat([cal_df, val_df], ignore_index=True)
    subset_df = subset_df.drop_duplicates(subset=["image_file"])
    print(f"  Subset total: {len(subset_df)} images")

    # Create subset zip
    print("\nCreating subset zip...")
    subset_path = args.output_dir / "calibration_data_subset.zip"
    create_zip(subset_path, subset_df, DATA_DIR, calibration_images, verbose=args.verbose)

    if not args.subset_only:
        # Create full zip
        print("\nCreating full zip...")
        full_path = args.output_dir / "calibration_data_full.zip"
        create_zip(full_path, meta_df, DATA_DIR, calibration_images, verbose=args.verbose)

    print("\nDone! Upload the zip files to sciebo and update the URLs in reproduce.sh")


if __name__ == "__main__":
    main()

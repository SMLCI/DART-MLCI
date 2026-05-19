"""Shared utilities for CLI scripts.

Provides common functions used across multiple scripts to reduce duplication.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


def load_json_config(path: Path | str, required_keys: list[str] | None = None) -> dict:
    """Load and validate a JSON configuration file.

    Args:
        path: Path to the JSON file.
        required_keys: Optional list of keys that must be present.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = json.load(f)

    if required_keys:
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Config missing required keys: {missing}")

    return config


def load_image_list(csv_path: Path | str) -> list[tuple[str, str]]:
    """Load image list from CSV file with image_path and chamber_type columns.

    Args:
        csv_path: Path to CSV file with columns: image_path, chamber_type

    Returns:
        List of (image_path, chamber_type) tuples.
    """
    df = pd.read_csv(csv_path, dtype=str).dropna()
    return list(zip(df["image_path"].str.strip(), df["chamber_type"].str.strip(), strict=False))


def validate_calibration_config(config: dict, config_path: Path | None = None) -> None:
    """Validate a map-calibration configuration and raise helpful errors.

    Args:
        config: Configuration dictionary to validate.
        config_path: Optional path to config file (used in error messages).

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    source = f" in '{config_path}'" if config_path else ""

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

    if "blueprint_map_path" not in config and "chip_config_path" not in config:
        raise ValueError(
            f"Must provide either 'blueprint_map_path' (CSV) or 'chip_config_path' "
            f"(chip JSON with blueprint_map){source}"
        )

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

    for i, img_config in enumerate(cal_images):
        prefix = f"calibration_images[{i}]{source}"

        if not isinstance(img_config, dict):
            raise ValueError(f"{prefix} must be a dictionary, got {type(img_config).__name__}")

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
                f"{prefix}.stage_position is missing required field(s): "
                f"{', '.join(stage_missing)}\n"
                f"stage_position must have 'x' and 'y' keys (and optionally 'z')"
            )

        image_path = Path(img_config["image_path"])
        if not image_path.exists():
            raise ValueError(f"{prefix}.image_path: File not found: {image_path}")

    pixel_size = config["pixel_size"]
    if not isinstance(pixel_size, int | float) or pixel_size <= 0:
        raise ValueError(f"'pixel_size'{source} must be a positive number, got {pixel_size}")

    if "blueprint_map_path" in config:
        blueprint_path = Path(config["blueprint_map_path"])
        if not blueprint_path.exists():
            raise ValueError(f"'blueprint_map_path'{source}: File not found: {blueprint_path}")

    if "chip_config_path" in config and config["chip_config_path"] is not None:
        chip_config_path = Path(config["chip_config_path"])
        if not chip_config_path.exists():
            raise ValueError(f"'chip_config_path'{source}: File not found: {chip_config_path}")

    if "model_path" in config and config["model_path"] is not None:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise ValueError(f"'model_path'{source}: File not found: {model_path}")


def validate_validation_config(config: dict, config_path: Path | None = None) -> None:
    """Validate a map-validation configuration and raise helpful errors."""
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

    calibrated_map_path = Path(config["calibrated_map_path"])
    if not calibrated_map_path.exists():
        raise ValueError(f"'calibrated_map_path'{source}: File not found: {calibrated_map_path}")

    meta_csv_path = Path(config["meta_csv_path"])
    if not meta_csv_path.exists():
        raise ValueError(f"'meta_csv_path'{source}: File not found: {meta_csv_path}")

    pixel_size = config["pixel_size"]
    if not isinstance(pixel_size, int | float) or pixel_size <= 0:
        raise ValueError(f"'pixel_size'{source} must be a positive number, got {pixel_size}")

    if "model_path" in config and config["model_path"] is not None:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise ValueError(f"'model_path'{source}: File not found: {model_path}")


def get_peak_gpu_memory_mb() -> float:
    """Return peak CUDA allocation in MB since the last reset, or 0.0 if CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory_stats() -> None:
    """Reset CUDA peak-memory tracking; no-op if torch/CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class Timer:
    """Context manager for precise timing measurements.

    Usage:
        >>> with Timer() as t:
        ...     do_work()
        >>> print(t.elapsed)
    """

    def __init__(self):
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start

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

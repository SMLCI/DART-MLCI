"""Centralized constants for the dart-mlci package."""

from pathlib import Path

DEFAULT_PIXEL_SIZE_UM: float = 0.065789
"""Default pixel size in microns per pixel."""

DEFAULT_MARKER_TOLERANCE_PX: int = 60
"""Default tolerance in pixels for marker matching."""

ARTIFACTS_DIR: Path = Path(__file__).parent.parent / "artifacts"
"""Root directory for bundled artifact files (models, configs, etc.)."""

DEFAULT_MODEL_PATH: Path = ARTIFACTS_DIR / "models/v26_detect_s_imgsz1280.pt"
"""Default path to the YOLO marker detection model weights."""

DEFAULT_CHIP_CONFIG_PATH: Path = ARTIFACTS_DIR / "chips/sak.json"
"""Default path to the unified SAK chip configuration JSON file."""

DEFAULT_STRUCTURE_LIBRARY_PATH: Path = ARTIFACTS_DIR / "chamber_structure.json"
"""Default path to the legacy chamber structure JSON file (deprecated)."""

"""Centralized constants for the dmc-masking package."""

from pathlib import Path

DEFAULT_PIXEL_SIZE_UM: float = 0.065789
"""Default pixel size in microns per pixel."""

DEFAULT_MARKER_TOLERANCE_PX: int = 60
"""Default tolerance in pixels for marker matching."""

DEFAULT_MODEL_PATH: Path = Path(__file__).parent.parent / "artifacts/models/v8_detect_s_imgsz640.pt"
"""Default path to the YOLO marker detection model weights."""

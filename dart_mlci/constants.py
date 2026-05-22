"""Centralized constants for the dart-mlci package."""

from pathlib import Path

from dart_mlci.artifacts import get_artifacts_dir

DEFAULT_PIXEL_SIZE_UM: float = 0.065789
"""Default pixel size in microns per pixel."""

DEFAULT_MARKER_TOLERANCE_PX: int = 60
"""Default tolerance in pixels for marker matching."""

ARTIFACTS_DIR: Path = get_artifacts_dir()
"""Root directory for bundled artifact files (models, configs, etc.).

Resolves to the repo's `artifacts/` directory in a source checkout, or to
the per-user cache (e.g. `~/.cache/dart-mlci/`) for pip-installed users.
See `dart_mlci.artifacts.get_artifacts_dir` for details.
"""

DEFAULT_MODEL_PATH: Path = ARTIFACTS_DIR / "models/v26_detect_s_imgsz1280.pt"
"""Default path to the YOLO marker detection model weights."""

DEFAULT_CHIP_CONFIG_PATH: Path = ARTIFACTS_DIR / "chips/sak.json"
"""Default path to the unified SAK chip configuration JSON file."""

DEFAULT_STRUCTURE_LIBRARY_PATH: Path = ARTIFACTS_DIR / "chamber_structure.json"
"""Default path to the legacy chamber structure JSON file (deprecated)."""

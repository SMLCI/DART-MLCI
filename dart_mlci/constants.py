"""Centralized constants for the dart-mlci package."""

from pathlib import Path

from dart_mlci.artifacts import ensure_artifact, get_artifacts_dir

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

# Relative paths under ARTIFACTS_DIR. Single source of truth for the default
# artifact names — change here and every consumer (scripts, library code,
# tests) picks it up.
DEFAULT_MODEL_RELPATH: str = "models/v26_detect_s_imgsz1280.pt"
DEFAULT_CHIP_CONFIG_RELPATH: str = "chips/sak.json"
DEFAULT_STRUCTURE_LIBRARY_RELPATH: str = "chamber_structure.json"

DEFAULT_MODEL_PATH: Path = ARTIFACTS_DIR / DEFAULT_MODEL_RELPATH
"""Default path to the YOLO marker detection model weights.

This is a static path — it does NOT trigger a download. Use
`ensure_default_model()` when you actually need the file on disk.
"""

DEFAULT_CHIP_CONFIG_PATH: Path = ARTIFACTS_DIR / DEFAULT_CHIP_CONFIG_RELPATH
"""Default path to the unified SAK chip configuration JSON file."""

DEFAULT_STRUCTURE_LIBRARY_PATH: Path = ARTIFACTS_DIR / DEFAULT_STRUCTURE_LIBRARY_RELPATH
"""Default path to the legacy chamber structure JSON file (deprecated)."""


def ensure_default_model() -> Path:
    """Return the default model path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_MODEL_RELPATH)


def ensure_default_chip_config() -> Path:
    """Return the default chip-config path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_CHIP_CONFIG_RELPATH)


def ensure_default_structure_library() -> Path:
    """Return the legacy structure-library path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_STRUCTURE_LIBRARY_RELPATH)

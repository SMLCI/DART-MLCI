"""Centralized constants for the dart-mlci package."""

from pathlib import Path

from dart_mlci.artifacts import ensure_artifact, get_artifacts_dir

DEFAULT_PIXEL_SIZE_UM: float = 0.065789
"""Default pixel size in microns per pixel."""

DEFAULT_MARKER_TOLERANCE_UM: float = 4.0
"""Default tolerance in microns for marker matching.

Expressed in physical units (rather than pixels) so it stays valid across
cameras with different pixel sizes. Convert to pixels via
``dart_mlci.calibration.coordinates.PixelToMicronTransform(pixel_size).inverse(...)``
at the point of use.
"""

DEFAULT_MAX_ANGLE_DEVIATION_DEG: float = 5.0
"""Default maximum allowed rotation-angle range across matched marker pairs, in degrees."""


def marker_tolerance_px(
    pixel_size: float, tolerance_um: float = DEFAULT_MARKER_TOLERANCE_UM
) -> float:
    """Convert a marker-matching tolerance from microns to pixels.

    Args:
        pixel_size: Size of one pixel in microns (camera-dependent).
        tolerance_um: Tolerance in microns. Defaults to DEFAULT_MARKER_TOLERANCE_UM.

    Returns:
        Tolerance in pixels.
    """
    return tolerance_um / pixel_size


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


# Canonical display order/numbering (1-8) for SAK chip chamber types, shared
# across scripts (timing tables, growth-rate summaries). Matches the "SAK RoI
# dimensions" reference table: (1) NormaleBox-inner 60x60, (2) BigBox-inner
# 60x100, (3) OpenBox-inner 60x80, (4) Mothermachine-inner 15x1x80,
# (5) NormaleBox-pillar-inner 60x60, (6) BigBox-pillar-inner 60x100,
# (7) OpenBox-collector-inner 60x80, (8) Mothermachine-2x-inner 7x2x80 [um^2].
CHAMBER_TYPE_ORDER: list[str] = [
    "NormaleBox-inner",
    "BigBox-inner",
    "OpenBox-inner",
    "Mothermachine-inner",
    "NormaleBox-pillar-inner",
    "BigBox-pillar-inner",
    "OpenBox-collector-inner",
    "Mothermachine-2x-inner",
]
CHAMBER_TYPE_NUMBERS: dict[str, int] = dict(
    zip(CHAMBER_TYPE_ORDER, range(1, len(CHAMBER_TYPE_ORDER) + 1), strict=False)
)


def ensure_default_model() -> Path:
    """Return the default model path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_MODEL_RELPATH)


def ensure_default_chip_config() -> Path:
    """Return the default chip-config path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_CHIP_CONFIG_RELPATH)


def ensure_default_structure_library() -> Path:
    """Return the legacy structure-library path, auto-downloading on first call."""
    return ensure_artifact(DEFAULT_STRUCTURE_LIBRARY_RELPATH)

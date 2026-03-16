"""Environment configuration for the DART API."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    """API settings loaded from environment variables."""

    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "DART_MODEL_PATH", "/app/artifacts/models/v26_detect_s_imgsz1280.pt"
        )
    )
    structure_library_path: str = field(
        default_factory=lambda: os.environ.get(
            "DART_STRUCTURE_LIBRARY_PATH", "/app/artifacts/chamber_structure.json"
        )
    )
    blueprint_map_path: str = field(
        default_factory=lambda: os.environ.get(
            "DART_BLUEPRINT_MAP_PATH", "/app/artifacts/sak_blueprint_map.csv"
        )
    )
    default_pixel_size: float = field(
        default_factory=lambda: float(os.environ.get("DART_PIXEL_SIZE", "0.065789"))
    )
    chip_config_path: str | None = field(
        default_factory=lambda: os.environ.get("DART_CHIP_CONFIG_PATH", None)
    )
    chip_configs_dir: str | None = field(
        default_factory=lambda: os.environ.get("DART_CHIP_CONFIGS_DIR", None)
    )
    device: str | None = field(default_factory=lambda: os.environ.get("DART_DEVICE", None))

    def __post_init__(self):
        """Validate paths exist where required."""
        # In Docker, paths may not exist until artifacts are mounted
        # So we only warn, not fail
        pass


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def resolve_path(path: str | Path, allow_relative: bool = True) -> Path:
    """Resolve a path, allowing both absolute and relative paths.

    Args:
        path: Path string or Path object
        allow_relative: If True, relative paths are resolved from CWD

    Returns:
        Resolved Path object
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if allow_relative:
        return Path.cwd() / p
    raise ValueError(f"Path must be absolute: {path}")

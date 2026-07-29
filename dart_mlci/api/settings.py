"""Environment configuration for the DART API."""

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Written by the Dockerfile's git-info build stage (see Dockerfile).
_GIT_COMMIT_SHA_FILE = Path("/app/git-commit-sha")
_GIT_COMMIT_MESSAGE_FILE = Path("/app/git-commit-message")

# Repo root when running outside Docker (dart_mlci/api/settings.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _detect_git_commit_sha() -> str | None:
    """Commit SHA of the running code: env override, baked-in file, or live git."""
    if sha := os.environ.get("GIT_COMMIT_SHA"):
        return sha
    if _GIT_COMMIT_SHA_FILE.exists():
        return _GIT_COMMIT_SHA_FILE.read_text().strip() or None
    return _run_git("rev-parse", "HEAD")


def _detect_git_commit_message() -> str | None:
    """Commit subject of the running code: env override, baked-in file, or live git."""
    if message := os.environ.get("GIT_COMMIT_MESSAGE"):
        return message
    if _GIT_COMMIT_MESSAGE_FILE.exists():
        return _GIT_COMMIT_MESSAGE_FILE.read_text().strip() or None
    return _run_git("log", "-1", "--format=%s")


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


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
    segmenter: str | None = field(default_factory=lambda: os.environ.get("DART_SEGMENTER", None))
    segmentation_filter_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DART_SEGMENTATION_FILTER_THRESHOLD", "0.5"))
    )
    git_commit_sha: str | None = field(default_factory=_detect_git_commit_sha)
    git_commit_message: str | None = field(default_factory=_detect_git_commit_message)

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

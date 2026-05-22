"""On-demand artifact resolution and download.

The DART artifact bundle (~40 MB of YOLO weights, sample images, default
chip configs) is hosted on Sciebo rather than shipped in the PyPI wheel.
This module makes that transparent: the first call to `ensure_artifact()`
or `sample_path()` triggers a one-time download to a user-cache directory;
all subsequent calls are silent cache hits.

There is no explicit pre-warm CLI by design — downloads happen lazily on
first use so the user never has to remember a setup step.

Public API:
    get_artifacts_dir()   → resolve cache root
    ensure_artifact(rel)  → resolve `<cache>/rel`, downloading if missing
    sample_path(rel)      → alias of ensure_artifact for README-friendly use
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_URL = "https://fz-juelich.sciebo.de/s/S4bYt6C9rtR3sF2/download"
"""Sciebo URL of the artifact zip. Override with $DART_ARTIFACTS_URL."""

_BUNDLE_BYTES_APPROX = 40 * 1024 * 1024  # for the log line; not enforced


def _repo_artifacts_dir() -> Path | None:
    """Return `<repo>/artifacts` if running from a source checkout, else None.

    Walks up from this file looking for a `pyproject.toml` sibling that also
    has an `artifacts/` directory. Preserves the dev workflow where everyone
    expects `<repo>/artifacts/` to be the canonical location.
    """
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "artifacts").is_dir():
            return parent / "artifacts"
    return None


def get_artifacts_dir() -> Path:
    """Resolve the artifact cache directory.

    Resolution order:
        1. `$DART_ARTIFACTS_DIR` if set.
        2. `<repo>/artifacts/` if running from a source checkout.
        3. `platformdirs.user_cache_dir("dart-mlci")` — typically
           `~/.cache/dart-mlci/` on Linux/macOS,
           `%LOCALAPPDATA%\\dart-mlci\\` on Windows.
    """
    env = os.environ.get("DART_ARTIFACTS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    repo = _repo_artifacts_dir()
    if repo is not None:
        return repo

    return Path(platformdirs.user_cache_dir("dart-mlci", appauthor=False))


def ensure_artifact(relpath: str | os.PathLike[str]) -> Path:
    """Resolve `<artifacts_dir>/relpath`, downloading the bundle if missing.

    Args:
        relpath: Path relative to the artifacts root, e.g.
            `"models/v26_detect_s_imgsz1280.pt"` or `"chips/sak.json"`.

    Returns:
        Absolute path to the resolved file.

    Raises:
        FileNotFoundError: If the file is missing from the downloaded bundle.
        urllib.error.URLError: If the download fails (network, server, ...).
    """
    root = get_artifacts_dir()
    target = root / Path(relpath)
    if target.exists():
        return target

    _download_bundle(root)

    if not target.exists():
        raise FileNotFoundError(
            f"Artifact {relpath!r} not found in downloaded bundle at {root}. "
            "The bundle may be out of date; check $DART_ARTIFACTS_URL."
        )
    return target


def sample_path(relpath: str | os.PathLike[str]) -> Path:
    """README-friendly alias of `ensure_artifact`.

    Use this in example snippets so readers can paste them verbatim after
    `pip install dart-mlci` without needing a repo clone.
    """
    return ensure_artifact(relpath)


def _download_bundle(target_dir: Path) -> None:
    """Download the artifact zip and extract its contents into `target_dir`.

    Atomic: download to a tmp file, extract to a staging dir, then move
    each top-level entry into place. Concurrent first-use is safe — the
    last writer wins on identical files via `os.replace`.

    Emits one INFO log on entry, optional stderr progress for interactive
    sessions, and one INFO log on completion. Subsequent calls (cache hit)
    do not enter this function and stay silent.
    """
    url = os.environ.get("DART_ARTIFACTS_URL", DEFAULT_ARTIFACTS_URL)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading DART artifacts (~%d MB) from %s to %s — this happens once, then is cached.",
        _BUNDLE_BYTES_APPROX // (1024 * 1024),
        url,
        target_dir,
    )

    with tempfile.TemporaryDirectory(prefix="dart-artifacts-", dir=target_dir) as staging:
        staging_dir = Path(staging)
        zip_path = staging_dir / "bundle.zip"

        req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            _stream_to_file(resp, zip_path, total)

        extract_dir = staging_dir / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        _merge_into(_unwrap_artifacts(extract_dir), target_dir)

    logger.info("DART artifacts ready at %s.", target_dir)


def _user_agent() -> str:
    from dart_mlci import __version__

    return f"dart-mlci/{__version__} (+https://github.com/SMLCI/DART-MLCI)"


def _stream_to_file(resp, dest: Path, total: int) -> None:
    """Copy `resp` to `dest`, drawing a progress line on stderr if interactive."""
    show_progress = total > 0 and sys.stderr.isatty()
    chunk = 1 << 16
    done = 0
    with dest.open("wb") as f:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if show_progress:
                pct = 100 * done / total
                sys.stderr.write(
                    f"\rdart-mlci artifacts: {pct:5.1f}% "
                    f"({done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB)"
                )
                sys.stderr.flush()
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _unwrap_artifacts(extract_dir: Path) -> Path:
    """Return the directory whose children should land in the cache root.

    The Sciebo zip ships an `artifacts/` top-level. If we see that, use its
    contents; otherwise treat the extract dir itself as the contents.
    """
    inner = extract_dir / "artifacts"
    if inner.is_dir():
        return inner
    return extract_dir


def _merge_into(src: Path, dst: Path) -> None:
    """Move every child of `src` into `dst`, overwriting existing files."""
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                # merge recursively
                _merge_into(child, target)
            else:
                shutil.move(str(child), str(target))
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))

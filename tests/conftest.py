"""Pytest fixtures for DMC masking tests."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_artifacts() -> None:
    """Pre-warm the artifact cache once per session.

    Many test modules read files via raw `Path(...).exists()` checks that
    bypass `dart_mlci.artifacts.ensure_artifact`. Calling `ensure_artifact`
    once at session start materializes the bundle (model weights + sample
    images + chip configs) in the same location those tests look at
    (`<repo>/artifacts/` for a source checkout). After this fires, all
    subsequent path checks just work.

    If the network is unavailable, skip the whole session with a clear
    reason rather than letting each test fail with FileNotFoundError.
    """
    from dart_mlci.artifacts import ensure_artifact

    try:
        ensure_artifact("models/v26_detect_s_imgsz1280.pt")
        ensure_artifact("images/sak/0000.png")
    except Exception as exc:  # network/zip/URL issues
        pytest.skip(f"DART artifact bundle unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def device_fixture():
    """Return available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

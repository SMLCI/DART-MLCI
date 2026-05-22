"""Unit tests for dart_mlci.artifacts (offline; no real network)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from dart_mlci import artifacts


def _build_bundle_zip() -> bytes:
    """Build a minimal in-memory zip mirroring the Sciebo bundle layout."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("artifacts/models/v26_detect_s_imgsz1280.pt", b"fake-weights")
        zf.writestr("artifacts/chips/sak.json", '{"name": "SAK"}')
        zf.writestr("artifacts/images/sak/0007.png", b"\x89PNG\r\n\x1a\n")
    return buf.getvalue()


def _serve_bundle(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Replace urllib.request.urlopen with a fake that yields `body`."""

    class _FakeResp:
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)
            self.headers = {"Content-Length": str(len(data))}

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self._buf.close()

    def _fake_urlopen(_req, timeout: int = 0):
        return _FakeResp(body)

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", _fake_urlopen)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DART_ARTIFACTS_DIR at a fresh temp dir for the duration of the test."""
    monkeypatch.setenv("DART_ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


def test_get_artifacts_dir_honors_env(isolated_cache: Path) -> None:
    assert artifacts.get_artifacts_dir() == isolated_cache


def test_ensure_artifact_downloads_on_miss(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_bundle(monkeypatch, _build_bundle_zip())

    chip = artifacts.ensure_artifact("chips/sak.json")
    assert chip.exists()
    assert chip.read_text() == '{"name": "SAK"}'

    # Sibling files from the same bundle land in the cache too.
    model = isolated_cache / "models" / "v26_detect_s_imgsz1280.pt"
    assert model.exists()
    assert model.read_bytes() == b"fake-weights"


def test_ensure_artifact_is_silent_on_cache_hit(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_bundle(monkeypatch, _build_bundle_zip())
    artifacts.ensure_artifact("chips/sak.json")

    # Second call must NOT call urlopen — disarm the fake to detect re-entry.
    def _boom(*_a, **_kw):
        raise AssertionError("urlopen called on cache hit")

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", _boom)
    result = artifacts.ensure_artifact("chips/sak.json")
    assert result.exists()


def test_ensure_artifact_raises_when_member_missing(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_bundle(monkeypatch, _build_bundle_zip())
    with pytest.raises(FileNotFoundError, match="not found in downloaded bundle"):
        artifacts.ensure_artifact("nonexistent/file.bin")


def test_sample_path_aliases_ensure_artifact(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_bundle(monkeypatch, _build_bundle_zip())
    assert artifacts.sample_path("chips/sak.json") == artifacts.ensure_artifact("chips/sak.json")

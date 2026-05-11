"""Integration tests for the artifact download script.

These tests download ~1.2 GB from Sciebo and verify extraction.
Run explicitly with: pytest -m integration tests/test_download_artifacts.py
"""

import os
import shutil
import stat
import subprocess

import pytest

ARTIFACTS_URL = "https://fz-juelich.sciebo.de/s/S4bYt6C9rtR3sF2/download"

EXPECTED_FILES = [
    "artifacts/models/v26_detect_s_imgsz1280.pt",
    "artifacts/models/v8_detect_s_imgsz640.pt",
    "artifacts/images/sak/0000.png",
    "artifacts/images/calibration_sample/0000.tif",
    "artifacts/images/image_stack.tif",
]


@pytest.fixture()
def fake_repo(tmp_path):
    """Create a minimal repo structure with the download script."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    src = os.path.join(os.path.dirname(__file__), "..", "scripts", "download_artifacts.sh")
    dest = scripts_dir / "download_artifacts.sh"
    shutil.copy2(src, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
    return tmp_path


@pytest.mark.integration
def test_download_and_extract(fake_repo):
    """Download the zip and verify expected files are extracted correctly."""
    result = subprocess.run(
        [str(fake_repo / "scripts" / "download_artifacts.sh")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    for rel_path in EXPECTED_FILES:
        full_path = fake_repo / rel_path
        assert full_path.exists(), f"Missing expected file: {rel_path}"
        assert full_path.stat().st_size > 0, f"File is empty: {rel_path}"

    # Verify no nested artifacts/artifacts/ directory
    assert not (fake_repo / "artifacts" / "artifacts").exists(), (
        "Nested artifacts/artifacts/ directory found — extraction path bug"
    )


@pytest.mark.integration
def test_skip_when_artifacts_exist(fake_repo):
    """Script should skip download when artifacts dirs already exist."""
    # Create the directories that trigger the skip check
    (fake_repo / "artifacts" / "models").mkdir(parents=True)
    (fake_repo / "artifacts" / "images").mkdir(parents=True)

    result = subprocess.run(
        [str(fake_repo / "scripts" / "download_artifacts.sh")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "skipping download" in result.stdout.lower()

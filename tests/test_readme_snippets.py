"""Verify that the Python snippets in README.md actually run.

If these tests fail after editing README.md, the README is out of sync with the
public API. Update the snippets there and here together.

Tests are skipped (not failed) when the YOLO model weights or the bundled
sample image are missing, since those are downloaded post-install via
`scripts/download_artifacts.sh` and may not exist in CI without the artifact
download step.
"""

from pathlib import Path

import cv2
import pytest

import dart_mlci

REPO_ROOT = Path(dart_mlci.__file__).parent.parent
SAK_CHIP_CONFIG = REPO_ROOT / "artifacts" / "chips" / "sak.json"
SAMPLE_IMAGE = REPO_ROOT / "artifacts" / "images" / "sak" / "0007.png"
# These constants must mirror the README. If you change either side, change both.
SAK_PIXEL_SIZE_UM = 0.065789
SAMPLE_ROI_ID = "0000"  # any NormaleBox-inner ROI matches the sample image


pytestmark = pytest.mark.skipif(
    not SAMPLE_IMAGE.exists() or not SAK_CHIP_CONFIG.exists(),
    reason="Sample image or chip config not present (run scripts/download_artifacts.sh)",
)


def _load_image():
    img = cv2.imread(str(SAMPLE_IMAGE))
    assert img is not None, f"failed to read {SAMPLE_IMAGE}"
    return img


def test_readme_snippet_marker_detection():
    """Snippet 1: load chip config + run MarkerDetectionModel on a sample image."""
    from dart_mlci import ChipStructureLibrary, MarkerDetectionModel

    lib = ChipStructureLibrary.from_file(str(SAK_CHIP_CONFIG), pixel_size=SAK_PIXEL_SIZE_UM)
    assert "NormaleBox-inner" in lib.chip_config.chamber_types

    model = MarkerDetectionModel()
    image = _load_image()
    markers = model.predict_markers(image)

    # The bundled 0007.png contains a NormaleBox-inner chamber whose two
    # cross+circle pairs are clearly detectable. If this fails, either the
    # weights regressed or the sample image was replaced.
    assert len(markers) >= 2, f"expected ≥2 markers in {SAMPLE_IMAGE.name}, got {len(markers)}"
    labels = {m["label"] for m in markers}
    assert labels <= {"cross", "circle"}, f"unexpected labels: {labels}"


def test_readme_snippet_full_pipeline():
    """Snippet 2: detect → match → rotate → mask returns a cropped image + mask."""
    from dart_mlci import (
        ChipStructureLibrary,
        ImageRotationStep,
        MarkerDetectionStep,
        MarkerMatchingStep,
        RoIMaskingStep,
    )

    lib = ChipStructureLibrary.from_file(str(SAK_CHIP_CONFIG), pixel_size=SAK_PIXEL_SIZE_UM)
    _, polygon, mgp = lib(SAMPLE_ROI_ID)

    image = _load_image()
    detect = MarkerDetectionStep()
    match = MarkerMatchingStep(marker_group_pixel=mgp)
    rotate = ImageRotationStep()
    mask = RoIMaskingStep(marker_group_pixels=mgp, roi_polygon=polygon)

    data = mask(rotate(match(detect(image))))
    cropped, chamber_mask = data["image"], data["mask"]

    assert cropped.ndim == 3 and cropped.shape[-1] == 3, (
        f"unexpected cropped shape: {cropped.shape}"
    )
    assert chamber_mask.shape == cropped.shape[:2], (
        f"mask shape {chamber_mask.shape} does not match cropped {cropped.shape[:2]}"
    )
    assert cropped.size > 0 and chamber_mask.size > 0

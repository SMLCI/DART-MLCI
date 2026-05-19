"""Tests for helpers that were promoted from scripts/ into the core dart_mlci
package during the pre-release cleanup pass.

Covers:
- ``dart_mlci.script_utils.validate_calibration_config`` /
  ``validate_validation_config``
- ``dart_mlci.script_utils.get_peak_gpu_memory_mb`` /
  ``reset_gpu_memory_stats``
- ``dart_mlci.io.save_image``
- ``dart_mlci.map.Map.to_csv``
- ``dart_mlci.calibration.core.CalibrationResult.save_stats``
- ``dart_mlci.calibration.validation.ValidationSummary.to_csv`` /
  ``from_csv``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dart_mlci.calibration.core import (
    CalibrationResult,
    ImageCalibrationResult,
)
from dart_mlci.calibration.validation import ValidationResult, ValidationSummary
from dart_mlci.io import load_image, save_image
from dart_mlci.map import AffineTransformResult, Map, RoIPosition
from dart_mlci.script_utils import (
    get_peak_gpu_memory_mb,
    reset_gpu_memory_stats,
    validate_calibration_config,
    validate_validation_config,
)


def _make_map(entries):
    return Map([RoIPosition(rid, np.array(pos, dtype=float)) for rid, pos in entries])


# ---------------------------------------------------------------------------
# validate_calibration_config
# ---------------------------------------------------------------------------


class TestValidateCalibrationConfig:
    def _good(self, image_paths):
        return {
            "calibration_images": [
                {
                    "image_path": str(p),
                    "roi_id": f"{i:04d}",
                    "stage_position": {"x": float(i), "y": float(i)},
                }
                for i, p in enumerate(image_paths)
            ],
            "pixel_size": 0.065789,
            "chip_config_path": None,
        }

    def test_happy_path(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.tif"
            p.write_bytes(b"x")
            paths.append(p)
        chip = tmp_path / "chip.json"
        chip.write_text("{}")
        cfg = self._good(paths)
        cfg["chip_config_path"] = str(chip)
        validate_calibration_config(cfg)

    def test_missing_required_field(self, tmp_path):
        with pytest.raises(ValueError, match="Missing required field"):
            validate_calibration_config({})

    def test_missing_blueprint_source(self, tmp_path):
        paths = [tmp_path / f"img_{i}.tif" for i in range(3)]
        for p in paths:
            p.write_bytes(b"x")
        cfg = self._good(paths)
        # neither blueprint_map_path nor chip_config_path
        del cfg["chip_config_path"]
        with pytest.raises(ValueError, match="blueprint_map_path"):
            validate_calibration_config(cfg)

    def test_too_few_images(self, tmp_path):
        p = tmp_path / "only.tif"
        p.write_bytes(b"x")
        cfg = self._good([p])
        cfg["blueprint_map_path"] = str(p)
        with pytest.raises(ValueError, match="at least 3"):
            validate_calibration_config(cfg)

    def test_missing_image_file(self, tmp_path):
        cfg = self._good([tmp_path / f"missing_{i}.tif" for i in range(3)])
        cfg["blueprint_map_path"] = str(tmp_path / "missing_blueprint.csv")
        with pytest.raises(ValueError, match="File not found"):
            validate_calibration_config(cfg)

    def test_bad_pixel_size(self, tmp_path):
        paths = [tmp_path / f"img_{i}.tif" for i in range(3)]
        for p in paths:
            p.write_bytes(b"x")
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = self._good(paths)
        cfg["blueprint_map_path"] = str(bp)
        cfg["pixel_size"] = -1.0
        with pytest.raises(ValueError, match="positive number"):
            validate_calibration_config(cfg)


# ---------------------------------------------------------------------------
# validate_validation_config
# ---------------------------------------------------------------------------


class TestValidateValidationConfig:
    def test_happy_path(self, tmp_path):
        cal = tmp_path / "cal.csv"
        cal.write_text("roi_id,x,y,z\n")
        meta = tmp_path / "meta.csv"
        meta.write_text("roi_id,position_x,position_y,image_file\n")
        validate_validation_config(
            {
                "calibrated_map_path": str(cal),
                "meta_csv_path": str(meta),
                "pixel_size": 0.065789,
            }
        )

    def test_missing_keys(self):
        with pytest.raises(ValueError, match="Missing required field"):
            validate_validation_config({})

    def test_missing_files(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            validate_validation_config(
                {
                    "calibrated_map_path": str(tmp_path / "nope.csv"),
                    "meta_csv_path": str(tmp_path / "nope2.csv"),
                    "pixel_size": 0.1,
                }
            )

    def test_bad_pixel_size(self, tmp_path):
        cal = tmp_path / "cal.csv"
        cal.write_text("x\n")
        meta = tmp_path / "meta.csv"
        meta.write_text("x\n")
        with pytest.raises(ValueError, match="positive number"):
            validate_validation_config(
                {
                    "calibrated_map_path": str(cal),
                    "meta_csv_path": str(meta),
                    "pixel_size": 0,
                }
            )


# ---------------------------------------------------------------------------
# GPU helpers — should no-op gracefully when CUDA absent
# ---------------------------------------------------------------------------


class TestGpuHelpers:
    def test_get_peak_returns_float(self):
        val = get_peak_gpu_memory_mb()
        assert isinstance(val, float)
        assert val >= 0.0

    def test_reset_does_not_raise(self):
        # Should be a no-op without CUDA; must not raise.
        reset_gpu_memory_stats()


# ---------------------------------------------------------------------------
# Map.to_csv
# ---------------------------------------------------------------------------


class TestMapToCsv:
    def test_round_trip_without_z(self, tmp_path):
        m = _make_map([("0001", [1.5, 2.5]), ("0002", [3.0, 4.0])])
        out = tmp_path / "map.csv"
        m.to_csv(out)
        df = pd.read_csv(out)
        assert set(df.columns) == {"roi_id", "x", "y", "z"}
        assert (df["z"] == 0.0).all()
        m2 = Map.from_csv(out)
        assert set(m2.roi_positions.keys()) == {"0001", "0002"}
        np.testing.assert_allclose(m2.roi_positions["0001"].position, [1.5, 2.5])

    def test_with_z_positions_uses_mean_for_missing(self, tmp_path):
        m = _make_map([("0001", [0.0, 0.0]), ("0002", [1.0, 1.0]), ("0003", [2.0, 2.0])])
        out = tmp_path / "map.csv"
        m.to_csv(out, z_positions={"0001": 10.0, "0002": 20.0})
        df = pd.read_csv(out).set_index("roi_id")
        df.index = [f"{i:04d}" for i in df.index]
        assert df.loc["0001", "z"] == 10.0
        assert df.loc["0002", "z"] == 20.0
        # 0003 was missing → mean of provided z's
        assert df.loc["0003", "z"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


class TestSaveImage:
    def _img_hwc(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)

    def test_png_round_trip(self, tmp_path):
        img = self._img_hwc()
        out = tmp_path / "out.png"
        mask_path = save_image(img, out)
        assert mask_path is None
        assert out.exists()
        # cv2 reads BGR; save_image converted RGB→BGR, so re-loading and
        # converting back should match the original.
        loaded = load_image(out)
        # PNG via OpenCV preserves values for uint8; channel order matches.
        assert loaded.shape == img.shape

    def test_tiff_with_mask(self, tmp_path):
        img = self._img_hwc()
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        out = tmp_path / "out.tif"
        mask_path = save_image(img, out, mask=mask)
        assert mask_path is not None
        assert mask_path.exists()
        assert mask_path.name == "out_mask.tif"

        import tifffile

        m = tifffile.imread(str(mask_path))
        assert m.shape == (16, 16)
        assert m.dtype == np.uint8
        assert m[5, 5] == 255
        assert m[0, 0] == 0

    def test_chw_input_transposed(self, tmp_path):
        img_chw = np.zeros((3, 16, 16), dtype=np.uint8)
        img_chw[0] = 100  # red channel
        out = tmp_path / "out.tif"
        save_image(img_chw, out)
        import tifffile

        loaded = tifffile.imread(str(out))
        assert loaded.shape == (16, 16, 3)
        assert loaded[0, 0, 0] == 100


# ---------------------------------------------------------------------------
# CalibrationResult.save_stats
# ---------------------------------------------------------------------------


class TestCalibrationResultSaveStats:
    def test_writes_expected_keys(self, tmp_path):
        successful = [
            ImageCalibrationResult(
                roi_id="0001",
                success=True,
                microscope_position=np.array([0.0, 0.0]),
                z_position=0.0,
            ),
            ImageCalibrationResult(
                roi_id="0002",
                success=True,
                microscope_position=np.array([1.0, 1.0]),
                z_position=0.0,
            ),
        ]
        failed = ImageCalibrationResult(
            roi_id="0003",
            success=False,
            microscope_position=None,
            z_position=None,
            error_message="bad image",
        )
        transform = AffineTransformResult(
            transform=lambda x: x,
            residuals=np.array([0.1, 0.2]),
            rmse=0.158,
            max_error=0.2,
        )
        result = CalibrationResult(
            measured_map=_make_map([("0001", [0, 0]), ("0002", [1, 1])]),
            transform_result=transform,
            calibrated_map=_make_map([("0001", [0, 0])]),
            image_results=[*successful, failed],
        )
        out = tmp_path / "stats.json"
        result.save_stats(out)
        data = json.loads(out.read_text())
        assert set(data) == {"transform_stats", "failed_images"}
        ts = data["transform_stats"]
        assert set(ts) == {"rmse", "max_error", "n_calibration_points", "residuals"}
        assert ts["n_calibration_points"] == 2
        assert set(ts["residuals"]) == {"0001", "0002"}
        assert data["failed_images"] == [{"roi_id": "0003", "error": "bad image"}]


# ---------------------------------------------------------------------------
# ValidationSummary.to_csv / from_csv
# ---------------------------------------------------------------------------


class TestValidationSummaryCsv:
    def _summary(self):
        results = [
            ValidationResult(
                roi_id="0001",
                success=True,
                map_x=1.0,
                map_y=2.0,
                measured_x=1.1,
                measured_y=2.1,
                error=0.14,
            ),
            ValidationResult(
                roi_id="0002",
                success=False,
                map_x=None,
                map_y=None,
                measured_x=None,
                measured_y=None,
                error=None,
                error_message="failed",
            ),
        ]
        return ValidationSummary(
            results=results,
            mean_error=0.14,
            median_error=0.14,
            std_error=0.0,
            max_error=0.14,
            min_error=0.14,
            p90_error=0.14,
            n_success=1,
            n_failed=1,
        )

    def test_round_trip(self, tmp_path):
        summary = self._summary()
        out = tmp_path / "results.csv"
        summary.to_csv(out, pixel_size=0.1)
        # to_csv mutates results to populate error_px
        assert summary.results[0].error_px == pytest.approx(1.4)
        loaded = ValidationSummary.from_csv(out)
        assert loaded.n_success == 1
        assert loaded.n_failed == 1
        assert loaded.results[0].error == pytest.approx(0.14)
        assert loaded.results[0].error_px == pytest.approx(1.4)
        assert loaded.results[1].error is None

"""Tests for dart_mlci.script_utils module."""

import json
from pathlib import Path

import pytest

from dart_mlci.script_utils import (
    Timer,
    get_peak_gpu_memory_mb,
    load_image_list,
    load_json_config,
    reset_gpu_memory_stats,
    validate_calibration_config,
    validate_validation_config,
)


def _three_image_paths(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.tif"
        p.write_bytes(b"x")
        paths.append(p)
    return paths


def _calibration_cfg(
    image_paths, *, blueprint_path: Path | None = None, chip_config_path: Path | None = None
):
    cfg = {
        "calibration_images": [
            {
                "image_path": str(p),
                "roi_id": f"{i:04d}",
                "stage_position": {"x": float(i), "y": float(i)},
            }
            for i, p in enumerate(image_paths)
        ],
        "pixel_size": 0.065789,
    }
    if blueprint_path is not None:
        cfg["blueprint_map_path"] = str(blueprint_path)
    if chip_config_path is not None:
        cfg["chip_config_path"] = str(chip_config_path)
    return cfg


class TestLoadJsonConfig:
    def test_load_valid_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"key1": "val1", "key2": 42}))
        result = load_json_config(config_path)
        assert result == {"key1": "val1", "key2": 42}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json_config(tmp_path / "nonexistent.json")

    def test_required_keys_present(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"a": 1, "b": 2}))
        result = load_json_config(config_path, required_keys=["a", "b"])
        assert result["a"] == 1

    def test_required_keys_missing_raises(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"a": 1}))
        with pytest.raises(ValueError, match="missing required keys"):
            load_json_config(config_path, required_keys=["a", "b", "c"])


class TestLoadImageList:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "image_path,chamber_type\n/a/b.tif,NormaleBox-inner\n/c/d.png,BigBox-inner\n"
        )
        result = load_image_list(csv_path)
        assert len(result) == 2
        assert result[0] == ("/a/b.tif", "NormaleBox-inner")
        assert result[1] == ("/c/d.png", "BigBox-inner")

    def test_strips_whitespace(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("image_path,chamber_type\n /a/b.tif , NormaleBox-inner \n")
        result = load_image_list(csv_path)
        assert result[0] == ("/a/b.tif", "NormaleBox-inner")


class TestTimer:
    def test_timer_measures_time(self):
        import time

        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04
        assert t.elapsed < 1.0

    def test_timer_default_zero(self):
        t = Timer()
        assert t.elapsed == 0.0


class TestValidateCalibrationConfig:
    """Happy path + all error branches of validate_calibration_config."""

    def test_happy_path_with_chip_config(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        chip = tmp_path / "chip.json"
        chip.write_text("{}")
        validate_calibration_config(_calibration_cfg(paths, chip_config_path=chip))

    def test_missing_required_field(self):
        with pytest.raises(ValueError, match="Missing required field"):
            validate_calibration_config({})

    def test_missing_blueprint_source(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        cfg = _calibration_cfg(paths)  # neither blueprint_map_path nor chip_config_path
        with pytest.raises(ValueError, match="blueprint_map_path"):
            validate_calibration_config(cfg)

    def test_cal_images_not_a_list(self, tmp_path):
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = {
            "calibration_images": "not a list",
            "pixel_size": 0.1,
            "blueprint_map_path": str(bp),
        }
        with pytest.raises(ValueError, match="must be a list"):
            validate_calibration_config(cfg)

    def test_too_few_images(self, tmp_path):
        p = tmp_path / "only.tif"
        p.write_bytes(b"x")
        cfg = _calibration_cfg([p], blueprint_path=p)
        with pytest.raises(ValueError, match="at least 3"):
            validate_calibration_config(cfg)

    def test_image_entry_not_a_dict(self, tmp_path):
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = {
            "calibration_images": ["a", "b", "c"],
            "pixel_size": 0.1,
            "blueprint_map_path": str(bp),
        }
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_calibration_config(cfg)

    def test_image_entry_missing_stage_position(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = {
            "calibration_images": [
                {"image_path": str(p), "roi_id": f"{i:04d}"} for i, p in enumerate(paths)
            ],
            "pixel_size": 0.1,
            "blueprint_map_path": str(bp),
        }
        with pytest.raises(ValueError, match="stage_position"):
            validate_calibration_config(cfg)

    def test_stage_position_not_a_dict(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = {
            "calibration_images": [
                {
                    "image_path": str(p),
                    "roi_id": f"{i:04d}",
                    "stage_position": "x=1,y=2",
                }
                for i, p in enumerate(paths)
            ],
            "pixel_size": 0.1,
            "blueprint_map_path": str(bp),
        }
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_calibration_config(cfg)

    def test_stage_position_missing_keys(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = {
            "calibration_images": [
                {
                    "image_path": str(p),
                    "roi_id": f"{i:04d}",
                    "stage_position": {"x": 1.0},  # missing y
                }
                for i, p in enumerate(paths)
            ],
            "pixel_size": 0.1,
            "blueprint_map_path": str(bp),
        }
        with pytest.raises(ValueError, match="stage_position is missing"):
            validate_calibration_config(cfg)

    def test_missing_image_file(self, tmp_path):
        cfg = _calibration_cfg(
            [tmp_path / f"missing_{i}.tif" for i in range(3)],
            blueprint_path=tmp_path / "missing_blueprint.csv",
        )
        with pytest.raises(ValueError, match="File not found"):
            validate_calibration_config(cfg)

    def test_bad_pixel_size(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = _calibration_cfg(paths, blueprint_path=bp)
        cfg["pixel_size"] = -1.0
        with pytest.raises(ValueError, match="positive number"):
            validate_calibration_config(cfg)

    def test_chip_config_path_missing(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        cfg = _calibration_cfg(paths, chip_config_path=tmp_path / "missing_chip.json")
        with pytest.raises(ValueError, match=r"chip_config_path.*File not found"):
            validate_calibration_config(cfg)

    def test_model_path_missing(self, tmp_path):
        paths = _three_image_paths(tmp_path)
        bp = tmp_path / "bp.csv"
        bp.write_text("roi_id,x,y\n")
        cfg = _calibration_cfg(paths, blueprint_path=bp)
        cfg["model_path"] = str(tmp_path / "no_model.pt")
        with pytest.raises(ValueError, match=r"model_path.*File not found"):
            validate_calibration_config(cfg)


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


class TestGpuHelpers:
    """No-op gracefully when CUDA / torch unavailable."""

    def test_get_peak_returns_float(self):
        val = get_peak_gpu_memory_mb()
        assert isinstance(val, float)
        assert val >= 0.0

    def test_reset_does_not_raise(self):
        reset_gpu_memory_stats()

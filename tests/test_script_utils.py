"""Tests for dart_mlci.script_utils module."""

import json

import pytest

from dart_mlci.script_utils import Timer, load_image_list, load_json_config


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

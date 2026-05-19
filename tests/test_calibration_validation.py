"""Tests for dart_mlci.calibration.validation module."""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from dart_mlci.calibration.validation import (
    ValidationDebugData,
    ValidationResult,
    ValidationSummary,
    process_validation_image,
)
from dart_mlci.mask import RoIPolygon


class TestValidationResult:
    def test_construction_success(self):
        r = ValidationResult(
            roi_id="0050",
            success=True,
            map_x=100.0,
            map_y=200.0,
            measured_x=100.5,
            measured_y=200.3,
            error=0.58,
        )
        assert r.success
        assert r.error == 0.58

    def test_construction_failure(self):
        r = ValidationResult(
            roi_id="0050",
            success=False,
            map_x=100.0,
            map_y=200.0,
            measured_x=None,
            measured_y=None,
            error=None,
            error_message="DETECTION: No markers found",
        )
        assert not r.success
        assert r.error_message is not None


class TestValidationDebugData:
    def test_defaults_are_none(self):
        d = ValidationDebugData()
        assert d.image is None
        assert d.markers is None
        assert d.error_microns is None


class TestValidationSummary:
    def test_construction(self):
        s = ValidationSummary(
            results=[],
            mean_error=1.0,
            median_error=0.8,
            std_error=0.3,
            max_error=2.0,
            min_error=0.1,
            p90_error=1.5,
            n_success=10,
            n_failed=2,
        )
        assert s.mean_error == 1.0
        assert s.n_success == 10


class TestValidationSummaryCsv:
    """ValidationSummary.to_csv / from_csv round-trip."""

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


# ---------------------------------------------------------------------------
# process_validation_image — failure paths (the most user-visible behaviours)
# ---------------------------------------------------------------------------


def _square_roi(size: int = 200) -> RoIPolygon:
    return RoIPolygon(Polygon([(0, 0), (size, 0), (size, size), (0, size)]))


def _write_blank_png(path, size: int = 256) -> None:
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _structure_library(roi_polygon, marker_group):
    """Mock structure_library callable: roi_id → (name, polygon, group)."""
    lib = MagicMock()
    lib.side_effect = lambda roi_id: ("NormaleBox-inner", roi_polygon, marker_group)
    return lib


class TestProcessValidationImageFailurePaths:
    """Each early-return failure path emits a ValidationResult with the right reason."""

    def _common(self, tmp_path):
        img_path = tmp_path / "validation.png"
        _write_blank_png(img_path)
        roi = _square_roi()
        # Realistic 4-marker group (2 crosses + 2 circles).
        marker_group = {
            "cross": np.array([[10.0, 10.0], [190.0, 10.0]]),
            "circle": np.array([[10.0, 190.0], [190.0, 190.0]]),
        }
        lib = _structure_library(roi, marker_group)
        expected = np.array([1000.0, 2000.0])
        return img_path, roi, marker_group, lib, expected

    def test_no_markers_returns_detection_error(self, tmp_path):
        img_path, _, _, lib, expected = self._common(tmp_path)
        detect = MagicMock(side_effect=lambda image: {"image": image, "markers": []})

        result = process_validation_image(
            image_path=img_path,
            roi_id="0050",
            stage_position={"x": 0.0, "y": 0.0, "z": 0.0},
            expected_position=expected,
            detection_step=detect,
            structure_library=lib,
            pixel_size=0.065789,
        )
        assert result.success is False
        assert result.error_message.startswith("DETECTION")
        # Map position is always populated, measured is None.
        assert result.map_x == expected[0] and result.map_y == expected[1]
        assert result.measured_x is None and result.measured_y is None

    def test_low_confidence_markers_filtered_out(self, tmp_path):
        """Markers below conf_threshold are dropped → falls through to DETECTION error."""
        img_path, _, _, lib, expected = self._common(tmp_path)
        low_conf = [
            {"bbox_center": np.array([10.0, 10.0]), "label": "cross", "conf": 0.1},
            {"bbox_center": np.array([90.0, 90.0]), "label": "circle", "conf": 0.1},
        ]
        detect = MagicMock(side_effect=lambda image: {"image": image, "markers": low_conf})

        result = process_validation_image(
            image_path=img_path,
            roi_id="0050",
            stage_position={"x": 0.0, "y": 0.0, "z": 0.0},
            expected_position=expected,
            detection_step=detect,
            structure_library=lib,
            pixel_size=0.065789,
            conf_threshold=0.5,
        )
        assert result.success is False
        assert result.error_message.startswith("DETECTION")

    def test_no_matched_pairs_returns_matching_error(self, tmp_path):
        """Markers exist but matching can't form pairs (mismatched labels) → MATCHING."""
        img_path, _, _, lib, expected = self._common(tmp_path)
        # All crosses, no circles → matching will not produce pairs.
        markers = [
            {"bbox_center": np.array([10.0, 10.0]), "label": "cross", "conf": 0.9},
            {"bbox_center": np.array([20.0, 20.0]), "label": "cross", "conf": 0.9},
        ]
        detect = MagicMock(side_effect=lambda image: {"image": image, "markers": markers})

        result = process_validation_image(
            image_path=img_path,
            roi_id="0050",
            stage_position={"x": 0.0, "y": 0.0, "z": 0.0},
            expected_position=expected,
            detection_step=detect,
            structure_library=lib,
            pixel_size=0.065789,
        )
        assert result.success is False
        assert result.error_message.startswith("MATCHING")

    def test_debug_data_collected_when_requested(self, tmp_path):
        img_path, _, _, lib, expected = self._common(tmp_path)
        detect = MagicMock(side_effect=lambda image: {"image": image, "markers": []})

        result = process_validation_image(
            image_path=img_path,
            roi_id="0050",
            stage_position={"x": 0.0, "y": 0.0, "z": 0.0},
            expected_position=expected,
            detection_step=detect,
            structure_library=lib,
            pixel_size=0.065789,
            collect_debug=True,
        )
        assert result.debug_data is not None
        assert result.debug_data.image is not None
        assert result.debug_data.pixel_size == 0.065789
        assert result.debug_data.structure_name == "NormaleBox-inner"

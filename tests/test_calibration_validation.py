"""Tests for dart_mlci.calibration.validation module."""

import pytest

from dart_mlci.calibration.validation import (
    ValidationDebugData,
    ValidationResult,
    ValidationSummary,
)


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

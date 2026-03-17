"""Tests for dart_mlci.calibration.validation module."""

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

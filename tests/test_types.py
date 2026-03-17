"""Tests for dart_mlci.types module."""

import pytest

from dart_mlci.types import FrameResult, PipelineError, PipelineTimings, StackResult


class TestPipelineError:
    def test_attributes(self):
        err = PipelineError("DETECTION", "No markers found")
        assert err.step == "DETECTION"
        assert err.message == "No markers found"
        assert "DETECTION" in str(err)
        assert "No markers found" in str(err)

    def test_is_exception(self):
        with pytest.raises(PipelineError):
            raise PipelineError("MATCHING", "fail")


class TestPipelineTimings:
    def test_defaults(self):
        t = PipelineTimings()
        assert t.detection == 0.0
        assert t.total == 0.0

    def test_total_sums_all(self):
        t = PipelineTimings(
            detection=1.0,
            matching=2.0,
            rotation=3.0,
            registration=4.0,
            masking=5.0,
            segmentation=6.0,
        )
        assert t.total == 21.0

    def test_as_dict(self):
        t = PipelineTimings(detection=1.5, matching=0.5)
        d = t.as_dict()
        assert d["t_detection"] == 1.5
        assert d["t_matching"] == 0.5
        assert d["t_total"] == 2.0
        assert "t_registration" in d


class TestFrameResult:
    def test_defaults(self):
        fr = FrameResult()
        assert fr.success is False
        assert fr.n_cells == 0
        assert fr.timings is not None
        assert fr.timings.total == 0.0

    def test_with_values(self):
        fr = FrameResult(success=True, n_cells=5)
        assert fr.success is True
        assert fr.n_cells == 5


class TestStackResult:
    def test_construction(self):
        sr = StackResult(folder="exp1", file_name="stack.tif", chamber_type="NormaleBox-inner")
        assert sr.folder == "exp1"
        assert sr.n_frames == 0
        assert sr.success is False
        assert sr.error == ""

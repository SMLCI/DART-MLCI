"""Tests for dart_mlci.timelapse module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dart_mlci.timelapse import TimelapseProcessor, TimelapseResult, create_segmenter

# ---------------------------------------------------------------------------
# Helpers for TimelapseProcessor.process_stack orchestration tests
# ---------------------------------------------------------------------------


def _make_detection_step(fail_frames: set[int] | None = None):
    """Return a mock detection step that returns dummy markers per frame.

    Tracks which frame it's called on via the call count; if the index is in
    fail_frames, returns no markers so matching downstream raises.
    """
    fail_frames = fail_frames or set()
    state = {"call_count": 0}

    def detect(frame):
        idx = state["call_count"]
        state["call_count"] += 1
        if idx in fail_frames:
            return {"image": frame, "markers": []}
        return {
            "image": frame,
            "markers": [
                {"bbox_center": np.array([10.0, 10.0]), "label": "cross", "conf": 0.9},
                {"bbox_center": np.array([90.0, 90.0]), "label": "circle", "conf": 0.9},
            ],
        }

    step = MagicMock(side_effect=detect)
    return step


def _make_pipeline_cache(structure_name: str, h: int = 40, w: int = 50):
    """Return a mock ChamberPipelineCache that hands out callable pipeline steps."""

    def matching(data):
        # Realistic behavior: no markers → no matched pairs.
        matched = [(0, 1)] if data.get("markers") else []
        return {**data, "matched_marker_indices": matched, "angle": 0.0}

    def rotation(data):
        return {**data}

    def masking(data):
        # Return cropped image + chamber mask
        img = np.full((h, w, 3), 100, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=bool)
        mask[:5, :] = True  # mark some pixels as outside-chamber
        return {"image": img, "mask": mask, "markers": data["markers"]}

    components = {
        "roi_polygon": None,
        "marker_group": None,
        "matching_step": MagicMock(side_effect=matching),
        "rotation_step": MagicMock(side_effect=rotation),
        "masking_step": MagicMock(side_effect=masking),
    }

    cache = MagicMock()
    cache.get.side_effect = lambda name: (
        components if name == structure_name else (_ for _ in ()).throw(KeyError(name))
    )
    return cache


# ---------------------------------------------------------------------------


class TestTimelapseResult:
    def test_defaults(self):
        r = TimelapseResult()
        assert r.n_frames == 0
        assert r.success is False
        assert r.cell_mask_stack is None

    def test_with_values(self):
        r = TimelapseResult(n_frames=10, n_success=8, success=True)
        assert r.n_frames == 10
        assert r.n_success == 8


class TestCreateSegmenter:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown segmenter"):
            create_segmenter("nonexistent")

    def test_cellpose_sam_requires_acia(self):
        try:
            seg = create_segmenter("cellpose-sam")
            assert seg is not None
        except ImportError:
            pytest.skip("acia not available")


class TestTimelapseProcessorAssembleStacks:
    def test_assemble_stacks_basic(self):
        """Test stack assembly with consistent frame sizes."""
        frame_data = [
            {"success": True, "n_cells": 2},
            {"success": False, "n_cells": 0},
            {"success": True, "n_cells": 1},
        ]
        images = [
            np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8),
        ]
        masks = [
            np.zeros((50, 60), dtype=np.uint16),
            np.zeros((50, 60), dtype=np.uint16),
        ]
        chambers = [
            np.zeros((50, 60), dtype=bool),
            np.zeros((50, 60), dtype=bool),
        ]

        result = TimelapseProcessor._assemble_stacks(frame_data, images, masks, chambers)
        assert result["image_stack"].shape == (3, 50, 60, 3)
        assert result["cell_mask_stack"].shape == (3, 50, 60)
        assert result["chamber_mask_stack"].shape == (3, 50, 60)

    def test_assemble_stacks_different_sizes(self):
        """Test stack assembly pads smaller frames."""
        frame_data = [
            {"success": True, "n_cells": 0},
            {"success": True, "n_cells": 0},
        ]
        images = [
            np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8),
            np.random.randint(0, 255, (48, 58, 3), dtype=np.uint8),
        ]
        masks = [
            np.zeros((50, 60), dtype=np.uint16),
            np.zeros((48, 58), dtype=np.uint16),
        ]
        chambers = [
            np.zeros((50, 60), dtype=bool),
            np.zeros((48, 58), dtype=bool),
        ]

        result = TimelapseProcessor._assemble_stacks(frame_data, images, masks, chambers)
        assert result["image_stack"].shape == (2, 50, 60, 3)


class TestTimelapseProcessorProcessStack:
    """End-to-end orchestration of process_stack with mocked pipeline steps."""

    @pytest.fixture
    def frames(self):
        # 3 frames, 32x32 grayscale uint8.
        return np.full((3, 32, 32), 128, dtype=np.uint8)

    def test_unknown_chamber_returns_error(self, frames):
        det = _make_detection_step()
        cache = _make_pipeline_cache("KnownBox")
        proc = TimelapseProcessor(det, cache)

        result = proc.process_stack(frames, structure_name="UnknownBox")
        assert result.success is False
        assert "UnknownBox" in result.error
        assert result.n_success == 0

    def test_all_frames_fail_detection(self, frames):
        # No markers detected for any frame → matching raises → all fail.
        det = _make_detection_step(fail_frames={0, 1, 2})
        cache = _make_pipeline_cache("NormaleBox-inner")
        proc = TimelapseProcessor(det, cache)

        result = proc.process_stack(frames, structure_name="NormaleBox-inner")
        assert result.success is False
        assert "failed detection" in result.error.lower()
        assert result.n_frames == 3
        assert result.n_success == 0

    def test_happy_path_no_segmenter(self, frames):
        det = _make_detection_step()
        cache = _make_pipeline_cache("NormaleBox-inner")
        proc = TimelapseProcessor(det, cache)

        result = proc.process_stack(frames, structure_name="NormaleBox-inner")
        assert result.success is True
        assert result.n_frames == 3
        assert result.n_success == 3
        # No segmenter ⇒ no cells.
        assert result.n_cells_total == 0
        # Stacks exist and have T = 3 frames.
        assert result.cropped_image_stack.shape[0] == 3
        assert result.cell_mask_stack.shape[0] == 3
        assert result.chamber_mask_stack.shape[0] == 3

    def test_partial_failure_accounting(self, frames):
        # Middle frame fails detection.
        det = _make_detection_step(fail_frames={1})
        cache = _make_pipeline_cache("NormaleBox-inner")
        proc = TimelapseProcessor(det, cache)

        result = proc.process_stack(frames, structure_name="NormaleBox-inner")
        assert result.success is True
        assert result.n_frames == 3
        assert result.n_success == 2
        # The failed frame has error recorded.
        assert result.frame_data[1]["success"] is False
        assert result.frame_data[1]["error"]

    def test_single_frame_2d_input_is_promoted(self):
        # A single HxW frame (no T axis) should be promoted to T=1.
        single = np.full((32, 32), 128, dtype=np.uint8)
        det = _make_detection_step()
        cache = _make_pipeline_cache("NormaleBox-inner")
        proc = TimelapseProcessor(det, cache)

        result = proc.process_stack(single, structure_name="NormaleBox-inner")
        assert result.n_frames == 1
        assert result.success is True
        assert result.cropped_image_stack.shape[0] == 1

    def test_segmenter_results_are_filtered_by_chamber_mask(self, frames):
        # Mock segmenter that returns a labeled mask with 3 cells, then filter
        # should keep only cells fully inside the chamber.
        det = _make_detection_step()
        cache = _make_pipeline_cache("NormaleBox-inner")

        def fake_segment(self, cropped_image):
            labeled = np.zeros(cropped_image.shape[:2], dtype=np.uint16)
            # cell 1 inside chamber, cell 2 partially outside (first 5 rows are outside)
            labeled[20:25, 20:25] = 1
            labeled[0:7, 0:7] = 2  # mostly outside
            return labeled

        # Inject by binding directly so we don't need a real acia segmenter.
        segmenter = MagicMock()
        proc = TimelapseProcessor(det, cache, segmenter=segmenter, filter_threshold=0.5)
        proc._segment_frame = fake_segment.__get__(proc, TimelapseProcessor)

        result = proc.process_stack(frames, structure_name="NormaleBox-inner")
        assert result.success is True
        # cell 2 should be filtered out (mostly outside chamber); cell 1 stays.
        # Each successful frame contributes one surviving cell.
        assert result.n_cells_total == 3

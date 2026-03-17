"""Tests for dart_mlci.timelapse module."""

import numpy as np
import pytest

from dart_mlci.timelapse import TimelapseProcessor, TimelapseResult, create_segmenter


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

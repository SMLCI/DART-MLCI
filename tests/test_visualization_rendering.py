"""Tests for dart_mlci.visualization.rendering module."""

import numpy as np
import pytest


class TestRenderCellVisualization:
    @pytest.fixture(autouse=True)
    def _skip_if_no_acia(self):
        pytest.importorskip("acia")

    def test_basic_rendering(self):
        from dart_mlci.visualization.rendering import render_cell_visualization

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        labeled_mask = np.zeros((100, 100), dtype=np.uint16)
        labeled_mask[30:50, 30:50] = 1
        chamber_mask = np.zeros((100, 100), dtype=bool)
        chamber_mask[:10, :] = True

        result = render_cell_visualization(
            cropped_image=image,
            labeled_mask=labeled_mask,
            chamber_mask=chamber_mask,
            pixel_size=0.065789,
            scalebar=False,
        )
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8


class TestAddScalebar:
    @pytest.fixture(autouse=True)
    def _skip_if_no_acia(self):
        pytest.importorskip("acia")

    def test_add_scalebar_returns_image(self):
        from dart_mlci.visualization.rendering import add_scalebar

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = add_scalebar(image, pixel_size=0.065789)
        assert result.shape[:2] == (100, 100)
        assert result.dtype == np.uint8

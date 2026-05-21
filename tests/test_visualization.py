"""Tests for the dart_mlci.visualization subpackage.

Smoke tests that exercise each visualization helper end-to-end with small
synthetic inputs and verify it actually produces output (pixels written,
files saved, frames of expected shape) — not just that it doesn't raise.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

from dart_mlci.mask import RoIPolygon
from dart_mlci.visualization.drawing import (
    add_step_title,
    draw_markers_cv,
    draw_matched_pairs_cv,
    draw_roi_polygon,
    prepare_frame,
)
from dart_mlci.visualization.plotting import (
    plot_marker_pairs,
    plot_markers,
    plot_markers_on_image,
    render_markers_to_frame,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_image():
    # 200x300 BGR image, mid-gray so changes show up clearly.
    return np.full((200, 300, 3), 128, dtype=np.uint8)


@pytest.fixture
def two_markers():
    """One cross + one circle, well separated for visible drawing."""
    return [
        {"bbox_center": np.array([60.0, 80.0]), "label": "cross", "conf": 0.91},
        {"bbox_center": np.array([220.0, 140.0]), "label": "circle", "conf": 0.85},
    ]


@pytest.fixture
def square_roi():
    return RoIPolygon(Polygon([(40, 40), (180, 40), (180, 160), (40, 160)]))


# ---------------------------------------------------------------------------
# drawing.py — OpenCV-based functions
# ---------------------------------------------------------------------------


class TestPrepareFrame:
    def test_returned_shape_matches_target(self, small_image):
        frame, scale, offset = prepare_frame(small_image, frame_size=(640, 480))
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        assert scale > 0
        assert isinstance(offset, tuple) and len(offset) == 2

    def test_grayscale_input_is_promoted_to_bgr(self):
        gray = np.full((100, 100), 200, dtype=np.uint8)
        frame, _, _ = prepare_frame(gray, frame_size=(320, 240))
        assert frame.ndim == 3
        assert frame.shape[2] == 3


class TestAddStepTitle:
    def test_pixels_in_title_band_change(self, small_image):
        frame = small_image.copy()
        before_band = frame[:60].copy()
        out = add_step_title(frame, "Detection")
        # The top 60-pixel band has been darkened and has text written on it.
        assert not np.array_equal(out[:60], before_band)


class TestDrawMarkersCv:
    def test_pixels_change_near_marker_centers(self, small_image, two_markers):
        baseline = small_image.copy()
        out = draw_markers_cv(small_image.copy(), two_markers, scale=1.0, offset=(0, 0))

        # Around each marker center, pixels must have been written.
        for m in two_markers:
            x, y = int(m["bbox_center"][0]), int(m["bbox_center"][1])
            region_before = baseline[max(0, y - 10) : y + 10, max(0, x - 10) : x + 10]
            region_after = out[max(0, y - 10) : y + 10, max(0, x - 10) : x + 10]
            assert not np.array_equal(region_before, region_after)

    def test_highlight_indices_use_distinct_color(self, small_image, two_markers):
        plain = draw_markers_cv(small_image.copy(), two_markers, scale=1.0, offset=(0, 0))
        highlighted = draw_markers_cv(
            small_image.copy(),
            two_markers,
            scale=1.0,
            offset=(0, 0),
            highlight_indices=[0],
        )
        # The highlighted region should differ from the plain region for marker 0.
        x, y = int(two_markers[0]["bbox_center"][0]), int(two_markers[0]["bbox_center"][1])
        plain_patch = plain[max(0, y - 10) : y + 10, max(0, x - 10) : x + 10]
        hl_patch = highlighted[max(0, y - 10) : y + 10, max(0, x - 10) : x + 10]
        assert not np.array_equal(plain_patch, hl_patch)


class TestDrawMatchedPairsCv:
    def test_line_drawn_between_pair(self, small_image, two_markers):
        baseline = small_image.copy()
        out = draw_matched_pairs_cv(
            small_image.copy(),
            two_markers,
            matched_indices=[(0, 1)],
            scale=1.0,
            offset=(0, 0),
        )
        # Sample a point on the line between the two centers — pixels there must change.
        mid_x = int((two_markers[0]["bbox_center"][0] + two_markers[1]["bbox_center"][0]) / 2)
        mid_y = int((two_markers[0]["bbox_center"][1] + two_markers[1]["bbox_center"][1]) / 2)
        assert not np.array_equal(baseline[mid_y, mid_x], out[mid_y, mid_x])


class TestDrawRoiPolygon:
    def test_polygon_overlay_modifies_inside(self, small_image, square_roi):
        out = draw_roi_polygon(small_image.copy(), square_roi, scale=1.0, offset=(0, 0))
        inside_before = small_image[100, 100]
        inside_after = out[100, 100]
        assert not np.array_equal(inside_before, inside_after)

    def test_inverted_dims_outside(self, small_image, square_roi):
        out = draw_roi_polygon(
            small_image.copy(), square_roi, scale=1.0, offset=(0, 0), inverted=True
        )
        # Outside the polygon should be darker than baseline 128.
        outside_pixel = out[10, 10]
        assert outside_pixel[0] < 128


# ---------------------------------------------------------------------------
# plotting.py — matplotlib-based functions
# ---------------------------------------------------------------------------


class TestPlotMarkersOnImage:
    def test_saves_file_when_output_path_provided(self, small_image, two_markers, tmp_path):
        out = tmp_path / "markers.png"
        plot_markers_on_image(small_image, two_markers, matched_indices=[(0, 1)], output_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000  # not an empty file

    def test_no_file_written_without_output_path(self, small_image, two_markers, tmp_path):
        # Should run without error and write nothing.
        plot_markers_on_image(small_image, two_markers, matched_indices=[(0, 1)])
        assert list(tmp_path.iterdir()) == []

    def test_grayscale_input_accepted(self, two_markers, tmp_path):
        gray = np.full((200, 300), 100, dtype=np.uint8)
        out = tmp_path / "markers_gray.png"
        plot_markers_on_image(gray, two_markers, matched_indices=[], output_path=out)
        assert out.exists()

    def test_highlight_indices_run(self, small_image, two_markers, tmp_path):
        out = tmp_path / "markers_hl.png"
        plot_markers_on_image(
            small_image,
            two_markers,
            matched_indices=[(0, 1)],
            output_path=out,
            highlight_indices=[0],
            selected_pair_idx=0,
        )
        assert out.exists()


class TestPlotMarkersWrappers:
    def test_plot_markers_writes_file(self, small_image, two_markers, tmp_path):
        out = tmp_path / "simple.png"
        plot_markers(small_image, two_markers, output_path=out)
        assert out.exists()

    def test_plot_marker_pairs_writes_file(self, small_image, two_markers, tmp_path):
        out = tmp_path / "pairs.png"
        plot_marker_pairs(small_image, [(0, 1)], two_markers, output_path=out)
        assert out.exists()


class TestRenderMarkersToFrame:
    def test_returns_bgr_frame_of_target_size(self, small_image, two_markers):
        frame = render_markers_to_frame(
            small_image, two_markers, matched_indices=[(0, 1)], frame_size=(640, 480)
        )
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_faded_indices_supported(self, small_image, two_markers):
        frame = render_markers_to_frame(
            small_image,
            two_markers,
            matched_indices=[],
            frame_size=(320, 240),
            faded_indices=[0],
        )
        assert frame.shape == (240, 320, 3)

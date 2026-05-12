"""Tests for ``dart_mlci.experiment`` frame-selection helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from dart_mlci.experiment import (
    absolutize_image_paths,
    load_tif_frame,
    resolve_chamber_type_from_folder_config,
    resolve_time_column,
    select_timelapse_frame,
)


def _make_df(time_col: str = "time") -> pd.DataFrame:
    # Intentionally out of chronological order to verify sort-by-time behavior.
    return pd.DataFrame(
        {
            "image_file": [
                "roi7_t02.png",
                "roi7_t00.png",
                "roi7_t01.png",
                "roi9_t00.png",
                "roi9_t01.png",
            ],
            "roi_id": ["0007", "0007", "0007", "0009", "0009"],
            time_col: [20.0, 0.0, 10.0, 0.0, 10.0],
        }
    )


def test_resolve_time_column_prefers_time():
    df = pd.DataFrame({"time": [0], "timestamp": [0]})
    assert resolve_time_column(df) == "time"


def test_resolve_time_column_falls_back_to_timestamp():
    df = pd.DataFrame({"timestamp": [0]})
    assert resolve_time_column(df) == "timestamp"


def test_resolve_time_column_raises_when_missing():
    with pytest.raises(ValueError, match=r"time.*timestamp"):
        resolve_time_column(pd.DataFrame({"foo": [1]}))


def test_absolutize_image_paths_leaves_absolute_untouched(tmp_path: Path):
    abs_path = str((tmp_path / "already.png").resolve())
    df = pd.DataFrame({"image_file": ["rel.png", abs_path]})
    out = absolutize_image_paths(df, tmp_path)
    assert out["image_file"].iloc[0] == str(tmp_path / "rel.png")
    assert out["image_file"].iloc[1] == abs_path


def test_absolutize_image_paths_no_column_is_noop(tmp_path: Path):
    df = pd.DataFrame({"other": [1, 2]})
    out = absolutize_image_paths(df, tmp_path)
    assert list(out.columns) == ["other"]


def test_select_timelapse_frame_returns_nth_by_time(tmp_path: Path):
    df = _make_df()

    p0 = select_timelapse_frame(df, "0007", 0, tmp_path)
    p1 = select_timelapse_frame(df, "0007", 1, tmp_path)
    p2 = select_timelapse_frame(df, "0007", 2, tmp_path)

    assert p0 == tmp_path / "roi7_t00.png"
    assert p1 == tmp_path / "roi7_t01.png"
    assert p2 == tmp_path / "roi7_t02.png"


def test_select_timelapse_frame_accepts_unpadded_roi_id(tmp_path: Path):
    df = _make_df()
    assert select_timelapse_frame(df, "7", 0, tmp_path) == tmp_path / "roi7_t00.png"


def test_select_timelapse_frame_supports_timestamp_column(tmp_path: Path):
    df = _make_df(time_col="timestamp")
    assert select_timelapse_frame(df, "0007", 0, tmp_path) == tmp_path / "roi7_t00.png"


def test_select_timelapse_frame_explicit_time_col(tmp_path: Path):
    df = _make_df()
    df["custom"] = [99, 1, 2, 3, 4]
    # Sorting by "custom" means roi7_t00 (value 1) is first, roi7_t01 (2), roi7_t02 (99).
    p = select_timelapse_frame(df, "0007", 2, tmp_path, time_col="custom")
    assert p == tmp_path / "roi7_t02.png"


def test_select_timelapse_frame_unknown_time_col_raises(tmp_path: Path):
    df = _make_df()
    with pytest.raises(ValueError, match="time_col"):
        select_timelapse_frame(df, "0007", 0, tmp_path, time_col="nope")


def test_select_timelapse_frame_unknown_roi_raises(tmp_path: Path):
    df = _make_df()
    with pytest.raises(ValueError, match="No rows"):
        select_timelapse_frame(df, "9999", 0, tmp_path)


def test_select_timelapse_frame_out_of_range_raises(tmp_path: Path):
    df = _make_df()
    with pytest.raises(IndexError, match="out of range"):
        select_timelapse_frame(df, "0007", 5, tmp_path)
    with pytest.raises(IndexError):
        select_timelapse_frame(df, "0007", -1, tmp_path)


# ---------------------------------------------------------------------------
# load_tif_frame
# ---------------------------------------------------------------------------


def _write_stack(path: Path, n_frames: int, h: int = 8, w: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 65535, size=(n_frames, h, w), dtype=np.uint16)
    tifffile.imwrite(str(path), stack, photometric="minisblack", metadata={"axes": "TYX"})
    return stack


def test_load_tif_frame_returns_hwc3_uint8(tmp_path: Path):
    tif = tmp_path / "stack.tif"
    _write_stack(tif, n_frames=5)

    frame = load_tif_frame(tif, 2)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    # All three RGB channels identical (greyscale stacked 3x)
    np.testing.assert_array_equal(frame[..., 0], frame[..., 1])
    np.testing.assert_array_equal(frame[..., 1], frame[..., 2])


def test_load_tif_frame_selects_correct_frame(tmp_path: Path):
    tif = tmp_path / "stack.tif"
    _write_stack(tif, n_frames=4)

    # Different frames should (with very high probability) differ
    f0 = load_tif_frame(tif, 0)
    f3 = load_tif_frame(tif, 3)
    assert not np.array_equal(f0, f3)


def test_load_tif_frame_flip(tmp_path: Path):
    tif = tmp_path / "stack.tif"
    _write_stack(tif, n_frames=2)

    normal = load_tif_frame(tif, 0, flip=False)
    flipped = load_tif_frame(tif, 0, flip=True)
    np.testing.assert_array_equal(flipped, normal[::-1])


def test_load_tif_frame_single_2d_image(tmp_path: Path):
    tif = tmp_path / "single.tif"
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    tifffile.imwrite(str(tif), img)

    frame = load_tif_frame(tif, 0)
    assert frame.shape == (8, 8, 3)

    with pytest.raises(IndexError):
        load_tif_frame(tif, 1)


def test_load_tif_frame_out_of_range(tmp_path: Path):
    tif = tmp_path / "stack.tif"
    _write_stack(tif, n_frames=3)

    with pytest.raises(IndexError):
        load_tif_frame(tif, 5)
    with pytest.raises(IndexError):
        load_tif_frame(tif, -1)


# ---------------------------------------------------------------------------
# resolve_chamber_type_from_folder_config
# ---------------------------------------------------------------------------


def _make_folder_config(tmp_path: Path) -> Path:
    cfg = {"folders": {"Big Chambers": "BigBox-inner", "Small Chambers": "NormaleBox-inner"}}
    p = tmp_path / "folder_config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_resolve_chamber_type_from_folder_config_path(tmp_path: Path):
    cfg = _make_folder_config(tmp_path)
    tif = tmp_path / "Big Chambers" / "Kammer 1.tif"
    tif.parent.mkdir()
    tif.touch()

    assert resolve_chamber_type_from_folder_config(tif, cfg) == "BigBox-inner"


def test_resolve_chamber_type_accepts_dict(tmp_path: Path):
    tif = tmp_path / "Small Chambers" / "Kammer 2.tif"
    tif.parent.mkdir()
    tif.touch()
    cfg_dict = {"folders": {"Small Chambers": "NormaleBox-inner"}}

    assert resolve_chamber_type_from_folder_config(tif, cfg_dict) == "NormaleBox-inner"


def test_resolve_chamber_type_unknown_folder_raises(tmp_path: Path):
    cfg = _make_folder_config(tmp_path)
    tif = tmp_path / "Mystery Folder" / "x.tif"
    tif.parent.mkdir()
    tif.touch()

    with pytest.raises(ValueError, match="Mystery Folder"):
        resolve_chamber_type_from_folder_config(tif, cfg)


def test_resolve_chamber_type_malformed_config_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_folders_key": {}}))
    tif = tmp_path / "Big Chambers" / "x.tif"
    tif.parent.mkdir()
    tif.touch()

    with pytest.raises(ValueError, match="folders"):
        resolve_chamber_type_from_folder_config(tif, bad)


# ---------------------------------------------------------------------------
# select_timelapse_frame (continued)
# ---------------------------------------------------------------------------


def test_select_timelapse_frame_preserves_absolute_paths(tmp_path: Path):
    abs_img = str((tmp_path / "elsewhere" / "abs.png").resolve())
    df = pd.DataFrame(
        {
            "image_file": [abs_img, "rel.png"],
            "roi_id": ["0001", "0001"],
            "time": [0.0, 1.0],
        }
    )
    assert select_timelapse_frame(df, "0001", 0, tmp_path) == Path(abs_img)
    assert select_timelapse_frame(df, "0001", 1, tmp_path) == tmp_path / "rel.png"

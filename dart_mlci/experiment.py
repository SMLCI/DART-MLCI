"""Experiment metadata helpers for time-lapse datasets.

Two input styles are supported, matching the two Dart processing scripts:

* **Metadata style** (``process_experiment.py``): a ``meta.parquet`` / ``meta.csv``
  pairs each single-image file with ``roi_id`` and a time column.
  Use :func:`select_timelapse_frame` to resolve ``(roi_id, image_number)``
  to an image path.

* **Folder/TIFF style** (``process_folder.py``): a folder-per-chamber-type
  layout where each ``.tif`` is a ``(T, H, W)`` time-lapse stack.
  Use :func:`load_tif_frame` to pull out one frame and
  :func:`resolve_chamber_type_from_folder_config` to look up the chamber
  type from the folder name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dart_mlci.utils import normalize_image


def resolve_time_column(df: pd.DataFrame) -> str:
    """Return the temporal-ordering column name (``time`` or ``timestamp``)."""
    if "time" in df.columns:
        return "time"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError(
        f"Metadata must have a 'time' or 'timestamp' column. Available columns: {list(df.columns)}"
    )


def absolutize_image_paths(df: pd.DataFrame, image_base_dir: Path) -> pd.DataFrame:
    """Return a copy of *df* with ``image_file`` made absolute against *image_base_dir*."""
    if "image_file" not in df.columns:
        return df
    out = df.copy()
    out["image_file"] = out["image_file"].apply(
        lambda x: str(image_base_dir / x) if not Path(x).is_absolute() else x
    )
    return out


def select_timelapse_frame(
    df: pd.DataFrame,
    roi_id: str,
    image_number: int,
    image_base_dir: Path,
    time_col: str | None = None,
) -> Path:
    """Return the absolute image path for the N-th frame of an ROI.

    The ROI's rows are sorted by the time column and ``image_number`` is the
    0-based index into that sorted sequence — matching the ``timepoint`` that
    ``process_experiment.py`` assigns in its time-lapse stacking mode.

    Args:
        df: Experiment metadata with at least ``image_file``, ``roi_id`` and a
            ``time`` or ``timestamp`` column.
        roi_id: ROI identifier to select. Compared as string against padded
            and unpadded forms so callers can pass either ``"7"`` or ``"0007"``.
        image_number: 0-based timepoint index within the ROI.
        image_base_dir: Directory used to resolve relative ``image_file`` paths.
        time_col: Optional override for the temporal-ordering column.

    Returns:
        Absolute filesystem path to the selected frame's image.
    """
    if time_col is None:
        time_col = resolve_time_column(df)
    elif time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in metadata columns")

    df = absolutize_image_paths(df, image_base_dir)

    roi_id_str = str(roi_id)
    roi_id_padded = roi_id_str.zfill(4)
    roi_col_str = df["roi_id"].astype(str)
    mask = (roi_col_str == roi_id_str) | (roi_col_str.str.zfill(4) == roi_id_padded)
    roi_df = df[mask].sort_values(time_col).reset_index(drop=True)

    if len(roi_df) == 0:
        raise ValueError(f"No rows found for roi_id={roi_id!r}")
    if image_number < 0 or image_number >= len(roi_df):
        raise IndexError(
            f"image_number {image_number} out of range for roi_id={roi_id!r} "
            f"(ROI has {len(roi_df)} frames)"
        )

    return Path(roi_df.iloc[image_number]["image_file"])


def load_tif_frame(
    tif_path: Path,
    frame_index: int,
    flip: bool = False,
) -> np.ndarray:
    """Load one frame of a ``(T, H, W)`` TIFF stack as an HxWx3 uint8 RGB array.

    Normalizes identically to ``process_folder.py``: ``normalize_image`` →
    stack the single channel 3x into RGB. A single-frame ``(H, W)`` TIFF is
    treated as a one-frame stack (only ``frame_index=0`` is valid).

    Args:
        tif_path: Path to the TIFF stack.
        frame_index: 0-based frame index within the stack.
        flip: If True, vertically flip the frame (matches the ``flip`` option
            in folder configs).

    Returns:
        HxWx3 uint8 array ready to feed into the pipeline renderer.
    """
    import tifffile

    stack = tifffile.imread(str(tif_path))
    if stack.ndim == 2:
        if frame_index != 0:
            raise IndexError(
                f"frame_index {frame_index} out of range: TIFF {tif_path} "
                f"is a single 2D image (only frame 0 exists)"
            )
        frame_raw = stack
    elif stack.ndim == 3:
        n_frames = stack.shape[0]
        if frame_index < 0 or frame_index >= n_frames:
            raise IndexError(
                f"frame_index {frame_index} out of range for {tif_path} "
                f"(stack has {n_frames} frames)"
            )
        frame_raw = stack[frame_index]
    else:
        raise ValueError(f"Unexpected TIFF stack shape {stack.shape} for {tif_path}")

    if flip:
        frame_raw = frame_raw[::-1]
    frame_uint8 = normalize_image(frame_raw)
    return np.stack((frame_uint8,) * 3, axis=-1)


def resolve_chamber_type_from_folder_config(
    tif_path: Path,
    folder_config: Path | dict,
) -> str:
    """Resolve the chamber type for a TIFF using the ``folder_config.json`` map.

    The TIFF's immediate parent folder name is looked up in
    ``config["folders"]``, which maps ``folder_name -> chamber_type`` (same
    mapping format that ``process_folder.py`` consumes).

    Args:
        tif_path: Path to a ``.tif`` inside one of the configured folders.
        folder_config: Path to the JSON config file, or an already-loaded dict.

    Returns:
        Chamber type string, usable as a key into
        ``ChipStructureLibrary.polygon_library`` / ``.marker_group_configs``.
    """
    if isinstance(folder_config, (str, Path)):
        with open(folder_config) as f:
            config = json.load(f)
    else:
        config = folder_config

    folders = config.get("folders")
    if not isinstance(folders, dict):
        raise ValueError("folder_config must contain a 'folders' mapping")

    parent = tif_path.parent.name
    if parent not in folders:
        raise ValueError(
            f"TIFF's parent folder {parent!r} not found in folder_config.folders "
            f"(known folders: {sorted(folders)})"
        )
    return folders[parent]

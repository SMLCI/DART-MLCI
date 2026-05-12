"""Render the pipeline walkthrough for a single Dart experiment frame.

Two input modes are supported:

1. **Metadata mode** (``process_experiment.py`` layout) — specify
   ``--dataset-dir``, ``--roi-id``, ``--image-number``:
       python scripts/generate_dart_frame_video.py \\
           --dataset-dir /data/dart_experiment_2026_02 \\
           --roi-id 0007 --image-number 0 --mode both

2. **TIFF-stack mode** (``process_folder.py`` layout — the actual Dart
   experiment) — specify ``--tif-path`` and ``--frame``. Chamber type
   must come from either ``--chamber-type`` or ``--folder-config``:
       python scripts/generate_dart_frame_video.py \\
           --tif-path "dart_experiment/DART_Experiment/Big Chambers/Kammer 1 Versuch 1.tif" \\
           --frame 0 \\
           --folder-config dart_experiment/folder_config.json \\
           --flip --mode both

Either way, the renderer is the unchanged ``generate_pipeline_video``
from ``generate_sak_videos.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_sak_videos import generate_pipeline_video, load_image
from process_experiment import load_experiment_metadata

from dart_mlci import (
    DEFAULT_MODEL_PATH,
    ChipStructureLibrary,
    MarkerDetectionModel,
    create_structure_library,
    load_tif_frame,
    resolve_chamber_type_from_folder_config,
    select_timelapse_frame,
)
from dart_mlci.constants import ARTIFACTS_DIR

SCRIPTS_DIR = Path(__file__).parent
DEFAULT_CHIP_CONFIG = ARTIFACTS_DIR / "chips" / "sak.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the pipeline video/stills for one Dart experiment frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input mode: metadata OR tif-path (mutually exclusive) ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Metadata-mode input: dataset dir containing raw_images/meta.{parquet,csv}",
    )
    mode_group.add_argument(
        "--tif-path",
        type=Path,
        default=None,
        help="TIFF-stack-mode input: path to a (T,H,W) .tif file",
    )

    # Metadata-mode selectors
    parser.add_argument(
        "--roi-id", type=str, default=None, help="[metadata mode] ROI id, e.g. '0007' or '7'"
    )
    parser.add_argument(
        "--image-number",
        type=int,
        default=None,
        help="[metadata mode] 0-based timepoint index within the ROI",
    )

    # TIFF-mode selectors
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="[tif mode] 0-based frame index within the TIFF stack",
    )
    parser.add_argument(
        "--chamber-type",
        type=str,
        default=None,
        help="[tif mode] Chamber type name (e.g. 'BigBox-inner'). "
        "Takes precedence over --folder-config.",
    )
    parser.add_argument(
        "--folder-config",
        type=Path,
        default=None,
        help="[tif mode] Path to folder_config.json; chamber type "
        "is looked up from the TIFF's parent folder name.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="[tif mode] Vertically flip the frame before processing "
        "(matches the 'flip' option in dart folder configs).",
    )

    # Pipeline config (shared). Defaults are None so that we can prefer values
    # read from --folder-config when it is provided (matches process_folder.py).
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometers (default: read from folder-config, else 0.065789)",
    )
    parser.add_argument(
        "--chip-config",
        type=Path,
        default=None,
        help=f"[tif mode] Path to chip config JSON (default: {DEFAULT_CHIP_CONFIG})",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="[metadata mode] Path to chamber structure JSON",
    )

    # Output config
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPTS_DIR / "output" / "dart_videos",
        help="Directory for generated videos",
    )
    parser.add_argument(
        "--stills-dir",
        type=Path,
        default=SCRIPTS_DIR / "output" / "dart_stills",
        help="Root directory for stills; one subdir per frame",
    )
    parser.add_argument(
        "--mode",
        choices=["video", "stills", "both"],
        default="both",
        help="What to produce (default: both — stills needed for paper)",
    )
    parser.add_argument("--no-titles", action="store_true", help="Omit the step-title text overlay")
    parser.add_argument(
        "--no-progress-bar", action="store_true", help="Omit the bottom progress-bar overlay"
    )
    parser.add_argument(
        "--crop-black-borders",
        action="store_true",
        help="Crop all-black letterbox borders from saved stills "
        "(auto-enabled when --no-titles and --no-progress-bar are both set)",
    )
    parser.add_argument(
        "--scalebar-um",
        type=float,
        default=10.0,
        help="Scale bar length in micrometers, drawn bottom-right of every still "
        "(default: 10; pass 0 to disable)",
    )

    args = parser.parse_args()

    # Cross-argument validation
    if args.dataset_dir is not None:
        missing = [
            n
            for n, v in [("--roi-id", args.roi_id), ("--image-number", args.image_number)]
            if v is None
        ]
        if missing:
            parser.error(f"--dataset-dir requires: {', '.join(missing)}")
    else:  # tif-path mode
        if args.frame is None:
            parser.error("--tif-path requires --frame")
        if args.chamber_type is None and args.folder_config is None:
            parser.error("--tif-path requires either --chamber-type or --folder-config")

    # Apply folder-config defaults for pixel_size / flip / chip_config.
    # Explicit CLI values take precedence; folder-config fills in the rest.
    folder_cfg = None
    if args.folder_config is not None:
        with open(args.folder_config) as f:
            folder_cfg = json.load(f)
        cfg_base = ARTIFACTS_DIR.parent  # repo root — matches process_folder.py

        if args.pixel_size is None and "pixel_size" in folder_cfg:
            args.pixel_size = float(folder_cfg["pixel_size"])
            print(f"Using pixel_size={args.pixel_size} from folder-config")
        if not args.flip and folder_cfg.get("flip", False):
            args.flip = True
            print("Enabling --flip from folder-config")
        if args.chip_config is None and "chip_config" in folder_cfg:
            cand = (cfg_base / folder_cfg["chip_config"]).resolve()
            if cand.exists():
                args.chip_config = cand
                print(f"Using chip_config={args.chip_config} from folder-config")

    if args.pixel_size is None:
        args.pixel_size = 0.065789

    if args.no_titles and args.no_progress_bar:
        args.crop_black_borders = True

    return args


def _load_metadata_mode(args):
    """Return (image, chamber_name, roi_polygon, marker_group_pixels, slug)."""
    raw_images_dir = args.dataset_dir / "raw_images"
    if not raw_images_dir.exists():
        raw_images_dir = args.dataset_dir
    print(f"Loading metadata from {raw_images_dir}...")
    df, _ = load_experiment_metadata(raw_images_dir)

    image_path = select_timelapse_frame(
        df,
        roi_id=args.roi_id,
        image_number=args.image_number,
        image_base_dir=raw_images_dir,
    )
    print(f"Selected frame: {image_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = load_image(image_path)

    print("Loading chamber structure library...")
    structure_library = create_structure_library(
        structure_library_path=args.structure_library,
        pixel_size=args.pixel_size,
    )
    chamber_name, roi_polygon, marker_group_pixels = structure_library(args.roi_id)
    slug = f"roi_{str(args.roi_id).zfill(4)}_t{args.image_number:04d}"
    return image, chamber_name, roi_polygon, marker_group_pixels, slug


def _load_tif_mode(args):
    """Return (image, chamber_name, roi_polygon, marker_group_pixels, slug)."""
    tif_path = args.tif_path.resolve()
    if not tif_path.exists():
        raise FileNotFoundError(f"TIFF not found: {tif_path}")

    if args.chamber_type is not None:
        chamber_name = args.chamber_type
    else:
        chamber_name = resolve_chamber_type_from_folder_config(tif_path, args.folder_config)
    print(f"Chamber type: {chamber_name}")

    print(f"Loading frame {args.frame} from {tif_path}...")
    image = load_tif_frame(tif_path, args.frame, flip=args.flip)

    chip_config_path = args.chip_config or DEFAULT_CHIP_CONFIG
    print(f"Loading chip structure library from {chip_config_path}...")
    chip_lib = ChipStructureLibrary.from_file(chip_config_path, pixel_size=args.pixel_size)
    if chamber_name not in chip_lib.polygon_library:
        raise ValueError(
            f"Unknown chamber type {chamber_name!r}. "
            f"Known types: {sorted(chip_lib.polygon_library)}"
        )
    roi_polygon = chip_lib.polygon_library[chamber_name]
    marker_group_pixels = chip_lib.marker_group_configs[chamber_name]

    slug = f"{tif_path.stem.replace(' ', '_')}_f{args.frame:04d}"
    return image, chamber_name, roi_polygon, marker_group_pixels, slug


def main() -> int:
    args = _parse_args()

    produce_video = args.mode in ("video", "both")
    produce_stills = args.mode in ("stills", "both")

    if args.dataset_dir is not None:
        image, chamber_name, roi_polygon, marker_group_pixels, slug = _load_metadata_mode(args)
    else:
        image, chamber_name, roi_polygon, marker_group_pixels, slug = _load_tif_mode(args)

    print("Loading marker detection model...")
    model = MarkerDetectionModel(DEFAULT_MODEL_PATH)

    video_path = None
    if produce_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.output_dir / f"{slug}.mp4"

    stills_dir = None
    if produce_stills:
        stills_dir = args.stills_dir / slug
        stills_dir.mkdir(parents=True, exist_ok=True)

    ok = generate_pipeline_video(
        image,
        chamber_name,
        roi_polygon,
        marker_group_pixels,
        model,
        video_path,
        stills_dir,
        args,
    )
    if not ok:
        print("Pipeline rendering failed (see warnings above).")
        return 1

    if video_path is not None:
        print(f"Saved video: {video_path}")
    if stills_dir is not None:
        print(f"Saved stills: {stills_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

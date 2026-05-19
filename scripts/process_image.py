#!/usr/bin/env python
"""Process a single image through the DMC masking pipeline.

Takes an image and chamber_id, performs chamber_type lookup and all pipeline
steps (marker detection, matching, rotation, masking), then saves the cropped
and masked image.

Example usage:
    python scripts/process_image.py --image /path/to/image.tif --chamber-id 0050
    python scripts/process_image.py --image /path/to/image.tif --chamber-id 0050 --output cropped.tif
    python scripts/process_image.py --image /path/to/image.tif --chamber-type NormaleBox-inner
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dart_mlci import (
    DEFAULT_MODEL_PATH,
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
    create_structure_library,
)
from dart_mlci.io import load_image, save_image
from dart_mlci.types import PipelineError, PipelineTimings

# Pipeline step names for error reporting
STEP_VALIDATION = "VALIDATION"
STEP_LOADING = "LOADING"
STEP_DETECTION = "DETECTION"
STEP_MATCHING = "MATCHING"
STEP_ROTATION = "ROTATION"
STEP_MASKING = "MASKING"
STEP_SAVING = "SAVING"


def print_error(step: str, message: str) -> None:
    """Print a structured error message to stderr."""
    print(f"ERROR: {step}: {message}", file=sys.stderr)


def _format_timings(timings: PipelineTimings, load_time: float, save_time: float) -> str:
    process = timings.total
    total = load_time + process + save_time
    return (
        f"total={total:.3f}s, "
        f"load={load_time:.3f}s, "
        f"process={process:.3f}s "
        f"(detect={timings.detection:.3f}s, match={timings.matching:.3f}s, "
        f"rotate={timings.rotation:.3f}s, mask={timings.masking:.3f}s), "
        f"save={save_time:.3f}s"
    )


def print_success(
    timings: PipelineTimings,
    load_time: float,
    save_time: float,
    output_path: Path,
    mask_path: Path | None,
) -> None:
    """Print a structured success message to stdout."""
    msg = f"SUCCESS: {_format_timings(timings, load_time, save_time)} | output={output_path}"
    if mask_path:
        msg += f" | mask={mask_path}"
    print(msg)


def process_image(
    image: np.ndarray,
    roi_polygon,
    marker_group: dict,
    model_path: Path,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, PipelineTimings]:
    """Run the full masking pipeline on an image.

    Args:
        image: Input image as HxWx3 numpy array
        roi_polygon: RoIPolygon object for masking
        marker_group: Marker group configuration (pixel coordinates)
        model_path: Path to YOLO model
        device: Device to run on (e.g., 'cuda:0', 'cpu'). None for auto.
        verbose: If True, show YOLO inference output

    Returns:
        Tuple of (cropped_image, mask, timings) where:
            - cropped_image: CxHxW numpy array
            - mask: HxW binary numpy array
            - timings: PipelineTimings with step durations

    Raises:
        PipelineError: If any pipeline step fails
    """
    timings = PipelineTimings()

    # Initialize pipeline steps
    detection_step = MarkerDetectionStep(model_path, device=device, verbose=verbose)
    matching_step = MarkerMatchingStep(marker_group, tolerance=60)
    rotation_step = ImageRotationStep()
    masking_step = RoIMaskingStep(marker_group, roi_polygon)

    # Step 1: Detection
    try:
        start = time.perf_counter()
        data = detection_step(image)
        timings.detection = time.perf_counter() - start

        markers = data.get("markers", [])
        if not markers:
            raise PipelineError(STEP_DETECTION, "No markers detected in image")
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(STEP_DETECTION, str(e)) from e

    # Step 2: Matching
    try:
        start = time.perf_counter()
        data = matching_step(data)
        timings.matching = time.perf_counter() - start

        matched_indices = data.get("matched_marker_indices", [])
        if not matched_indices:
            raise PipelineError(STEP_MATCHING, "No valid marker pairs found")
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(STEP_MATCHING, str(e)) from e

    # Step 3: Rotation
    try:
        start = time.perf_counter()
        data = rotation_step(data)
        timings.rotation = time.perf_counter() - start
    except Exception as e:
        raise PipelineError(STEP_ROTATION, str(e)) from e

    # Step 4: Masking
    try:
        start = time.perf_counter()
        data = masking_step(data)
        timings.masking = time.perf_counter() - start
    except Exception as e:
        raise PipelineError(STEP_MASKING, str(e)) from e

    return data["image"], data["mask"], timings


def main():
    parser = argparse.ArgumentParser(
        description="Process a single image through the DMC masking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using chamber_id (performs lookup to determine chamber_type)
  python scripts/process_image.py --image image.tif --chamber-id 0050

  # Using chamber_type directly
  python scripts/process_image.py --image image.tif --chamber-type NormaleBox-inner

  # Specify output path
  python scripts/process_image.py --image image.tif --chamber-id 0050 --output cropped.tif

Chamber ID patterns:
  0000-0099: NormaleBox-inner
  0100-0199: BigBox-inner
  0200-0299: OpenBox-inner
  0300-0399: Mothermachine-inner
  1000-1099: NormaleBox-pillar-inner
  1100-1199: BigBox-pillar-inner
  1200-1299: OpenBox-collector-inner
  1300-1399: Mothermachine-2x-inner
        """,
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image file",
    )

    # Chamber specification (at least one required, chamber-type takes precedence)
    parser.add_argument(
        "--chamber-id",
        type=str,
        help="Chamber ID (e.g., '0050', '1200'). Used to lookup chamber_type.",
    )
    parser.add_argument(
        "--chamber-type",
        type=str,
        help="Chamber type directly (e.g., 'NormaleBox-inner', 'OpenBox-inner'). "
        "Takes precedence over --chamber-id if both are provided.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for cropped image (default: <input>_cropped.<ext>)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not save the mask separately",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to YOLO model (default: artifacts/models/v26_detect_s_imgsz1280.pt)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.065789,
        help="Pixel size in micrometers (default: 0.065789)",
    )
    parser.add_argument(
        "--chip-config",
        type=Path,
        default=None,
        help="Path to unified chip config JSON file (preferred over --structure-library)",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="Path to chamber structure JSON file (deprecated, use --chip-config instead)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cpu'). Default: auto",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show YOLO inference output",
    )

    args = parser.parse_args()

    load_time = 0.0
    save_time = 0.0
    timings = PipelineTimings()

    # Validate input file
    if not args.image.exists():
        print_error(STEP_VALIDATION, f"Input image not found: {args.image}")
        sys.exit(1)

    # Set default model path
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH

    if not args.model_path.exists():
        print_error(STEP_VALIDATION, f"Model not found: {args.model_path}")
        sys.exit(1)

    # Validate that at least one chamber specification is provided
    if not args.chamber_id and not args.chamber_type:
        print_error(STEP_VALIDATION, "Either --chamber-id or --chamber-type must be provided")
        sys.exit(1)

    # Initialize structure library: prefer --chip-config, fall back to --structure-library
    try:
        if args.chip_config is not None:
            if not args.chip_config.exists():
                print_error(STEP_VALIDATION, f"Chip config not found: {args.chip_config}")
                sys.exit(1)
            structure_library = create_structure_library(
                chip_config_path=args.chip_config,
                pixel_size=args.pixel_size,
            )
        else:
            structure_library = create_structure_library(
                structure_library_path=args.structure_library,
                pixel_size=args.pixel_size,
            )
    except Exception as e:
        print_error(STEP_VALIDATION, f"Failed to load structure library: {e}")
        sys.exit(1)

    # Determine chamber configuration (chamber-type takes precedence)
    if args.chamber_type:
        if args.chamber_id:
            print(
                f"Warning: Both --chamber-id and --chamber-type provided. "
                f"Using --chamber-type '{args.chamber_type}'.",
                file=sys.stderr,
            )
        structure_name = args.chamber_type
        if structure_name not in structure_library.polygon_library:
            available = list(structure_library.polygon_library.keys())
            print_error(
                STEP_VALIDATION,
                f"Unknown chamber type: {structure_name}. Available: {available}",
            )
            sys.exit(1)
        roi_polygon = structure_library.polygon_library[structure_name]
        marker_group = structure_library.marker_group_configs[structure_name]
    else:
        try:
            structure_name, roi_polygon, marker_group = structure_library(args.chamber_id)
        except ValueError as e:
            print_error(STEP_VALIDATION, f"Invalid chamber ID '{args.chamber_id}': {e}")
            sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = args.image.parent / f"{args.image.stem}_cropped{args.image.suffix}"

    # Load image
    try:
        start = time.perf_counter()
        image = load_image(args.image)
        load_time = time.perf_counter() - start
    except Exception as e:
        print_error(STEP_LOADING, f"Failed to load image '{args.image}': {e}")
        sys.exit(1)

    # Process image
    try:
        cropped_image, mask, timings = process_image(
            image=image,
            roi_polygon=roi_polygon,
            marker_group=marker_group,
            model_path=args.model_path,
            device=args.device,
            verbose=args.verbose,
        )
    except PipelineError as e:
        print_error(e.step, e.message)
        sys.exit(1)
    except Exception as e:
        print_error(STEP_MASKING, f"Unexpected error: {e}")
        sys.exit(1)

    # Save output
    try:
        start = time.perf_counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mask_path = save_image(
            cropped_image,
            args.output,
            mask=None if args.no_mask else mask,
        )
        save_time = time.perf_counter() - start
    except Exception as e:
        print_error(STEP_SAVING, f"Failed to save output: {e}")
        sys.exit(1)

    # Print success message with timings
    print_success(timings, load_time, save_time, args.output, mask_path)


if __name__ == "__main__":
    main()

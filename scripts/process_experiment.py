#!/usr/bin/env python
"""Process experiment dataset with full DMC masking and cell segmentation pipeline.

Applies the DMC masking pipeline (marker detection, matching, rotation, ROI masking)
followed by cell segmentation to all images in an experiment dataset.

Example usage:
    python scripts/process_experiment.py --dataset-dir /path/to/experiment
    python scripts/process_experiment.py --dataset-dir /path/to/experiment --max-images 5 --verbose
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from acia.segm.local import THWCSequenceSource
    from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter

    ACIA_AVAILABLE = True
except ImportError:
    ACIA_AVAILABLE = False

import dmc_masking
from dmc_masking import (
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
)
from dmc_masking.io import load_image
from dmc_masking.mask import SAKRoIStructureLibrary

# Pipeline step names for tracking
STEP_LOADING = "Loading"
STEP_STRUCTURE = "Structure"
STEP_DETECTION = "Detection"
STEP_MATCHING = "Matching"
STEP_ROTATION = "Rotation"
STEP_MASKING = "Masking"
STEP_SEGMENTATION = "Segmentation"
STEP_SAVING = "Saving"

ALL_STEPS = [
    STEP_LOADING,
    STEP_STRUCTURE,
    STEP_DETECTION,
    STEP_MATCHING,
    STEP_ROTATION,
    STEP_MASKING,
    STEP_SEGMENTATION,
    STEP_SAVING,
]


@dataclass
class ImageResult:
    """Stores result for a single image processing attempt."""

    image_file: str
    roi_id: str
    success: bool = False
    failed_step: str | None = None
    error_message: str | None = None
    n_cells: int = 0
    structure_name: str | None = None
    output_path: str | None = None


class ExperimentProcessor:
    """Processes experiment images through the full DMC masking + segmentation pipeline."""

    def __init__(
        self,
        model_path: Path,
        structure_library_path: Path,
        pixel_size: float = 0.065789,
        device: str | None = None,
        verbose: bool = False,
        save_cropped: bool = False,
    ):
        """Initialize the experiment processor.

        Args:
            model_path: Path to the YOLO model weights
            structure_library_path: Path to chamber structure JSON file
            pixel_size: Pixel size in micrometers
            device: Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). None for auto.
            verbose: If True, show detailed progress
            save_cropped: If True, save cropped images alongside masks
        """
        self.pixel_size = pixel_size
        self.device = device
        self.verbose = verbose
        self.save_cropped = save_cropped

        # Initialize SAK structure library (provides polygon and marker configs)
        self.structure_library = SAKRoIStructureLibrary(
            lookup_path=structure_library_path,
            pixel_size=pixel_size,
        )

        # Initialize detection step (shared across all images)
        self.detection_step = MarkerDetectionStep(model_path, device=device, verbose=verbose)

        # Cache for chamber-specific pipeline components (keyed by structure_name)
        self._chamber_cache: dict[str, dict] = {}

        # Initialize segmenter
        self.segmenter = None
        if not ACIA_AVAILABLE:
            raise ImportError("acia library not available. Install it with: pip install acia")
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        self.segmenter = CellposeSAMSegmenter()

    def _get_chamber_components(self, structure_name: str) -> dict:
        """Get or create pipeline components for a structure type.

        Args:
            structure_name: Chamber structure name (e.g., "NormaleBox-inner")

        Returns:
            Component dict with pipeline steps
        """
        if structure_name not in self._chamber_cache:
            roi_polygon = self.structure_library.polygon_library[structure_name]
            marker_group = self.structure_library.marker_group_configs[structure_name]
            self._chamber_cache[structure_name] = {
                "roi_polygon": roi_polygon,
                "marker_group": marker_group,
                "matching_step": MarkerMatchingStep(marker_group, tolerance=60),
                "rotation_step": ImageRotationStep(),
                "masking_step": RoIMaskingStep(marker_group, roi_polygon),
            }
        return self._chamber_cache[structure_name]

    def process_image(
        self,
        image_path: Path,
        roi_id: str,
        output_path: Path,
        cropped_output_path: Path | None = None,
    ) -> ImageResult:
        """Process a single image through the pipeline.

        Args:
            image_path: Path to the image file
            roi_id: ROI ID for determining chamber structure
            output_path: Path to save the segmentation mask
            cropped_output_path: Optional path to save cropped image

        Returns:
            ImageResult with success status and cell count
        """
        result = ImageResult(image_file=str(image_path.name), roi_id=roi_id)

        # Step 1: Loading
        try:
            image = load_image(image_path)
            if image is None or image.size == 0:
                raise ValueError("Image is empty or failed to load")
        except Exception as e:
            result.failed_step = STEP_LOADING
            result.error_message = str(e)
            return result

        # Step 2: Get structure from roi_id
        try:
            structure_name, roi_polygon, marker_group = self.structure_library(roi_id)
            result.structure_name = structure_name
            components = self._get_chamber_components(structure_name)
        except Exception as e:
            result.failed_step = STEP_STRUCTURE
            result.error_message = str(e)
            return result

        # Step 3: Detection
        try:
            detection_result = self.detection_step(image)
        except Exception as e:
            result.failed_step = STEP_DETECTION
            result.error_message = str(e)
            return result

        # Step 4: Matching
        try:
            matching_result = components["matching_step"](detection_result)
            if not matching_result.get("matched_marker_indices", []):
                raise ValueError("No valid marker pairs found")
        except Exception as e:
            result.failed_step = STEP_MATCHING
            result.error_message = str(e)
            return result

        # Step 5: Rotation
        try:
            rotation_result = components["rotation_step"](matching_result)
        except Exception as e:
            result.failed_step = STEP_ROTATION
            result.error_message = str(e)
            return result

        # Step 6: Masking
        try:
            masking_result = components["masking_step"](rotation_result)
            cropped_image = masking_result["image"]
            # Convert CxHxW to HxWxC if needed
            if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                cropped_image = np.moveaxis(cropped_image, 0, -1)
        except Exception as e:
            result.failed_step = STEP_MASKING
            result.error_message = str(e)
            return result

        # Step 7: Segmentation
        try:
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            height, width = cropped_rgb.shape[:2]
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)  # TxHxWxC
            source = THWCSequenceSource(segm_input)

            with torch.no_grad():
                segmentation_result = self.segmenter(source.to_channel(0))

            # Extract instance-labeled mask (binary_mask=False gives uint16 labels)
            masks = segmentation_result.toMasks(height, width, binary_mask=False)
            labeled_mask = masks[0]  # First (only) frame
            result.n_cells = int(labeled_mask.max())  # Count unique cell IDs
        except Exception as e:
            result.failed_step = STEP_SEGMENTATION
            result.error_message = str(e)
            return result

        # Step 8: Saving
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                output_path,
                labeled_mask.astype(np.uint16),
                compression="zlib",
            )
            result.output_path = str(output_path)

            # Optionally save cropped image
            if cropped_output_path is not None:
                cropped_output_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    cropped_output_path,
                    cropped_image.astype(np.uint8),
                    compression="zlib",
                )
        except Exception as e:
            result.failed_step = STEP_SAVING
            result.error_message = str(e)
            return result

        # All steps passed
        result.success = True
        return result


def load_experiment_metadata(raw_images_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load experiment metadata from parquet or csv file.

    Args:
        raw_images_dir: Path to raw_images directory containing meta.parquet or meta.csv

    Returns:
        Tuple of (DataFrame with experiment metadata, path to metadata file)
    """
    # Try parquet first, then csv
    meta_parquet = raw_images_dir / "meta.parquet"
    meta_csv = raw_images_dir / "meta.csv"

    if meta_parquet.exists():
        df = pd.read_parquet(meta_parquet)
        meta_path = meta_parquet
    elif meta_csv.exists():
        df = pd.read_csv(meta_csv)
        meta_path = meta_csv
    else:
        raise FileNotFoundError(
            f"Metadata file not found. Expected meta.parquet or meta.csv in {raw_images_dir}"
        )

    # Validate required columns
    required_cols = ["image_file", "roi_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {meta_path.name}: {missing}")

    return df, meta_path


def generate_summary(results: list[ImageResult]) -> str:
    """Generate text summary of processing results.

    Args:
        results: List of ImageResult objects

    Returns:
        Formatted summary string
    """
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    total_cells = sum(r.n_cells for r in results if r.success)

    # Count failures by step
    step_counts = {step: 0 for step in ALL_STEPS}
    for r in results:
        if not r.success and r.failed_step:
            step_counts[r.failed_step] += 1

    # Build summary
    lines = [
        "=" * 80,
        "EXPERIMENT PROCESSING SUMMARY",
        "=" * 80,
        f"Total Images:     {total}",
        f"Passed:           {passed} ({100*passed/total:.1f}%)"
        if total > 0
        else "Passed:           0",
        f"Failed:           {failed} ({100*failed/total:.1f}%)"
        if total > 0
        else "Failed:           0",
        f"Total Cells:      {total_cells}",
        "",
    ]

    # Add failure breakdown if there are failures
    if failed > 0:
        lines.append("Failed by Step:")
        for step in ALL_STEPS:
            if step_counts[step] > 0:
                lines.append(f"  {step:15} {step_counts[step]} error(s)")
        lines.append("")

        lines.append("Failed Images:")
        for r in results:
            if not r.success:
                lines.append(
                    f"  {r.image_file} (roi={r.roi_id}): [{r.failed_step}] {r.error_message}"
                )
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def export_results_csv(results: list[ImageResult], output_path: Path) -> None:
    """Export detailed results to CSV file.

    Args:
        results: List of ImageResult objects
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(
        [
            {
                "image_file": r.image_file,
                "roi_id": r.roi_id,
                "structure_name": r.structure_name or "",
                "success": r.success,
                "n_cells": r.n_cells,
                "failed_step": r.failed_step or "",
                "error_message": r.error_message or "",
                "output_path": r.output_path or "",
            }
            for r in results
        ]
    )
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Process experiment dataset with DMC masking and cell segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_experiment.py --dataset-dir /path/to/experiment
  python scripts/process_experiment.py --dataset-dir /path/to/experiment --max-images 5 --verbose

Dataset structure:
  <dataset-dir>/
      raw_images/           # Subfolder containing metadata and image files
          meta.parquet      # (or meta.csv) Metadata with columns: image_file, roi_id, ...
          image1.tif
          image2.tif
          ...

Output structure:
  <output-dir>/
      image1.tif            # Segmentation mask (uint16, 0=background, 1..N=cell IDs)
      image2.tif
      ...
      results.csv           # Per-image results (success, n_cells, errors)
      summary.txt           # Overall statistics
        """,
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset directory containing raw_images/ subfolder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for segmentation masks (default: <dataset-dir>/segmentation_masks)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.065789,
        help="Pixel size in micrometers (default: 0.065789)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to YOLO model (default: artifacts/models/v8_detect_s_imgsz640.pt)",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="Path to chamber structure JSON file (default: artifacts/chamber_structure.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda:0', 'cuda:1', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--save-cropped",
        action="store_true",
        help="Also save cropped images alongside masks",
    )

    args = parser.parse_args()

    # Validate dataset directory
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    # Set default paths
    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "segmentation_masks"

    if args.model_path is None:
        args.model_path = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/v8_detect_s_imgsz640.pt"
        )

    if args.structure_library is None:
        args.structure_library = (
            Path(dmc_masking.__file__).parent.parent / "artifacts/chamber_structure.json"
        )

    # Validate paths
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    if not args.structure_library.exists():
        raise FileNotFoundError(f"Structure library not found: {args.structure_library}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create cropped output directory if needed
    cropped_dir = None
    if args.save_cropped:
        cropped_dir = args.output_dir / "cropped_images"
        cropped_dir.mkdir(parents=True, exist_ok=True)

    # Determine raw_images directory
    raw_images_dir = args.dataset_dir / "raw_images"
    if not raw_images_dir.exists():
        # Maybe dataset_dir itself is the raw_images folder
        raw_images_dir = args.dataset_dir

    # Load metadata (from raw_images directory)
    print(f"Loading metadata from {raw_images_dir}...")
    df, meta_path = load_experiment_metadata(raw_images_dir)
    print(f"Found {len(df)} images in dataset (from {meta_path.name})")

    # Limit images if requested
    if args.max_images is not None:
        df = df.head(args.max_images)
        print(f"Processing first {len(df)} images (--max-images)")

    # Images are in the same directory as the metadata file
    image_base_dir = raw_images_dir

    # Initialize processor
    print("\nInitializing processor...")
    print(f"  Model: {args.model_path}")
    print(f"  Structure library: {args.structure_library}")
    print(f"  Pixel size: {args.pixel_size}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Output: {args.output_dir}")

    processor = ExperimentProcessor(
        model_path=args.model_path,
        structure_library_path=args.structure_library,
        pixel_size=args.pixel_size,
        device=args.device,
        verbose=args.verbose,
        save_cropped=args.save_cropped,
    )

    # Process images
    print(f"\nProcessing {len(df)} images...")
    results = []

    for i, row in enumerate(df.itertuples(), 1):
        image_file = row.image_file
        roi_id = str(row.roi_id)

        # Construct paths
        image_path = image_base_dir / image_file
        output_name = Path(image_file).stem + ".tif"
        output_path = args.output_dir / output_name

        cropped_output_path = None
        if cropped_dir is not None:
            cropped_output_path = cropped_dir / output_name

        # Progress output
        if args.verbose:
            print(f"[{i}/{len(df)}] Processing {image_file} (roi_id={roi_id})...")
        else:
            print(f"[{i}/{len(df)}] {image_file}", end=" ")

        # Process image
        result = processor.process_image(
            image_path=image_path,
            roi_id=roi_id,
            output_path=output_path,
            cropped_output_path=cropped_output_path,
        )
        results.append(result)

        # Status output
        if result.success:
            if args.verbose:
                print(f"  -> PASSED ({result.n_cells} cells)")
            else:
                print(f"-> OK ({result.n_cells} cells)")
        else:
            if args.verbose:
                print(f"  -> FAILED at {result.failed_step}: {result.error_message}")
            else:
                print(f"-> FAILED [{result.failed_step}]")

    # Generate and print summary
    summary = generate_summary(results)
    print(f"\n{summary}")

    # Save summary to file
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")

    # Export detailed results CSV
    results_csv_path = args.output_dir / "results.csv"
    export_results_csv(results, results_csv_path)
    print(f"Detailed results saved to: {results_csv_path}")


if __name__ == "__main__":
    main()

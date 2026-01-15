#!/usr/bin/env python
"""Batch masking script for DMC masking pipeline.

Processes a list of images with their chamber types, runs the masking pipeline
on each, and provides detailed pass/fail reporting with visual debug outputs.

Example usage:
    python scripts/batch_masking.py --input images.csv --output-dir ./output
    python scripts/batch_masking.py --input images.csv --output-dir ./output --skip-segmentation
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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
from dmc_masking.mask import SAKRoIStructureLibrary
from dmc_masking.visualization import plot_markers_on_image

# Import load_image from benchmark script
from scripts.benchmark import load_image

# Pipeline step names for tracking
STEP_LOADING = "Loading"
STEP_DETECTION = "Detection"
STEP_MATCHING = "Matching"
STEP_ROTATION = "Rotation"
STEP_MASKING = "Masking"
STEP_SEGMENTATION = "Segmentation"

ALL_STEPS = [
    STEP_LOADING,
    STEP_DETECTION,
    STEP_MATCHING,
    STEP_ROTATION,
    STEP_MASKING,
    STEP_SEGMENTATION,
]


@dataclass
class ImageResult:
    """Stores result for a single image processing attempt."""

    image_path: str
    chamber_type: str
    success: bool = False
    failed_step: str | None = None
    error_message: str | None = None
    # Partial results for visualization
    image: np.ndarray | None = field(default=None, repr=False)
    markers: list | None = field(default=None, repr=False)
    matched_indices: list | None = field(default=None, repr=False)
    angle: float | None = None


class BatchMaskingRunner:
    """Runs masking pipeline on multiple images with different chamber types."""

    def __init__(
        self,
        model_path: Path,
        pixel_size: float = 0.065789,
        structure_library_path: Path | None = None,
        skip_segmentation: bool = False,
        device: str | None = None,
    ):
        """Initialize the batch runner.

        Args:
            model_path: Path to the YOLO model weights
            pixel_size: Pixel size in micrometers
            structure_library_path: Path to chamber structure JSON file
            skip_segmentation: If True, skip the cell segmentation step
            device: Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). None for auto.
        """
        self.model_path = model_path
        self.pixel_size = pixel_size
        self.skip_segmentation = skip_segmentation
        self.device = device

        # Default structure library path
        if structure_library_path is None:
            structure_library_path = (
                Path(dmc_masking.__file__).parent.parent / "artifacts/chamber_structure.json"
            )

        # Initialize SAK structure library (provides polygon and marker configs)
        self.structure_library = SAKRoIStructureLibrary(
            lookup_path=structure_library_path,
            pixel_size=pixel_size,
        )

        # Initialize detection step (shared across all images)
        self.detection_step = MarkerDetectionStep(model_path, device=device)

        # Cache for chamber-specific pipeline components (keyed by structure_name)
        self._chamber_cache: dict[str, dict] = {}

        # Initialize segmenter if enabled
        self.segmenter = None
        if not skip_segmentation:
            if not ACIA_AVAILABLE:
                print("Warning: acia library not available, skipping segmentation")
            else:
                import warnings

                warnings.filterwarnings("ignore", category=FutureWarning)
                self.segmenter = CellposeSAMSegmenter()

    def _get_chamber_components(self, chamber_type: str) -> tuple[str, dict]:
        """Get or create pipeline components for a chamber type.

        Args:
            chamber_type: Chamber structure name (e.g., "NormaleBox-inner", "BigBox-inner")

        Returns:
            Tuple of (structure_name, component dict with pipeline steps)
        """
        # Use chamber_type directly as the structure name
        structure_name = chamber_type
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
        return structure_name, self._chamber_cache[structure_name]

    def process_image(self, image_path: str, chamber_type: str) -> ImageResult:
        """Process a single image through the pipeline with step-by-step error handling.

        Args:
            image_path: Path to the image file
            chamber_type: Chamber structure name (e.g., "NormaleBox-inner", "BigBox-inner")

        Returns:
            ImageResult with success status and partial results
        """
        result = ImageResult(image_path=image_path, chamber_type=chamber_type)
        path = Path(image_path)

        # Step 1: Loading
        try:
            image = load_image(path)
            if image is None or image.size == 0:
                raise ValueError("Image is empty or failed to load")
            result.image = image
        except Exception as e:
            result.failed_step = STEP_LOADING
            result.error_message = str(e)
            return result

        # Get chamber-specific components (validates chamber_type via SAKRoIStructureLibrary)
        try:
            _, components = self._get_chamber_components(chamber_type)
        except Exception as e:
            result.failed_step = STEP_LOADING
            result.error_message = f"Failed to load chamber configuration: {e}"
            return result

        # Step 2: Detection
        try:
            detection_result = self.detection_step(image)
            result.markers = detection_result.get("markers", [])
        except Exception as e:
            result.failed_step = STEP_DETECTION
            result.error_message = str(e)
            return result

        # Step 3: Matching
        try:
            matching_result = components["matching_step"](detection_result)
            result.matched_indices = matching_result.get("matched_marker_indices", [])
            if not result.matched_indices:
                raise ValueError("No valid marker pairs found")
        except Exception as e:
            result.failed_step = STEP_MATCHING
            result.error_message = str(e)
            return result

        # Step 4: Rotation
        try:
            rotation_result = components["rotation_step"](matching_result)
            result.angle = rotation_result.get("angle", 0.0)
        except Exception as e:
            result.failed_step = STEP_ROTATION
            result.error_message = str(e)
            return result

        # Step 5: Masking
        try:
            masking_result = components["masking_step"](rotation_result)
        except Exception as e:
            result.failed_step = STEP_MASKING
            result.error_message = str(e)
            return result

        # Step 6: Segmentation (optional)
        if self.segmenter is not None:
            try:
                cropped_image = masking_result["image"]
                # Convert CxHxW to HxWxC if needed
                if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                    cropped_image = np.moveaxis(cropped_image, 0, -1)
                cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
                source = THWCSequenceSource(segm_input)
                with torch.no_grad():
                    _ = self.segmenter(source.to_channel(0))
            except Exception as e:
                result.failed_step = STEP_SEGMENTATION
                result.error_message = str(e)
                return result

        # All steps passed
        result.success = True
        return result

    def run(self, image_list: list[tuple[str, str]]) -> list[ImageResult]:
        """Process all images and collect results.

        Args:
            image_list: List of (image_path, chamber_type) tuples

        Returns:
            List of ImageResult objects
        """
        results = []
        total = len(image_list)

        for i, (image_path, chamber_type) in enumerate(image_list, 1):
            print(f"Processing {i}/{total}: {Path(image_path).name} ({chamber_type})")
            result = self.process_image(image_path, chamber_type)

            if result.success:
                print("  -> PASSED")
            else:
                print(f"  -> FAILED at {result.failed_step}: {result.error_message}")

            results.append(result)

        return results


def save_debug_visualization(result: ImageResult, output_path: Path) -> None:
    """Save debug visualization for a failed image.

    Args:
        result: ImageResult with partial results
        output_path: Path to save the visualization
    """
    if result.image is None:
        # Loading failed, nothing to visualize
        return

    # Determine what to visualize based on failed step
    markers = result.markers or []
    matched_indices = result.matched_indices or []

    # Build title with failure info
    title = (
        f"{Path(result.image_path).name}\nFailed at: {result.failed_step}\n{result.error_message}"
    )

    # Use plot_markers_on_image for visualization
    plot_markers_on_image(
        image=result.image,
        markers=markers,
        matched_indices=matched_indices,
        title=title,
        output_path=output_path,
    )


def load_image_list(csv_path: Path) -> list[tuple[str, str]]:
    """Load image list from CSV file.

    Args:
        csv_path: Path to CSV file with columns: image_path, chamber_type

    Returns:
        List of (image_path, chamber_type) tuples
    """
    df = pd.read_csv(csv_path, dtype=str).dropna()
    return list(zip(df["image_path"].str.strip(), df["chamber_type"].str.strip(), strict=False))


def generate_summary(results: list[ImageResult]) -> str:
    """Generate text summary of batch results.

    Args:
        results: List of ImageResult objects

    Returns:
        Formatted summary string
    """
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed

    # Count failures by step
    step_counts = {step: 0 for step in ALL_STEPS}
    for r in results:
        if not r.success and r.failed_step:
            step_counts[r.failed_step] += 1

    # Build summary
    lines = [
        "=" * 80,
        "BATCH MASKING SUMMARY",
        "=" * 80,
        f"Total Images:     {total}",
        f"Passed:           {passed} ({100*passed/total:.1f}%)"
        if total > 0
        else "Passed:           0",
        f"Failed:           {failed} ({100*failed/total:.1f}%)"
        if total > 0
        else "Failed:           0",
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
                    f"  {r.image_path} ({r.chamber_type}): [{r.failed_step}] {r.error_message}"
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
                "image_path": r.image_path,
                "chamber_type": r.chamber_type,
                "success": r.success,
                "failed_step": r.failed_step or "",
                "error_message": r.error_message or "",
                "angle": r.angle if r.angle is not None else "",
            }
            for r in results
        ]
    )
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images through the DMC masking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/batch_masking.py --input images.csv --output-dir ./output
  python scripts/batch_masking.py --input images.csv --output-dir ./output --skip-segmentation

CSV format (chamber_type is a structure name string):
  image_path,chamber_type
  /path/to/image1.tif,NormaleBox-inner
  /path/to/image2.png,BigBox-inner
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file with columns: image_path, chamber_type (structure name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results and debug images",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to YOLO model (default: artifacts/models/best34.pt)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.065789,
        help="Pixel size in micrometers (default: 0.065789)",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="Path to chamber structure JSON file",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip the cell segmentation step",
    )
    parser.add_argument(
        "--save-passed",
        action="store_true",
        help="Also save visualizations for passed images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default: auto",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    # Set default model path
    if args.model_path is None:
        args.model_path = Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = args.output_dir / "failed"
    failed_dir.mkdir(exist_ok=True)
    if args.save_passed:
        passed_dir = args.output_dir / "passed"
        passed_dir.mkdir(exist_ok=True)

    # Load image list
    image_list = load_image_list(args.input)
    print(f"Loaded {len(image_list)} images from {args.input}\n")

    if not image_list:
        print("No images to process")
        return

    # Create runner and process images
    runner = BatchMaskingRunner(
        model_path=args.model_path,
        pixel_size=args.pixel_size,
        structure_library_path=args.structure_library,
        skip_segmentation=args.skip_segmentation,
        device=args.device,
    )

    results = runner.run(image_list)

    # Save debug visualizations for failed images
    print("\nSaving debug visualizations...")
    for r in results:
        if not r.success:
            if r.image is None:
                print(f"  Skipped (no image loaded): {r.image_path}")
            else:
                stem = Path(r.image_path).stem
                output_path = failed_dir / f"{stem}_{r.failed_step.lower()}.png"
                save_debug_visualization(r, output_path)
                print(f"  Saved: {output_path}")

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

#!/usr/bin/env python
"""Benchmark script for DMC masking pipeline.

Measures execution time and GPU memory usage for each pipeline step.
Supports warmup iterations, CSV export, and console summary with statistics.

Example usage:
    python scripts/benchmark.py --input images.csv --output results.csv
    python scripts/benchmark.py --input images.csv --warmup 5 --skip-segmentation

CSV format (chamber_type is a structure name string):
    image_path,chamber_type
    /path/to/image1.tif,NormaleBox-inner
    /path/to/image2.png,BigBox-inner
"""

import argparse
import time
import warnings
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
from dmc_masking.utils import normalize_image

# Available chamber types
CHAMBER_TYPES = [
    "NormaleBox-inner",
    "BigBox-inner",
    "OpenBox-inner",
    "Mothermachine-inner",
    "NormaleBox-pillar-inner",
    "BigBox-pillar-inner",
    "OpenBox-collector-inner",
    "Mothermachine-2x-inner",
]


class Timer:
    """Context manager for precise timing measurements."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB (returns 0 if CUDA unavailable)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory tracking statistics."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def load_image_list(csv_path: Path) -> list[tuple[str, str]]:
    """Load image list from CSV file.

    Args:
        csv_path: Path to CSV file with columns: image_path, chamber_type

    Returns:
        List of (image_path, chamber_type) tuples
    """
    df = pd.read_csv(csv_path, dtype=str).dropna()
    return list(zip(df["image_path"].str.strip(), df["chamber_type"].str.strip(), strict=False))


def load_image(image_path: Path) -> np.ndarray:
    """Load and prepare image for pipeline.

    Handles single images as well as TIFF stacks (TxCxHxW format).
    For stacks, extracts the first frame and first channel.

    Args:
        image_path: Path to the image file

    Returns:
        Image as HxWx3 numpy array in uint8 format
    """
    import tifffile

    suffix = image_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        # Use tifffile directly to handle multi-dimensional TIFFs
        image = tifffile.imread(str(image_path))

        # Handle multi-dimensional TIFF stacks (TxCxHxW or CxHxW)
        if image.ndim == 4:
            # TxCxHxW format - take first time point and first channel
            image = image[0, 0]
        elif image.ndim == 3:
            # Could be CxHxW, TxHxW, or HxWxC
            if image.shape[0] <= 4:
                # Likely CxHxW - take first channel
                image = image[0]
            elif image.shape[2] <= 4:
                # Likely HxWxC - keep as is
                pass
            else:
                # Likely TxHxW - take first time point
                image = image[0]

        # Normalize to uint8
        if image.dtype != np.uint8:
            image = normalize_image(image)
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

    # Normalize if uint16 (shouldn't happen after above, but safety check)
    if image.dtype == np.uint16:
        image = normalize_image(image)

    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Single channel HxWx1
        image = np.stack((image[:, :, 0],) * 3, axis=-1)

    return image


class PipelineBenchmark:
    """Benchmark runner for the DMC masking pipeline."""

    def __init__(
        self,
        model_path: Path,
        pixel_size: float = 0.065789,
        structure_library_path: Path | None = None,
        skip_segmentation: bool = False,
        device: str | None = None,
    ):
        """Initialize the benchmark with pipeline configuration.

        Args:
            model_path: Path to the YOLO model weights
            pixel_size: Pixel size in micrometers
            structure_library_path: Path to chamber structure JSON file
            skip_segmentation: If True, skip the cell segmentation step
            device: Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). None for auto.
        """
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

        # Initialize detection step (shared across all chamber types)
        self.detection_step = MarkerDetectionStep(model_path, device=device)

        # Cache for chamber-specific pipeline components (keyed by structure_name)
        self._chamber_cache: dict[str, dict] = {}

        # Initialize segmenter (if enabled and acia is available)
        self.segmenter = None
        if not skip_segmentation:
            if not ACIA_AVAILABLE:
                print("Warning: acia library not available, skipping segmentation")
            else:
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

    def warmup(self, image: np.ndarray, chamber_type: str, n_warmup: int = 3):
        """Run warmup iterations to stabilize GPU/JIT compilation.

        Args:
            image: Input image for warmup
            chamber_type: Chamber structure name for warmup
            n_warmup: Number of warmup iterations
        """
        print(f"Running {n_warmup} warmup iterations...")
        for i in range(n_warmup):
            self._run_pipeline(image, chamber_type)
            print(f"  Warmup {i + 1}/{n_warmup} complete")
        print("Warmup complete.\n")

    def _run_pipeline(self, image: np.ndarray, chamber_type: str):
        """Run full pipeline without timing.

        Args:
            image: Input image
            chamber_type: Chamber structure name for pipeline components

        Returns:
            Pipeline result dictionary
        """
        _, components = self._get_chamber_components(chamber_type)

        result = self.detection_step(image)
        result = components["matching_step"](result)
        result = components["rotation_step"](result)
        result = components["masking_step"](result)

        # Run segmentation if enabled
        if self.segmenter is not None:
            cropped_image = result["image"]
            # Convert CxHxW to HxWxC if needed
            if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                cropped_image = np.moveaxis(cropped_image, 0, -1)
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)
            with torch.no_grad():
                _ = self.segmenter(source.to_channel(0))

        return result

    def benchmark_image(self, image_path: str, chamber_type: str) -> dict:
        """Benchmark a single image through all pipeline steps.

        Args:
            image_path: Path to the image file
            chamber_type: Chamber structure name (e.g., "NormaleBox-inner", "BigBox-inner")

        Returns:
            Dictionary with timing and memory measurements
        """
        path = Path(image_path)

        # Get chamber-specific components (also returns structure_name)
        structure_name, components = self._get_chamber_components(chamber_type)

        timings = {
            "image_name": path.name,
            "chamber_type": chamber_type,
            "structure_name": structure_name,
        }

        # Load image (not timed)
        image = load_image(path)

        # Marker detection (with GPU memory tracking)
        reset_gpu_memory_stats()
        with Timer() as t:
            result = self.detection_step(image)
        timings["detection_time"] = t.elapsed
        timings["detection_gpu_memory_mb"] = get_peak_gpu_memory_mb()

        # Step 3: Marker matching
        with Timer() as t:
            result = components["matching_step"](result)
        timings["matching_time"] = t.elapsed

        # Step 4: Image rotation (with GPU memory tracking)
        reset_gpu_memory_stats()
        with Timer() as t:
            result = components["rotation_step"](result)
        timings["rotation_time"] = t.elapsed
        timings["rotation_gpu_memory_mb"] = get_peak_gpu_memory_mb()

        # Step 5: RoI masking
        with Timer() as t:
            result = components["masking_step"](result)
        timings["masking_time"] = t.elapsed

        # Step 6: Cell segmentation (if enabled)
        if self.segmenter is not None:
            cropped_image = result["image"]
            # Convert CxHxW to HxWxC if needed
            if cropped_image.ndim == 3 and cropped_image.shape[0] <= 4:
                cropped_image = np.moveaxis(cropped_image, 0, -1)
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            segm_input = cropped_rgb[None, :, :, :].astype(np.uint8)
            source = THWCSequenceSource(segm_input)

            reset_gpu_memory_stats()
            with Timer() as t, torch.no_grad():
                _ = self.segmenter(source.to_channel(0))
            timings["segmentation_time"] = t.elapsed
            timings["segmentation_gpu_memory_mb"] = get_peak_gpu_memory_mb()
        else:
            timings["segmentation_time"] = 0.0
            timings["segmentation_gpu_memory_mb"] = 0.0

        # Total time
        timings["total_time"] = (
            timings["detection_time"]
            + timings["matching_time"]
            + timings["rotation_time"]
            + timings["masking_time"]
            + timings["segmentation_time"]
        )

        return timings

    def run(self, image_list: list[tuple[str, str]], n_warmup: int = 3) -> list[dict]:
        """Run benchmark on all images.

        Args:
            image_list: List of (image_path, chamber_type) tuples
            n_warmup: Number of warmup iterations

        Returns:
            List of timing dictionaries, one per image
        """
        if not image_list:
            raise ValueError("No images to benchmark")

        # Load first image for warmup
        first_path, first_chamber = image_list[0]
        first_image = load_image(Path(first_path))
        self.warmup(first_image, first_chamber, n_warmup)

        # Benchmark all images
        results = []
        total = len(image_list)
        for i, (image_path, chamber_type) in enumerate(image_list):
            print(f"Processing {i + 1}/{total}: {Path(image_path).name} ({chamber_type})")
            try:
                results.append(self.benchmark_image(image_path, chamber_type))
            except Exception as e:
                print(f"  Error processing {Path(image_path).name}: {e}")
                continue

        return results


def export_csv(results: list[dict], output_path: Path):
    """Export benchmark results to CSV file.

    Args:
        results: List of timing dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to export")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nResults exported to: {output_path}")


def print_summary(results: list[dict], n_warmup: int, skip_segmentation: bool):
    """Print benchmark summary to console.

    Args:
        results: List of timing dictionaries
        n_warmup: Number of warmup iterations used
        skip_segmentation: Whether segmentation was skipped
    """
    if not results:
        print("No results to summarize")
        return

    n = len(results)

    # Count structure types
    structure_counts: dict[str, int] = {}
    for r in results:
        structure_name = r.get("structure_name", "unknown")
        structure_counts[structure_name] = structure_counts.get(structure_name, 0) + 1

    # Extract timing arrays
    detection_times = np.array([r["detection_time"] for r in results])
    matching_times = np.array([r["matching_time"] for r in results])
    rotation_times = np.array([r["rotation_time"] for r in results])
    masking_times = np.array([r["masking_time"] for r in results])
    segmentation_times = np.array([r["segmentation_time"] for r in results])
    total_times = np.array([r["total_time"] for r in results])
    detection_gpu_memory = np.array([r["detection_gpu_memory_mb"] for r in results])
    rotation_gpu_memory = np.array([r.get("rotation_gpu_memory_mb", 0.0) for r in results])
    segmentation_gpu_memory = np.array([r["segmentation_gpu_memory_mb"] for r in results])

    print("\n" + "=" * 60)
    print("Pipeline Benchmark Results")
    print("=" * 60)
    print(f"Images Processed:   {n}")
    print(f"Structure Types:    {', '.join(f'{k} ({v})' for k, v in structure_counts.items())}")
    print(f"Warmup Iterations:  {n_warmup}")
    print(f"Segmentation:       {'Disabled' if skip_segmentation else 'Enabled'}")

    print("\nStep Timings (seconds):")
    print(
        f"  Marker Detection: {detection_times.mean():.4f} +- {detection_times.std():.4f}"
        + (
            f"  [GPU: {detection_gpu_memory.mean():.1f} MB]"
            if detection_gpu_memory.mean() > 0
            else ""
        )
    )
    print(f"  Marker Matching:  {matching_times.mean():.4f} +- {matching_times.std():.4f}")
    print(
        f"  Image Rotation:   {rotation_times.mean():.4f} +- {rotation_times.std():.4f}"
        + (
            f"  [GPU: {rotation_gpu_memory.mean():.1f} MB]"
            if rotation_gpu_memory.mean() > 0
            else ""
        )
    )
    print(f"  RoI Masking:      {masking_times.mean():.4f} +- {masking_times.std():.4f}")
    if not skip_segmentation:
        print(
            f"  Segmentation:     {segmentation_times.mean():.4f} +- {segmentation_times.std():.4f}"
            + (
                f"  [GPU: {segmentation_gpu_memory.mean():.1f} MB]"
                if segmentation_gpu_memory.mean() > 0
                else ""
            )
        )
    print("  " + "-" * 56)
    if not skip_segmentation:
        # Show total without segmentation
        total_without_segm = total_times - segmentation_times
        print(
            f"  Total (w/o segm): {total_without_segm.mean():.4f} +- {total_without_segm.std():.4f}"
        )
        print(f"  Total (w/ segm):  {total_times.mean():.4f} +- {total_times.std():.4f}")
    else:
        print(f"  Total:            {total_times.mean():.4f} +- {total_times.std():.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the DMC masking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --input images.csv --output results.csv
  python scripts/benchmark.py --input images.csv --warmup 5 --skip-segmentation

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
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Output CSV path (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="Number of warmup iterations (default: 3)",
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
        args.model_path = Path(
            "/home/seiffarth_l/projects/DMC_new/dmc-train/runs/v8_detect_s_imgsz640/weights/best.pt"
        )  # Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Load image list from CSV
    image_list = load_image_list(args.input)
    print(f"Loaded {len(image_list)} images from {args.input}\n")

    if not image_list:
        print("No images to benchmark")
        return

    # Create benchmark runner
    benchmark = PipelineBenchmark(
        model_path=args.model_path,
        pixel_size=args.pixel_size,
        structure_library_path=args.structure_library,
        skip_segmentation=args.skip_segmentation,
        device=args.device,
    )

    # Run benchmark
    results = benchmark.run(image_list, n_warmup=args.warmup)

    # Export and summarize
    export_csv(results, args.output)
    print_summary(results, args.warmup, args.skip_segmentation)


if __name__ == "__main__":
    main()

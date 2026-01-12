#!/usr/bin/env python
"""Benchmark script for DMC masking pipeline.

Measures execution time and GPU memory usage for each pipeline step.
Supports warmup iterations, CSV export, and console summary with statistics.

Example usage:
    python scripts/benchmark.py --images tests/data/ --chamber-type NormaleBox-inner
    python scripts/benchmark.py --images image.png --chamber-type OpenBox-inner --warmup 5
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import dmc_masking
from dmc_masking import (
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
)
from dmc_masking.mask import SingleRoIStructureLibrary
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


def collect_images(input_path: Path) -> list[Path]:
    """Collect image paths from directory or single file.

    Args:
        input_path: Path to image directory or single image file

    Returns:
        List of image paths sorted alphabetically
    """
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in extensions])
    elif input_path.is_file():
        if input_path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")


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
        chamber_type: str,
        model_path: Path,
        pixel_size: float = 0.065789,
        roi_id: str = "0000",
        structure_library_path: Path | None = None,
    ):
        """Initialize the benchmark with pipeline configuration.

        Args:
            chamber_type: Name of the chamber type (e.g., "NormaleBox-inner")
            model_path: Path to the YOLO model weights
            pixel_size: Pixel size in micrometers
            roi_id: ROI identifier for structure lookup
            structure_library_path: Path to chamber structure JSON file
        """
        self.chamber_type = chamber_type

        # Default structure library path
        if structure_library_path is None:
            structure_library_path = (
                Path(dmc_masking.__file__).parent.parent / "artifacts/chamber_structure.json"
            )

        # Load chamber structure
        self.library = SingleRoIStructureLibrary(
            lookup_path=structure_library_path,
            structure_name=chamber_type,
            pixel_size=pixel_size,
        )
        _, self.roi_polygon, self.marker_group = self.library(roi_id)

        # Initialize pipeline steps
        self.detection_step = MarkerDetectionStep(model_path)
        self.matching_step = MarkerMatchingStep(self.marker_group, tolerance=60)
        self.rotation_step = ImageRotationStep()
        self.masking_step = RoIMaskingStep(self.marker_group, self.roi_polygon)

    def warmup(self, image: np.ndarray, n_warmup: int = 3):
        """Run warmup iterations to stabilize GPU/JIT compilation.

        Args:
            image: Input image for warmup
            n_warmup: Number of warmup iterations
        """
        print(f"Running {n_warmup} warmup iterations...")
        for i in range(n_warmup):
            self._run_pipeline(image)
            print(f"  Warmup {i + 1}/{n_warmup} complete")
        print("Warmup complete.\n")

    def _run_pipeline(self, image: np.ndarray):
        """Run full pipeline without timing.

        Args:
            image: Input image

        Returns:
            Pipeline result dictionary
        """
        result = self.detection_step(image)
        result = self.matching_step(result)
        result = self.rotation_step(result)
        result = self.masking_step(result)
        return result

    def benchmark_image(self, image_path: Path) -> dict:
        """Benchmark a single image through all pipeline steps.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with timing and memory measurements
        """
        timings = {"image_name": image_path.name}

        # Step 1: Load image
        with Timer() as t:
            image = load_image(image_path)
        timings["load_time"] = t.elapsed

        # Step 2: Marker detection (with GPU memory tracking)
        reset_gpu_memory_stats()
        with Timer() as t:
            result = self.detection_step(image)
        timings["detection_time"] = t.elapsed
        timings["detection_gpu_memory_mb"] = get_peak_gpu_memory_mb()

        # Step 3: Marker matching
        with Timer() as t:
            result = self.matching_step(result)
        timings["matching_time"] = t.elapsed

        # Step 4: Image rotation
        with Timer() as t:
            result = self.rotation_step(result)
        timings["rotation_time"] = t.elapsed

        # Step 5: RoI masking
        with Timer() as t:
            result = self.masking_step(result)
        timings["masking_time"] = t.elapsed

        # Total time
        timings["total_time"] = (
            timings["load_time"]
            + timings["detection_time"]
            + timings["matching_time"]
            + timings["rotation_time"]
            + timings["masking_time"]
        )

        return timings

    def run(self, image_paths: list[Path], n_warmup: int = 3) -> list[dict]:
        """Run benchmark on all images.

        Args:
            image_paths: List of image paths to benchmark
            n_warmup: Number of warmup iterations

        Returns:
            List of timing dictionaries, one per image
        """
        if not image_paths:
            raise ValueError("No images to benchmark")

        # Load first image for warmup
        first_image = load_image(image_paths[0])
        self.warmup(first_image, n_warmup)

        # Benchmark all images
        results = []
        for i, path in enumerate(image_paths):
            print(f"Processing {i + 1}/{len(image_paths)}: {path.name}")
            try:
                results.append(self.benchmark_image(path))
            except Exception as e:
                print(f"  Error processing {path.name}: {e}")
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

    fieldnames = [
        "image_name",
        "load_time",
        "detection_time",
        "detection_gpu_memory_mb",
        "matching_time",
        "rotation_time",
        "masking_time",
        "total_time",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults exported to: {output_path}")


def print_summary(results: list[dict], chamber_type: str, n_warmup: int):
    """Print benchmark summary to console.

    Args:
        results: List of timing dictionaries
        chamber_type: Name of the chamber type
        n_warmup: Number of warmup iterations used
    """
    if not results:
        print("No results to summarize")
        return

    n = len(results)

    # Extract timing arrays
    load_times = np.array([r["load_time"] for r in results])
    detection_times = np.array([r["detection_time"] for r in results])
    matching_times = np.array([r["matching_time"] for r in results])
    rotation_times = np.array([r["rotation_time"] for r in results])
    masking_times = np.array([r["masking_time"] for r in results])
    total_times = np.array([r["total_time"] for r in results])
    gpu_memory = np.array([r["detection_gpu_memory_mb"] for r in results])

    print("\n" + "=" * 50)
    print("Pipeline Benchmark Results")
    print("=" * 50)
    print(f"Chamber Type:       {chamber_type}")
    print(f"Images Processed:   {n}")
    print(f"Warmup Iterations:  {n_warmup}")

    print("\nStep Timings (seconds):")
    print(f"  Image Loading:    {load_times.mean():.4f} +- {load_times.std():.4f}")
    print(
        f"  Marker Detection: {detection_times.mean():.4f} +- {detection_times.std():.4f}"
        + (f"  [GPU: {gpu_memory.mean():.1f} MB]" if gpu_memory.mean() > 0 else "")
    )
    print(f"  Marker Matching:  {matching_times.mean():.4f} +- {matching_times.std():.4f}")
    print(f"  Image Rotation:   {rotation_times.mean():.4f} +- {rotation_times.std():.4f}")
    print(f"  RoI Masking:      {masking_times.mean():.4f} +- {masking_times.std():.4f}")
    print("  " + "-" * 46)
    print(f"  Total:            {total_times.mean():.4f} +- {total_times.std():.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the DMC masking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --images tests/data/ --chamber-type NormaleBox-inner
  python scripts/benchmark.py --images image.png --chamber-type OpenBox-inner --warmup 5
  python scripts/benchmark.py --images ./images/ --chamber-type BigBox-inner --output results.csv
        """,
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to image directory or single image file",
    )
    parser.add_argument(
        "--chamber-type",
        type=str,
        required=True,
        choices=CHAMBER_TYPES,
        help="Chamber type name",
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
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--roi-id",
        type=str,
        default="0000",
        help="ROI ID for structure lookup (default: 0000)",
    )
    parser.add_argument(
        "--structure-library",
        type=Path,
        default=None,
        help="Path to chamber structure JSON file",
    )

    args = parser.parse_args()

    # Set default model path
    if args.model_path is None:
        args.model_path = Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"

    # Validate paths
    if not args.images.exists():
        raise FileNotFoundError(f"Images path not found: {args.images}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Collect images
    image_paths = collect_images(args.images)
    print(f"Found {len(image_paths)} image(s) to benchmark\n")

    # Create benchmark runner
    benchmark = PipelineBenchmark(
        chamber_type=args.chamber_type,
        model_path=args.model_path,
        pixel_size=args.pixel_size,
        roi_id=args.roi_id,
        structure_library_path=args.structure_library,
    )

    # Run benchmark
    results = benchmark.run(image_paths, n_warmup=args.warmup)

    # Export and summarize
    export_csv(results, args.output)
    print_summary(results, args.chamber_type, args.warmup)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Test script for DMC Masking API endpoints.

Tests the /health, /chamber-types, /process-image, and /calibrate endpoints
using sample calibration data.

Usage:
    python scripts/test_api_endpoints.py
    python scripts/test_api_endpoints.py --base-url http://localhost:8000
    python scripts/test_api_endpoints.py --data-dir scripts/calibration_sample
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"✗ {message}", file=sys.stderr)


def test_health(base_url: str) -> bool:
    """Test the /health endpoint."""
    print_section("Testing /health endpoint")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model_loaded')}")
        print(f"GPU available: {data.get('gpu_available')}")
        print(f"Device: {data.get('device')}")
        print(f"Default structure library: {data.get('default_structure_library')}")

        if data.get("status") == "healthy" and data.get("model_loaded"):
            print_success("Health check passed")
            return True
        else:
            print_error("Health check failed: service not healthy or model not loaded")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_chamber_types(base_url: str) -> bool:
    """Test the /chamber-types endpoint."""
    print_section("Testing /chamber-types endpoint")

    try:
        response = requests.get(f"{base_url}/chamber-types", timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"Found {len(data)} chamber types:")
        for chamber_type in data:
            print(f"  - {chamber_type['name']}: {chamber_type['roi_pattern'][:50]}...")

        print_success("Chamber types retrieved successfully")
        return True
    except Exception as e:
        print_error(f"Failed to get chamber types: {e}")
        return False


def test_process_image(base_url: str, image_path: Path, roi_id: str, pixel_size: float) -> bool:
    """Test the /process-image endpoint."""
    print_section(f"Testing /process-image endpoint with {image_path.name}")

    try:
        # Prepare the request
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/tiff")}
            data = {
                "roi_id": roi_id,
                "pixel_size": pixel_size,
                "return_uncropped": False,
            }

            print(f"  Image: {image_path.name}")
            print(f"  ROI ID: {roi_id}")
            print(f"  Pixel size: {pixel_size}")
            print("  Uploading and processing...")

            start_time = time.time()
            response = requests.post(
                f"{base_url}/process-image",
                files=files,
                data=data,
                timeout=60,
            )
            elapsed = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            print_success(f"Image processed successfully in {elapsed:.2f}s")
            print(f"  Chamber type: {result.get('chamber_type')}")
            print(f"  Rotation angle: {result.get('rotation_angle', 0):.2f}°")
            print(f"  Cropped image size: {len(result.get('cropped_image', ''))} bytes (base64)")
            print(f"  Mask size: {len(result.get('mask', ''))} bytes (base64)")
            return True
        else:
            print_error(f"Image processing failed: {result.get('error_message')}")
            return False

    except Exception as e:
        print_error(f"Failed to process image: {e}")
        return False


def test_calibrate(base_url: str, config_path: Path, data_dir: Path) -> bool:
    """Test the /calibrate endpoint."""
    print_section("Testing /calibrate endpoint")

    try:
        # Load config
        with open(config_path) as f:
            config = json.load(f)

        print(f"Loaded config with {len(config['calibration_images'])} calibration images")
        print(f"  Pixel size: {config['pixel_size']}")
        print(f"  Blueprint map: {config['blueprint_map_path']}")

        # Prepare the calibration config for API (without image paths)
        api_config = {
            "calibration_images": [
                {
                    "roi_id": img["roi_id"],
                    "stage_position": img["stage_position"],
                }
                for img in config["calibration_images"]
            ],
            "pixel_size": config["pixel_size"],
            "blueprint_map_path": config["blueprint_map_path"],
        }

        # Prepare files - read into memory to avoid open file handles
        files = []
        for img_info in config["calibration_images"]:
            img_path = data_dir / img_info["image_path"]
            if not img_path.exists():
                print_error(f"Image not found: {img_path}")
                return False
            # Read file contents into memory
            with open(img_path, "rb") as f:
                file_contents = f.read()
            files.append(("images", (img_path.name, file_contents, "image/tiff")))

        # Prepare form data
        form_data = {
            "config": json.dumps(api_config),
        }

        print("  Uploading images and running calibration...")
        start_time = time.time()

        response = requests.post(
            f"{base_url}/calibrate",
            files=files,
            data=form_data,
            timeout=120,
        )
        elapsed = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            print_success(f"Calibration completed successfully in {elapsed:.2f}s")

            # Print statistics
            stats = result.get("statistics", {})
            print("\n  Statistics:")
            print(f"    RMSE: {stats.get('rmse', 0):.4f}")
            print(f"    Max error: {stats.get('max_error', 0):.4f}")
            print(f"    Number of points: {stats.get('n_points', 0)}")

            # Print image results
            print("\n  Image results:")
            for img_result in result.get("image_results", []):
                status = "✓" if img_result.get("success") else "✗"
                print(f"    {status} ROI {img_result.get('roi_id')}: ", end="")
                if img_result.get("success"):
                    pos = img_result.get("microscope_position", [])
                    if pos:
                        print(f"position = ({pos[0]:.2f}, {pos[1]:.2f})")
                    else:
                        print("success")
                else:
                    print(f"failed - {img_result.get('error_message')}")

            # Show first few calibrated map entries
            cal_map = result.get("calibrated_map", [])
            print(f"\n  Calibrated map ({len(cal_map)} entries, first 5):")
            for entry in cal_map[:5]:
                print(f"    {entry}")
            if len(cal_map) > 5:
                print(f"    ... ({len(cal_map) - 5} more entries)")

            return True
        else:
            print_error(f"Calibration failed: {result.get('error_message')}")
            return False

    except Exception as e:
        print_error(f"Failed to run calibration: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test DMC Masking API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "calibration_sample",
        help="Directory containing test data (default: scripts/calibration_sample)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check",
    )
    parser.add_argument(
        "--skip-chamber-types",
        action="store_true",
        help="Skip chamber types test",
    )
    parser.add_argument(
        "--skip-process-image",
        action="store_true",
        help="Skip process-image tests",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Skip calibration test",
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print_error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    config_path = args.data_dir / "calibration_config.json"
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nTesting DMC Masking API at: {args.base_url}")
    print(f"Using data from: {args.data_dir}")

    # Track results
    results = {}

    # Test health
    if not args.skip_health:
        results["health"] = test_health(args.base_url)
        if not results["health"]:
            print_error("\nHealth check failed. Aborting tests.")
            sys.exit(1)

    # Test chamber types
    if not args.skip_chamber_types:
        results["chamber_types"] = test_chamber_types(args.base_url)

    # Test process-image for each calibration image
    if not args.skip_process_image:
        results["process_image"] = []
        for img_info in config["calibration_images"]:
            img_path = args.data_dir / img_info["image_path"]
            success = test_process_image(
                args.base_url,
                img_path,
                img_info["roi_id"],
                config["pixel_size"],
            )
            results["process_image"].append(success)

    # Test calibration
    if not args.skip_calibrate:
        results["calibrate"] = test_calibrate(args.base_url, config_path, args.data_dir)

    # Print summary
    print_section("Test Summary")

    total_tests = 0
    passed_tests = 0

    if "health" in results:
        total_tests += 1
        if results["health"]:
            passed_tests += 1
        status = "✓" if results["health"] else "✗"
        print(f"{status} Health check")

    if "chamber_types" in results:
        total_tests += 1
        if results["chamber_types"]:
            passed_tests += 1
        status = "✓" if results["chamber_types"] else "✗"
        print(f"{status} Chamber types")

    if "process_image" in results:
        for i, success in enumerate(results["process_image"]):
            total_tests += 1
            if success:
                passed_tests += 1
            status = "✓" if success else "✗"
            img_name = config["calibration_images"][i]["image_path"]
            print(f"{status} Process image: {img_name}")

    if "calibrate" in results:
        total_tests += 1
        if results["calibrate"]:
            passed_tests += 1
        status = "✓" if results["calibrate"] else "✗"
        print(f"{status} Calibration")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print_success("\nAll tests passed!")
        sys.exit(0)
    else:
        print_error(f"\n{total_tests - passed_tests} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

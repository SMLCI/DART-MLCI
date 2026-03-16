#!/usr/bin/env python3
"""
Comprehensive Docker API integration tests using real images from artifacts.

This script tests the DMC Masking API running in a Docker container
with real images from the artifacts and test fixtures folders.

Usage:
    # Start the container first
    docker run -d -p 8000:8000 --name dart-mlci-api jugit-registry.fz-juelich.de/emsig/dart-mlci:latest

    # Run tests
    python test_docker_integration.py

    # Or specify custom URL
    TEST_API_URL=http://localhost:8001 python test_docker_integration.py
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

# Import dart_mlci for proper TIFF loading
sys.path.insert(0, str(Path(__file__).parent))
from dart_mlci.io import load_image

# Configuration
API_URL = os.environ.get("TEST_API_URL", "http://localhost:8000")
FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR = Path(__file__).parent / "tests" / "test_output" / "docker_api_visualizations"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text):
    """Print a colored header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string.

    For TIFF files, uses dart_mlci.io.load_image for proper handling,
    then converts to PNG before encoding.
    """
    image_path = Path(image_path)

    # Check if it's a TIFF file
    if image_path.suffix.lower() in [".tif", ".tiff"]:
        # Load TIFF using dart_mlci's load_image function
        # This properly handles multi-dimensional TIFFs and normalization
        img_array = load_image(image_path)  # Returns HxWx3 uint8 array

        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array)

        # Save as PNG to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
    else:
        # For non-TIFF files, encode directly
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def test_health_endpoint():
    """Test the /health endpoint."""
    print_header("Testing Health Endpoint")

    try:
        response = requests.get(f"{API_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print_success("Health endpoint returned 200 OK")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Model loaded: {data.get('model_loaded')}")
            print_info(f"GPU available: {data.get('gpu_available')}")
            print_info(f"Device: {data.get('device')}")
            return True
        else:
            print_error(f"Health endpoint returned {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to connect to API: {e}")
        print_warning(f"Make sure the Docker container is running on {API_URL}")
        return False


def test_chamber_types_endpoint():
    """Test the /chamber-types endpoint."""
    print_header("Testing Chamber Types Endpoint")

    try:
        response = requests.get(f"{API_URL}/chamber-types", timeout=10)

        if response.status_code == 200:
            chamber_types = response.json()
            print_success(f"Chamber types endpoint returned {len(chamber_types)} types")
            for ct in chamber_types[:5]:  # Show first 5
                print_info(f"  - {ct['name']}: {ct['roi_pattern']}")
            if len(chamber_types) > 5:
                print_info(f"  ... and {len(chamber_types) - 5} more")
            return True
        elif response.status_code == 503:
            print_warning("Structure library not loaded in container")
            return False
        else:
            print_error(f"Chamber types endpoint returned {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to query chamber types: {e}")
        return False


def test_process_image_endpoint():
    """Test the /process-image endpoint with real images."""
    print_header("Testing Process Image Endpoint")

    # Use SAK artifact image which should have detectable markers
    test_image_path = Path(
        "/home/seiffarth_l/projects/DART_new/dart-mlci/artifacts/images/sak/0000.png"
    )

    if not test_image_path.exists():
        print_warning(f"Test image not found: {test_image_path}")
        return False

    print_info(f"Using test image: {test_image_path}")

    # Encode image to base64
    try:
        image_b64 = encode_image_to_base64(test_image_path)
        print_success(f"Encoded image to base64 ({len(image_b64)} chars)")
    except Exception as e:
        print_error(f"Failed to encode image: {e}")
        return False

    # Send request
    request_data = {
        "image": image_b64,
        "roi_id": "0000",
        "pixel_size": 0.065789,
    }

    print_info("Sending POST request to /process-image...")

    try:
        response = requests.post(
            f"{API_URL}/process-image",
            json=request_data,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("success"):
                print_success("Image processing successful!")
                print_info("ROI ID: 0000")
                print_info(f"Chamber type: {data.get('chamber_type', 'N/A')}")
                print_info(f"Rotation angle: {data.get('rotation_angle', 0):.2f}°")
                print_info(f"Cropped image size: {len(data.get('cropped_image', ''))} chars")
                print_info(f"Mask size: {len(data.get('mask', ''))} chars")

                # Validate base64 outputs
                try:
                    cropped_bytes = base64.b64decode(data["cropped_image"])
                    mask_bytes = base64.b64decode(data["mask"])
                    print_success(
                        f"Output images are valid base64 (cropped: {len(cropped_bytes)} bytes, mask: {len(mask_bytes)} bytes)"
                    )
                except Exception as e:
                    print_error(f"Failed to decode output images: {e}")
                    return False

                return True
            else:
                error_msg = data.get("error_message", "Unknown error")
                print_error(f"Processing failed: {error_msg}")
                return False
        else:
            print_error(f"Process image endpoint returned {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Failed to process image: {e}")
        return False


def test_calibrate_endpoint():
    """Test the /calibrate endpoint with calibration images."""
    print_header("Testing Calibrate Endpoint")

    # Check for calibration sample
    cal_dir = Path(__file__).parent / "scripts" / "calibration_sample"
    if not cal_dir.exists():
        print_warning(f"Calibration sample not found at {cal_dir}")
        return False

    # Load calibration config
    cal_config_path = cal_dir / "calibration_config.json"
    if not cal_config_path.exists():
        print_warning(f"Calibration config not found at {cal_config_path}")
        return False

    import json

    with open(cal_config_path) as f:
        config = json.load(f)

    # Load first 3 calibration images
    calibration_images = []
    for img_meta in config["calibration_images"][:3]:
        roi_id = img_meta["roi_id"]
        img_path = cal_dir / f"{roi_id}.tif"

        if not img_path.exists():
            print_warning(f"Calibration image {img_path} not found")
            continue

        try:
            image_b64 = encode_image_to_base64(img_path)
            calibration_images.append(
                {
                    "image": image_b64,
                    "roi_id": roi_id,
                    "stage_position": img_meta["stage_position"],
                }
            )
            print_info(f"Loaded calibration image: {roi_id}")
        except Exception as e:
            print_error(f"Failed to load {img_path}: {e}")

    if len(calibration_images) < 3:
        print_warning(f"Need at least 3 calibration images, found {len(calibration_images)}")
        return False

    # Build request
    request_data = {
        "chip_name": "SAK",
        "calibration_images": calibration_images,
        "pixel_size": config["pixel_size"],
        "blueprint_map_path": config["blueprint_map_path"],
    }

    print_info(f"Sending calibration request with {len(calibration_images)} images...")

    try:
        response = requests.post(
            f"{API_URL}/calibrate",
            json=request_data,
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("success"):
                print_success("Calibration successful!")
                stats = data.get("statistics", {})
                print_info(f"RMSE: {stats.get('rmse', 'N/A'):.4f}")
                print_info(f"Max error: {stats.get('max_error', 'N/A'):.4f}")
                print_info(f"Points: {stats.get('n_points', 'N/A')}")

                cal_map = data.get("calibrated_map", [])
                print_info(f"Calibrated map: {len(cal_map)} entries")

                return True
            else:
                error_msg = data.get("error_message", "Unknown error")
                print_error(f"Calibration failed: {error_msg}")
                return False
        else:
            print_error(f"Calibrate endpoint returned {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Failed to calibrate: {e}")
        return False


def test_segment_pipeline():
    """Test the /process-image -> /segment pipeline with fixture assertions."""
    print_header("Testing Segment Pipeline (process-image -> segment)")

    # Use the calibration test fixture
    test_image_path = FIXTURES_DIR / "calibration_image_0000.tif"

    if not test_image_path.exists():
        print_warning(f"Test image not found: {test_image_path}")
        return False

    print_info(f"Using test image: {test_image_path}")

    # Expected fixture values (from local pipeline run with cellpose-sam)
    EXPECTED_CELL_COUNT = 185
    EXPECTED_TOTAL_CELL_AREA = 135913
    CELL_COUNT_TOLERANCE = 0.30  # 30%
    CELL_AREA_TOLERANCE = 0.40  # 40%

    # Encode image
    try:
        image_b64 = encode_image_to_base64(test_image_path)
    except Exception as e:
        print_error(f"Failed to encode image: {e}")
        return False

    # Step 1: Check health for segmenter status
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=10)
        health = health_resp.json()
        if not health.get("segmenter_loaded"):
            print_warning("Segmenter not loaded in container (DART_SEGMENTER not set?)")
            return False
        print_success(f"Segmenter loaded: {health.get('segmenter')}")
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

    # Step 2: Process image
    print_info("Step 1: Processing image via /process-image...")
    try:
        proc_resp = requests.post(
            f"{API_URL}/process-image",
            json={
                "image": image_b64,
                "roi_id": "0000",
                "pixel_size": 0.065789,
            },
            timeout=120,
        )
        proc_data = proc_resp.json()
        if not proc_data.get("success"):
            print_error(f"Process image failed: {proc_data.get('error_message')}")
            return False
        print_success(f"Process image succeeded (chamber: {proc_data['chamber_type']})")
    except Exception as e:
        print_error(f"Process image request failed: {e}")
        return False

    # Step 3: Segment
    print_info("Step 2: Running segmentation via /segment...")
    try:
        seg_resp = requests.post(
            f"{API_URL}/segment",
            json={
                "image": proc_data["cropped_image"],
                "mask": proc_data["mask"],
            },
            timeout=120,
        )
        seg_data = seg_resp.json()
        if not seg_data.get("success"):
            print_error(f"Segmentation failed: {seg_data.get('error_message')}")
            return False

        cell_count = seg_data["cell_count"]
        total_area = seg_data["total_cell_area"]
        print_success(f"Segmentation succeeded: {cell_count} cells, {total_area} px total area")
    except Exception as e:
        print_error(f"Segment request failed: {e}")
        return False

    # Step 4: Validate against expected fixtures
    ok = True

    count_lo = EXPECTED_CELL_COUNT * (1 - CELL_COUNT_TOLERANCE)
    count_hi = EXPECTED_CELL_COUNT * (1 + CELL_COUNT_TOLERANCE)
    if count_lo <= cell_count <= count_hi:
        print_success(
            f"Cell count {cell_count} within expected range "
            f"[{count_lo:.0f}, {count_hi:.0f}] (expected ~{EXPECTED_CELL_COUNT})"
        )
    else:
        print_error(
            f"Cell count {cell_count} OUTSIDE expected range "
            f"[{count_lo:.0f}, {count_hi:.0f}] (expected ~{EXPECTED_CELL_COUNT})"
        )
        ok = False

    area_lo = EXPECTED_TOTAL_CELL_AREA * (1 - CELL_AREA_TOLERANCE)
    area_hi = EXPECTED_TOTAL_CELL_AREA * (1 + CELL_AREA_TOLERANCE)
    if area_lo <= total_area <= area_hi:
        print_success(
            f"Total cell area {total_area} within expected range "
            f"[{area_lo:.0f}, {area_hi:.0f}] (expected ~{EXPECTED_TOTAL_CELL_AREA})"
        )
    else:
        print_error(
            f"Total cell area {total_area} OUTSIDE expected range "
            f"[{area_lo:.0f}, {area_hi:.0f}] (expected ~{EXPECTED_TOTAL_CELL_AREA})"
        )
        ok = False

    # Step 5: Validate the mask can be decoded as 16-bit PNG
    try:
        mask_bytes = base64.b64decode(seg_data["segmentation_mask"])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        print_success(
            f"Segmentation mask is valid PNG (mode={mask_img.mode}, size={mask_img.size})"
        )
    except Exception as e:
        print_error(f"Failed to decode segmentation mask: {e}")
        ok = False

    return ok


def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}DMC Masking Docker API Integration Tests{Colors.RESET}")
    print(f"API URL: {Colors.BOLD}{API_URL}{Colors.RESET}")
    print(f"Fixtures: {FIXTURES_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")

    # Wait a moment for API to be ready
    print_info("Waiting for API to be ready...")
    time.sleep(2)

    results = {}

    # Run tests
    results["health"] = test_health_endpoint()
    results["chamber_types"] = test_chamber_types_endpoint()
    results["process_image"] = test_process_image_endpoint()
    results["segment_pipeline"] = test_segment_pipeline()
    results["calibrate"] = test_calibrate_endpoint()

    # Summary
    print_header("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "PASSED" if passed_test else "FAILED"
        color = Colors.GREEN if passed_test else Colors.RED
        print(f"{color}{status:8}{Colors.RESET} {test_name}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}\n")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

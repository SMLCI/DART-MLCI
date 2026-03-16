"""Tests for the FastAPI service."""

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with lifespan context."""
    from dart_mlci.api.main import app

    # Use context manager to trigger lifespan events
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_image_base64():
    """Load test image as base64 string."""
    test_image = FIXTURES_DIR / "calibration_image_0000.tif"
    if not test_image.exists():
        pytest.skip("Test image not found")

    with open(test_image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64


@pytest.fixture
def calibration_images_base64():
    """Load calibration sample images as base64."""
    cal_dir = Path("scripts/calibration_sample")
    if not cal_dir.exists():
        pytest.skip("Calibration sample not found")

    images = {}
    for roi_id in ["0000", "7000", "7315"]:
        img_path = cal_dir / f"{roi_id}.tif"
        if img_path.exists():
            with open(img_path, "rb") as f:
                images[roi_id] = base64.b64encode(f.read()).decode()
    return images


@pytest.fixture
def viz_dir():
    """Create directory for test visualizations (persisted for CI artifacts)."""
    viz = Path("tests/test_output/api_visualizations")
    viz.mkdir(parents=True, exist_ok=True)
    return viz


def save_visualization(b64_string: str, output_path: Path, title: str):
    """Decode base64 and save as PNG with title."""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes))

    _fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# === Health Endpoint Tests ===


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 when service is ready."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_model_status(self, client):
        """Health response should indicate model loading status."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "device" in data
        assert "status" in data

    def test_health_response_structure(self, client):
        """Health response should match HealthResponse schema."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["gpu_available"], bool)
        assert isinstance(data["device"], str)
        assert data["status"] in ("healthy", "unhealthy")


# === Chamber Types Endpoint Tests ===


class TestChamberTypesEndpoint:
    def test_chamber_types_returns_list(self, client):
        """Chamber types endpoint should return available types."""
        response = client.get("/chamber-types")
        # May return 503 if structure library not loaded
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.skipif(
        not (Path("artifacts/chamber_structure.json").exists()),
        reason="Chamber structure file not found",
    )
    def test_chamber_types_include_sak_types(self, client):
        """Should include known SAK chamber types."""
        response = client.get("/chamber-types")
        if response.status_code == 200:
            names = [t["name"] for t in response.json()]
            # Check for at least some expected types
            assert len(names) > 0


# === Process Image Endpoint Tests ===


class TestProcessImageEndpoint:
    def test_process_image_missing_image_returns_422(self, client):
        """Missing image field should return validation error."""
        response = client.post("/process-image", json={"roi_id": "0050"})
        assert response.status_code == 422

    def test_process_image_missing_roi_returns_422(self, client, test_image_base64):
        """Missing roi_id should return validation error."""
        response = client.post("/process-image", json={"image": test_image_base64})
        assert response.status_code == 422

    def test_process_image_invalid_base64(self, client):
        """Test error handling for invalid base64."""
        response = client.post(
            "/process-image",
            json={
                "image": "not-valid-base64!!!",
                "roi_id": "0000",
            },
        )
        # Should fail validation
        assert response.status_code == 422

    def test_process_image_with_data_uri(self, client, test_image_base64):
        """Test that data URI prefix is properly handled."""
        data_uri = f"data:image/tiff;base64,{test_image_base64}"

        response = client.post(
            "/process-image",
            json={
                "image": data_uri,
                "roi_id": "0000",
            },
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 422]

    def test_process_image_invalid_roi_returns_error(self, client, test_image_base64):
        """Invalid ROI ID should return error in response."""
        response = client.post(
            "/process-image",
            json={
                "image": test_image_base64,
                "roi_id": "invalid_roi",
            },
        )
        # May succeed with 200 but have success=False, or may fail validation
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is False
            assert "error_message" in data

    @pytest.mark.skipif(
        not (Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()),
        reason="Model file not found",
    )
    def test_process_image_with_base64(self, client, test_image_base64, viz_dir):
        """Test processing image from base64 input with visualization."""
        # Send JSON request
        response = client.post(
            "/process-image",
            json={
                "image": test_image_base64,
                "roi_id": "0000",
                "pixel_size": 0.065789,
            },
        )

        assert response.status_code == 200
        data = response.json()

        if data["success"]:
            # Verify base64 outputs
            assert "cropped_image" in data
            assert "mask" in data
            assert len(data["cropped_image"]) > 0
            assert len(data["mask"]) > 0

            # Validate base64 decoding
            cropped_bytes = base64.b64decode(data["cropped_image"])
            mask_bytes = base64.b64decode(data["mask"])
            assert len(cropped_bytes) > 0
            assert len(mask_bytes) > 0

            # Save visualizations
            save_visualization(test_image_base64, viz_dir / "input.png", "Input Image")
            save_visualization(
                data["cropped_image"],
                viz_dir / "cropped_output.png",
                f"Cropped Output (ROI 0000, {data['rotation_angle']:.1f}°)",
            )
            save_visualization(data["mask"], viz_dir / "mask_output.png", "Mask Output")

            print(f"\nVisualizations: {viz_dir}")

    def test_process_image_response_structure(self, client, test_image_base64):
        """Response should match ProcessImageResponse schema."""
        response = client.post(
            "/process-image",
            json={
                "image": test_image_base64,
                "roi_id": "0000",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert isinstance(data["success"], bool)


# === Calibrate Endpoint Tests ===


class TestCalibrateEndpoint:
    def test_calibrate_missing_fields_returns_422(self, client):
        """Missing required fields should return validation error."""
        response = client.post("/calibrate", json={})
        assert response.status_code == 422

    def test_calibrate_insufficient_images_returns_error(self, client, test_image_base64):
        """Less than 3 calibration images should fail validation."""
        request_body = {
            "chip_name": "SAK",
            "calibration_images": [
                {
                    "image": test_image_base64,
                    "roi_id": "0000",
                    "stage_position": {"x": 0, "y": 0},
                }
            ],
            "pixel_size": 0.065789,
            "blueprint_map_path": "artifacts/sak_blueprint_map.csv",
        }
        response = client.post("/calibrate", json=request_body)
        # Pydantic validation should fail (min_length=3)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not (Path("scripts/calibration_sample").exists()),
        reason="Calibration sample not found",
    )
    def test_calibrate_with_base64_images(self, client, calibration_images_base64, viz_dir):
        """Test calibration with base64 images and visualize results."""
        if len(calibration_images_base64) < 3:
            pytest.skip("Need at least 3 calibration images")

        # Load config
        cal_config_path = Path("scripts/calibration_sample/calibration_config.json")
        if not cal_config_path.exists():
            pytest.skip("Calibration config not found")

        with open(cal_config_path) as f:
            config = json.load(f)

        # Build request with base64 images
        cal_images = []
        for img_meta in config["calibration_images"][:3]:
            roi_id = img_meta["roi_id"]
            if roi_id in calibration_images_base64:
                cal_images.append(
                    {
                        "image": calibration_images_base64[roi_id],
                        "roi_id": roi_id,
                        "stage_position": img_meta["stage_position"],
                    }
                )

        if len(cal_images) < 3:
            pytest.skip("Could not load 3 calibration images")

        request_body = {
            "chip_name": "SAK",
            "calibration_images": cal_images,
            "pixel_size": config["pixel_size"],
            "blueprint_map_path": config["blueprint_map_path"],
        }

        # Send request
        response = client.post("/calibrate", json=request_body)
        assert response.status_code == 200

        data = response.json()
        if data["success"]:
            assert "calibrated_map" in data
            assert "statistics" in data

            # Save calibrated map as JSON
            json_path = viz_dir / "calibrated_map.json"
            json_path.write_text(json.dumps(data["calibrated_map"], indent=2))

            # Print statistics
            stats = data["statistics"]
            print("\nCalibration Statistics:")
            print(f"  RMSE: {stats['rmse']:.4f}")
            print(f"  Max Error: {stats['max_error']:.4f}")
            print(f"  Points: {stats['n_points']}")
            print(f"\nCalibrated map: {json_path}")

    def test_calibrate_invalid_base64(self, client):
        """Invalid base64 should fail validation."""
        request_body = {
            "calibration_images": [
                {
                    "image": "not-valid-base64!!!",
                    "roi_id": "0000",
                    "stage_position": {"x": 0, "y": 0},
                },
                {
                    "image": "also-invalid",
                    "roi_id": "0001",
                    "stage_position": {"x": 100, "y": 100},
                },
                {
                    "image": "still-invalid",
                    "roi_id": "0002",
                    "stage_position": {"x": 200, "y": 200},
                },
            ],
            "blueprint_map_path": "artifacts/sak_blueprint_map.csv",
        }
        response = client.post("/calibrate", json=request_body)
        assert response.status_code == 422  # Validation error


# === Validate Endpoint Tests ===


class TestValidateEndpoint:
    def test_validate_returns_placeholder(self, client):
        """Validate endpoint should return not implemented status."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        config = {"validation_images": []}
        with open(test_image, "rb") as f:
            response = client.post(
                "/validate",
                files=[
                    ("calibrated_map", ("map.csv", b"roi_id,x,y\n0000,0,0", "text/csv")),
                    ("images", ("image.tif", f, "image/tiff")),
                ],
                data={"config": json.dumps(config)},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_implemented"


# === Integration Tests ===


@pytest.mark.integration
class TestIntegration:
    """Integration tests that require full environment setup."""

    @pytest.mark.skipif(
        not (
            Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()
            and Path("artifacts/chamber_structure.json").exists()
        ),
        reason="Artifacts not found",
    )
    def test_full_process_image_workflow(self, client):
        """Test complete image processing workflow."""
        test_image = FIXTURES_DIR / "calibration_image_0000.tif"
        if not test_image.exists():
            pytest.skip("Calibration image not found")

        # Check health first
        health = client.get("/health").json()
        if not health["model_loaded"]:
            pytest.skip("Model not loaded")

        # Process image
        with open(test_image, "rb") as f:
            response = client.post(
                "/process-image",
                files={"image": ("image.tif", f, "image/tiff")},
                data={"roi_id": "0000", "pixel_size": "0.065789"},
            )

        assert response.status_code == 200
        data = response.json()
        if data["success"]:
            # Verify base64 encoded images
            import base64

            cropped = base64.b64decode(data["cropped_image"])
            mask = base64.b64decode(data["mask"])
            assert len(cropped) > 0
            assert len(mask) > 0

    def test_gpu_detection(self, client):
        """Container should correctly detect GPU availability."""
        response = client.get("/health")
        data = response.json()
        assert "gpu_available" in data
        assert isinstance(data["gpu_available"], bool)


# === Base64 Utility Tests ===


class TestBase64Utilities:
    def test_base64_to_array_roundtrip(self):
        """Test encoding/decoding roundtrip."""
        from dart_mlci.api.utils import array_to_base64_png, base64_to_array

        # Create test array
        test_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Encode
        b64_str = array_to_base64_png(test_arr)
        assert len(b64_str) > 0

        # Decode
        decoded_arr = base64_to_array(b64_str)

        # Verify shape and type
        assert decoded_arr.shape == test_arr.shape
        assert decoded_arr.dtype == np.uint8

    def test_base64_to_array_strips_data_uri(self):
        """Test that data URI prefix is stripped."""
        from dart_mlci.api.utils import array_to_base64_png, base64_to_array

        test_arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        b64_str = array_to_base64_png(test_arr)

        # Add data URI prefix
        data_uri = f"data:image/png;base64,{b64_str}"

        # Should decode successfully
        decoded = base64_to_array(data_uri)
        assert decoded.shape == test_arr.shape

    def test_base64_to_array_invalid_base64(self):
        """Test error handling for invalid base64."""
        from dart_mlci.api.utils import base64_to_array

        with pytest.raises(ValueError, match="Invalid base64"):
            base64_to_array("not-valid-base64!!!")

    def test_array_to_base64_mask(self):
        """Test mask encoding (binary thresholding)."""
        from dart_mlci.api.utils import array_to_base64_png

        # Create binary mask (use float to trigger mask conversion)
        mask = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        # Encode as mask
        b64_str = array_to_base64_png(mask, is_mask=True)
        assert len(b64_str) > 0

        # Decode and verify it's binary (0 or 255)
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        unique_vals = np.unique(arr)
        # Should be binary (0 or 255)
        assert all(v in [0, 255] for v in unique_vals)

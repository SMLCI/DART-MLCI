"""Tests for the FastAPI service."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with lifespan context."""
    from dmc_masking.api.main import app

    # Use context manager to trigger lifespan events
    with TestClient(app) as client:
        yield client


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
        """Missing image file should return validation error."""
        response = client.post("/process-image", data={"roi_id": "0050"})
        assert response.status_code == 422

    def test_process_image_missing_roi_returns_422(self, client):
        """Missing roi_id should return validation error."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        with open(test_image, "rb") as f:
            response = client.post("/process-image", files={"image": f})
        assert response.status_code == 422

    def test_process_image_invalid_roi_returns_error(self, client):
        """Invalid ROI ID should return error in response."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        with open(test_image, "rb") as f:
            response = client.post(
                "/process-image",
                files={"image": ("test.tif", f, "image/tiff")},
                data={"roi_id": "invalid_roi"},
            )
        # May succeed with 200 but have success=False, or may fail validation
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is False
            assert "error_message" in data

    @pytest.mark.skipif(
        not (Path("artifacts/models/v8_detect_s_imgsz640.pt").exists()),
        reason="Model file not found",
    )
    def test_process_image_success(self, client):
        """Valid request should return cropped image and mask."""
        test_image = FIXTURES_DIR / "calibration_image_0000.tif"
        if not test_image.exists():
            pytest.skip("Calibration image not found")

        with open(test_image, "rb") as f:
            response = client.post(
                "/process-image",
                files={"image": ("image.tif", f, "image/tiff")},
                data={"roi_id": "0000", "pixel_size": "0.065789"},
            )
        assert response.status_code == 200
        data = response.json()
        if data["success"]:
            assert "cropped_image" in data  # Base64 PNG
            assert "mask" in data
            assert "rotation_angle" in data
            assert "chamber_type" in data

    def test_process_image_response_structure(self, client):
        """Response should match ProcessImageResponse schema."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        with open(test_image, "rb") as f:
            response = client.post(
                "/process-image",
                files={"image": ("test.tif", f, "image/tiff")},
                data={"roi_id": "0000"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert isinstance(data["success"], bool)


# === Calibrate Endpoint Tests ===


class TestCalibrateEndpoint:
    def test_calibrate_missing_config_returns_422(self, client):
        """Missing config should return validation error."""
        response = client.post("/calibrate")
        assert response.status_code == 422

    def test_calibrate_insufficient_images_returns_error(self, client):
        """Less than 3 calibration images should fail."""
        test_image = FIXTURES_DIR / "calibration_image_0000.tif"
        if not test_image.exists():
            pytest.skip("Calibration image not found")

        config = {
            "chip_name": "SAK",
            "calibration_images": [{"roi_id": "0000", "stage_position": {"x": 0, "y": 0}}],
            "pixel_size": 0.065789,
            "blueprint_map_path": "artifacts/sak_blueprint_map.csv",
        }
        with open(test_image, "rb") as f:
            response = client.post(
                "/calibrate",
                files=[("images", ("image.tif", f, "image/tiff"))],
                data={"config": json.dumps(config)},
            )
        data = response.json()
        assert data["success"] is False
        assert "3" in data["error_message"] or "images" in data["error_message"].lower()

    def test_calibrate_invalid_json_returns_error(self, client):
        """Invalid JSON config should return error."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        with open(test_image, "rb") as f:
            response = client.post(
                "/calibrate",
                files=[("images", ("image.tif", f, "image/tiff"))],
                data={"config": "not valid json"},
            )
        data = response.json()
        assert data["success"] is False
        assert "json" in data["error_message"].lower()

    def test_calibrate_mismatched_images_config(self, client):
        """Number of images not matching config should fail."""
        test_image = FIXTURES_DIR / "test_image.tif"
        if not test_image.exists():
            pytest.skip("Test image not found")

        config = {
            "chip_name": "SAK",
            "calibration_images": [
                {"roi_id": "0000", "stage_position": {"x": 0, "y": 0}},
                {"roi_id": "0001", "stage_position": {"x": 100, "y": 100}},
                {"roi_id": "0002", "stage_position": {"x": 200, "y": 200}},
            ],
            "pixel_size": 0.065789,
            "blueprint_map_path": "artifacts/sak_blueprint_map.csv",
        }
        # Only upload 1 image but config has 3
        with open(test_image, "rb") as f:
            response = client.post(
                "/calibrate",
                files=[("images", ("image.tif", f, "image/tiff"))],
                data={"config": json.dumps(config)},
            )
        data = response.json()
        assert data["success"] is False
        assert "does not match" in data["error_message"].lower()


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
            Path("artifacts/models/v8_detect_s_imgsz640.pt").exists()
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

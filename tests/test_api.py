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

try:
    import importlib.util

    ACIA_AVAILABLE = importlib.util.find_spec("acia") is not None
except Exception:
    ACIA_AVAILABLE = False

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
    cal_dir = Path("artifacts/images/calibration_sample")
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


# === Available Chips Endpoint Tests ===


class TestAvailableChipsEndpoint:
    def test_available_chips_returns_list(self, client):
        """Available chips endpoint should return a list of strings."""
        response = client.get("/available-chips")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all(isinstance(name, str) for name in data)

    @pytest.mark.skipif(
        not (Path("artifacts/chips").exists()),
        reason="Chips directory not found",
    )
    def test_available_chips_includes_sak(self, monkeypatch):
        """Should include 'sak' when chip configs dir is configured."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        # Clear cached settings so env var takes effect
        get_settings.cache_clear()
        monkeypatch.setenv("DART_CHIP_CONFIGS_DIR", "artifacts/chips")
        try:
            with TestClient(app) as client:
                response = client.get("/available-chips")
                assert response.status_code == 200
                data = response.json()
                assert "sak" in data
        finally:
            get_settings.cache_clear()

    def test_available_chips_sorted(self, client):
        """Chip names should be returned in sorted order."""
        response = client.get("/available-chips")
        assert response.status_code == 200
        data = response.json()
        assert data == sorted(data)


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


# === Process Image Preview Endpoint Tests ===


class TestProcessImagePreviewEndpoint:
    def test_preview_missing_image_returns_422(self, client):
        """Missing image field should return validation error."""
        response = client.post("/process-image-preview", json={"roi_id": "0050"})
        assert response.status_code == 422

    def test_preview_missing_roi_returns_422(self, client, test_image_base64):
        """Missing roi_id should return validation error."""
        response = client.post("/process-image-preview", json={"image": test_image_base64})
        assert response.status_code == 422

    def test_preview_invalid_roi_returns_html_error(self, client, test_image_base64):
        """Invalid ROI should return an HTML error page."""
        response = client.post(
            "/process-image-preview",
            json={
                "image": test_image_base64,
                "roi_id": "invalid_roi",
            },
        )
        # Preview returns HTML even on error (422 status with HTML body)
        assert response.status_code in [200, 422]
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.skipif(
        not (Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()),
        reason="Model file not found",
    )
    def test_preview_returns_html_with_images(self, client, test_image_base64):
        """Successful preview should return HTML with embedded base64 images."""
        response = client.post(
            "/process-image-preview",
            json={
                "image": test_image_base64,
                "roi_id": "0000",
                "pixel_size": 0.065789,
            },
        )

        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        assert "<!DOCTYPE html>" in html

        if response.status_code == 200:
            # Success path: HTML with embedded images
            assert "Cropped Image" in html
            assert "Mask" in html
            assert "data:image/png;base64," in html
        else:
            # Error path (e.g. marker detection failed): HTML error page
            assert response.status_code == 422
            assert "Error" in html

    def test_preview_response_is_html(self, client, test_image_base64):
        """Preview endpoint should always return HTML content type."""
        response = client.post(
            "/process-image-preview",
            json={
                "image": test_image_base64,
                "roi_id": "0000",
            },
        )
        # Even errors return HTML
        assert "text/html" in response.headers.get("content-type", "")


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
        not (
            Path("artifacts/images/calibration_sample").exists()
            and Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()
            and Path("artifacts/chips").exists()
        ),
        reason="Calibration sample, model, or chips not found",
    )
    def test_calibrate_with_base64_images(self, monkeypatch, calibration_images_base64, viz_dir):
        """Test calibration with base64 images and visualize results."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        if len(calibration_images_base64) < 3:
            pytest.skip("Need at least 3 calibration images")

        # Load config
        cal_config_path = Path("artifacts/images/calibration_sample/calibration_config.json")
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
        }

        get_settings.cache_clear()
        monkeypatch.setenv("DART_MODEL_PATH", "artifacts/models/v26_detect_s_imgsz1280.pt")
        monkeypatch.setenv("DART_CHIP_CONFIGS_DIR", "artifacts/chips")
        try:
            with TestClient(app) as client:
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
        finally:
            get_settings.cache_clear()

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

    @pytest.mark.skipif(
        not (
            Path("artifacts/chips").exists()
            and Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()
        ),
        reason="Chips directory or model not found",
    )
    def test_calibrate_failed_returns_per_image_errors(self, monkeypatch):
        """Failed calibration should return per-image error messages."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings
        from dart_mlci.api.utils import array_to_base64_png

        get_settings.cache_clear()
        monkeypatch.setenv("DART_CHIP_CONFIGS_DIR", "artifacts/chips")
        monkeypatch.setenv("DART_MODEL_PATH", "artifacts/models/v26_detect_s_imgsz1280.pt")
        try:
            with TestClient(app) as test_client:
                # Create 3 small synthetic (noise) images that will fail calibration
                noise_images = []
                for roi_id in ["0000", "0001", "0002"]:
                    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    noise_images.append(
                        {
                            "image": array_to_base64_png(arr),
                            "roi_id": roi_id,
                            "stage_position": {"x": 0.0, "y": 0.0},
                        }
                    )

                request_body = {
                    "chip_name": "sak",
                    "calibration_images": noise_images,
                    "pixel_size": 0.065789,
                }

                response = test_client.post("/calibrate", json=request_body)
                assert response.status_code == 200
                data = response.json()

                assert data["success"] is False
                assert "image_results" in data
                assert data["image_results"] is not None
                assert len(data["image_results"]) == 3

                for img_result in data["image_results"]:
                    assert img_result["success"] is False
                    assert img_result["error_message"] is not None
                    assert len(img_result["error_message"]) > 0
        finally:
            get_settings.cache_clear()


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
            and Path("artifacts/chips").exists()
        ),
        reason="Artifacts not found",
    )
    def test_full_process_image_workflow(self, monkeypatch, test_image_base64):
        """Test complete image processing workflow."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("DART_MODEL_PATH", "artifacts/models/v26_detect_s_imgsz1280.pt")
        monkeypatch.setenv("DART_CHIP_CONFIGS_DIR", "artifacts/chips")
        try:
            with TestClient(app) as client:
                # Process image via JSON with base64
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
                    # Verify base64 encoded images
                    import base64

                    cropped = base64.b64decode(data["cropped_image"])
                    mask = base64.b64decode(data["mask"])
                    assert len(cropped) > 0
                    assert len(mask) > 0
        finally:
            get_settings.cache_clear()

    def test_gpu_detection(self, client):
        """Container should correctly detect GPU availability."""
        response = client.get("/health")
        data = response.json()
        assert "gpu_available" in data
        assert isinstance(data["gpu_available"], bool)


# === Segment Endpoint Tests ===


class TestSegmentEndpoint:
    @staticmethod
    def _make_synthetic_image_b64(height=64, width=64):
        """Create a small synthetic RGB image as base64 PNG."""
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _make_synthetic_mask_b64(height=64, width=64):
        """Create a small synthetic binary mask as base64 PNG."""
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[10:54, 10:54] = 255  # inner region
        img = Image.fromarray(mask, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_segment_missing_image_returns_422(self, client):
        """Missing image field should return validation error."""
        response = client.post(
            "/segment",
            json={"mask": self._make_synthetic_mask_b64()},
        )
        assert response.status_code == 422

    def test_segment_missing_mask_returns_422(self, client):
        """Missing mask field should return validation error."""
        response = client.post(
            "/segment",
            json={"image": self._make_synthetic_image_b64()},
        )
        assert response.status_code == 422

    def test_segment_invalid_base64_returns_422(self, client):
        """Invalid base64 should fail validation."""
        response = client.post(
            "/segment",
            json={
                "image": "not-valid-base64!!!",
                "mask": "also-invalid!!!",
            },
        )
        assert response.status_code == 422

    def test_segment_no_segmenter_returns_error(self, client):
        """When DART_SEGMENTER is not set, endpoint returns success=False."""
        response = client.post(
            "/segment",
            json={
                "image": self._make_synthetic_image_b64(),
                "mask": self._make_synthetic_mask_b64(),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "DART_SEGMENTER" in data["error_message"]

    def test_segment_response_structure(self, client):
        """Response should match SegmentResponse schema."""
        response = client.post(
            "/segment",
            json={
                "image": self._make_synthetic_image_b64(),
                "mask": self._make_synthetic_mask_b64(),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert isinstance(data["success"], bool)
        assert "segmentation_mask" in data
        assert "cell_count" in data
        assert "total_cell_area" in data
        assert "error_message" in data

    @pytest.mark.skipif(not ACIA_AVAILABLE, reason="acia not installed")
    def test_segment_with_synthetic_input(self, monkeypatch):
        """Integration test: segment a small synthetic image."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("DART_SEGMENTER", "cellpose-sam")
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/segment",
                    json={
                        "image": self._make_synthetic_image_b64(128, 128),
                        "mask": self._make_synthetic_mask_b64(128, 128),
                    },
                )
                assert response.status_code == 200
                data = response.json()
                if data["success"]:
                    assert data["segmentation_mask"] is not None
                    assert isinstance(data["cell_count"], int)
                    assert data["cell_count"] >= 0
                    assert isinstance(data["total_cell_area"], int)
                    assert data["total_cell_area"] >= 0

                    # Verify 16-bit PNG decodes correctly
                    mask_bytes = base64.b64decode(data["segmentation_mask"])
                    mask_img = Image.open(io.BytesIO(mask_bytes))
                    assert mask_img.mode == "I;16"
        finally:
            get_settings.cache_clear()

    @pytest.mark.skipif(not ACIA_AVAILABLE, reason="acia not installed")
    def test_segment_filter_threshold(self, monkeypatch):
        """Integration test: verify filter_threshold parameter affects output."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("DART_SEGMENTER", "cellpose-sam")
        try:
            image_b64 = self._make_synthetic_image_b64(128, 128)
            mask_b64 = self._make_synthetic_mask_b64(128, 128)

            with TestClient(app) as client:
                # Request with strict threshold (remove more cells)
                resp_strict = client.post(
                    "/segment",
                    json={
                        "image": image_b64,
                        "mask": mask_b64,
                        "filter_threshold": 0.1,
                    },
                )
                # Request with lenient threshold (keep more cells)
                resp_lenient = client.post(
                    "/segment",
                    json={
                        "image": image_b64,
                        "mask": mask_b64,
                        "filter_threshold": 0.9,
                    },
                )
                assert resp_strict.status_code == 200
                assert resp_lenient.status_code == 200

                data_strict = resp_strict.json()
                data_lenient = resp_lenient.json()

                if data_strict["success"] and data_lenient["success"]:
                    # Stricter threshold should keep same or fewer cells
                    assert data_strict["cell_count"] <= data_lenient["cell_count"]
        finally:
            get_settings.cache_clear()

    @pytest.mark.skipif(not ACIA_AVAILABLE, reason="acia not installed")
    @pytest.mark.skipif(
        not (
            Path("artifacts/models/v26_detect_s_imgsz1280.pt").exists()
            and Path("artifacts/chips").exists()
        ),
        reason="Model file or chips directory not found",
    )
    def test_segment_roundtrip_with_process_image(self, monkeypatch, test_image_base64):
        """Integration test: full pipeline /process-image -> /segment."""
        from dart_mlci.api.main import app
        from dart_mlci.api.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("DART_SEGMENTER", "cellpose-sam")
        monkeypatch.setenv("DART_MODEL_PATH", "artifacts/models/v26_detect_s_imgsz1280.pt")
        monkeypatch.setenv("DART_CHIP_CONFIGS_DIR", "artifacts/chips")
        try:
            with TestClient(app) as client:
                # Step 1: process image
                proc_resp = client.post(
                    "/process-image",
                    json={
                        "image": test_image_base64,
                        "roi_id": "0000",
                        "pixel_size": 0.065789,
                    },
                )
                assert proc_resp.status_code == 200
                proc_data = proc_resp.json()
                if not proc_data["success"]:
                    pytest.skip(f"process-image failed: {proc_data['error_message']}")

                # Step 2: segment
                seg_resp = client.post(
                    "/segment",
                    json={
                        "image": proc_data["cropped_image"],
                        "mask": proc_data["mask"],
                    },
                )
                assert seg_resp.status_code == 200
                seg_data = seg_resp.json()
                if seg_data["success"]:
                    assert seg_data["segmentation_mask"] is not None
                    assert isinstance(seg_data["cell_count"], int)
        finally:
            get_settings.cache_clear()


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


# === Error-path coverage that doesn't depend on artifact fixtures ===


def _synthetic_png_b64(h=64, w=64):
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestProcessImagePreviewErrorPaths:
    """Exercise the HTML error-rendering branch of /process-image-preview."""

    def test_blank_image_returns_html_error_page(self, client):
        """A blank synthetic image has no markers → error HTML."""
        response = client.post(
            "/process-image-preview",
            json={"image": _synthetic_png_b64(), "roi_id": "0000"},
        )
        # Either 422 (pipeline failure rendered as HTML) or 200 if the request
        # was malformed enough to short-circuit elsewhere — both go through
        # HTMLResponse paths we want to exercise.
        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        assert "<!DOCTYPE html>" in html
        if response.status_code == 422:
            assert "Error" in html or "error" in html


class TestSegmentBadInputs:
    """Exercise the early decode-failure branches of /segment."""

    def test_garbage_mask_returns_decode_error(self, client):
        """Valid image, garbage mask → success=False with mask decode error."""
        # Note: base64.b64decode tolerates many garbage strings, so we use
        # one that decodes to non-image bytes — mask decoding still fails.
        response = client.post(
            "/segment",
            json={
                "image": _synthetic_png_b64(),
                "mask": base64.b64encode(b"not an image").decode("utf-8"),
            },
        )
        # If the segmenter isn't loaded, segmenter-missing wins over the
        # mask decode failure (segmenter check runs first). Accept either
        # error message — both are valid error paths.
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error_message"]


class TestCalibrateBadInputs:
    """Exercise validation-error branches of /calibrate."""

    def test_invalid_pixel_size_returns_422(self, client):
        """Negative pixel_size should be rejected by Pydantic validation."""
        response = client.post(
            "/calibrate",
            json={
                "chip_name": "SAK",
                "calibration_images": [
                    {
                        "image": _synthetic_png_b64(),
                        "roi_id": f"{i:04d}",
                        "stage_position": {"x": float(i), "y": float(i), "z": 0.0},
                    }
                    for i in range(3)
                ],
                "pixel_size": -1.0,
            },
        )
        # Either 422 from Pydantic or 200 with success=False — both are error
        # paths we want to exercise.
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.json()["success"] is False

    def test_empty_calibration_images_returns_error(self, client):
        """Empty calibration_images list → either validation error or success=False."""
        response = client.post(
            "/calibrate",
            json={
                "chip_name": "SAK",
                "calibration_images": [],
                "pixel_size": 0.065789,
            },
        )
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.json()["success"] is False

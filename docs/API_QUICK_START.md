# DMC Masking API - Quick Start Guide

## Start the API Server

```bash
conda activate dmc-masking-claude
uvicorn dmc_masking.api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

## Quick Examples

### 1. Process a Single Image

```python
import base64
import requests
from pathlib import Path

# Load and encode image
image_path = Path("my_image.tif")
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send request
response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "image": image_b64,
        "roi_id": "0050",
        "pixel_size": 0.065789
    }
)

# Handle response
result = response.json()
if result["success"]:
    print(f"Chamber type: {result['chamber_type']}")
    print(f"Rotation angle: {result['rotation_angle']:.2f}°")

    # Save outputs
    with open("cropped.png", "wb") as f:
        f.write(base64.b64decode(result["cropped_image"]))
    with open("mask.png", "wb") as f:
        f.write(base64.b64decode(result["mask"]))
else:
    print(f"Error: {result['error_message']}")
```

### 2. Calibrate from Images

```python
import base64
import requests
from pathlib import Path

def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Prepare calibration request
request_data = {
    "chip_name": "SAK",
    "calibration_images": [
        {
            "image": load_image_b64("cal_0000.tif"),
            "roi_id": "0000",
            "stage_position": {"x": 0.0, "y": 0.0}
        },
        {
            "image": load_image_b64("cal_7000.tif"),
            "roi_id": "7000",
            "stage_position": {"x": 461.25, "y": 0.0}
        },
        {
            "image": load_image_b64("cal_7315.tif"),
            "roi_id": "7315",
            "stage_position": {"x": 480.94, "y": 20.69}
        }
    ],
    "pixel_size": 0.065789,
    "blueprint_map_path": "artifacts/sak_blueprint_map.csv"
}

# Send request
response = requests.post(
    "http://localhost:8000/calibrate",
    json=request_data
)

# Save results
result = response.json()
if result["success"]:
    import json
    with open("calibrated_map.json", "w") as f:
        json.dump(result["calibrated_map"], f, indent=2)

    stats = result["statistics"]
    print(f"Calibration complete:")
    print(f"  RMSE: {stats['rmse']:.4f}")
    print(f"  Max error: {stats['max_error']:.4f}")
    print(f"  Points: {stats['n_points']}")
else:
    print(f"Error: {result['error_message']}")
```

### 3. Get HTML Preview

```python
import base64
import requests
from pathlib import Path

# Load image
with open("my_image.tif", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Request preview
response = requests.post(
    "http://localhost:8000/process-image-preview",
    json={
        "image": image_b64,
        "roi_id": "0050",
        "pixel_size": 0.065789
    }
)

# Save HTML
with open("preview.html", "w") as f:
    f.write(response.text)

print("Open preview.html in your browser")
```

## Helper Function

```python
def process_image_file(image_path: str, roi_id: str, api_url: str = "http://localhost:8000"):
    """Process an image file and return results."""
    import base64
    import requests
    from pathlib import Path

    # Load and encode
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Request
    response = requests.post(
        f"{api_url}/process-image",
        json={"image": image_b64, "roi_id": roi_id}
    )

    return response.json()

# Usage
result = process_image_file("image.tif", "0050")
if result["success"]:
    print(f"Success! Angle: {result['rotation_angle']:.2f}°")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check service health |
| `/available-chips` | GET | List loaded chip configurations |
| `/chamber-types` | GET | List available chamber types |
| `/process-image` | POST | Process single image (JSON) |
| `/process-image-preview` | POST | Get HTML preview (JSON) |
| `/calibrate` | POST | Calibrate map (JSON) |
| `/docs` | GET | Interactive API documentation |

### Chip Selection

When multiple chip configs are loaded (via `DMC_CHIP_CONFIGS_DIR`), specify
`chip_name` in your request:

```python
response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "image": image_b64,
        "roi_id": "0050",
        "chip_name": "sak",
    }
)
```

List available chips:

```python
chips = requests.get("http://localhost:8000/available-chips").json()
print(chips)  # ["sak"]
```

## Run Tests

```bash
# Run all tests
conda run -n dmc-masking-claude pytest tests/test_api.py -v

# Run specific test
conda run -n dmc-masking-claude pytest tests/test_api.py::TestProcessImageEndpoint -v

# Run with coverage
conda run -n dmc-masking-claude pytest tests/test_api.py --cov=dmc_masking.api
```

## Troubleshooting

### Import Error
```bash
# Verify environment
conda activate dmc-masking-claude
python -c "from dmc_masking.api.main import app; print('OK')"
```

### Port Already in Use
```bash
# Use different port
uvicorn dmc_masking.api.main:app --port 8888
```

### Invalid Base64
```python
# The API auto-strips data URI prefixes
image_b64 = "data:image/tiff;base64,iVBORw..."  # Works!
image_b64 = "iVBORw..."  # Also works!
```

## More Information

- **Migration Guide**: `docs/API_BASE64_MIGRATION.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Interactive Docs**: http://localhost:8000/docs (when server is running)

# API Base64 Migration Guide

## Overview

The DMC Masking API has been updated to accept JSON requests with base64-encoded images instead of multipart file uploads. This provides a cleaner API design with consistent input/output formats.

**Breaking Change:** Existing clients using multipart/form-data uploads will need to migrate to JSON requests.

## New Features

- **JSON-based requests**: All endpoints now accept JSON with base64-encoded images
- **Data URI support**: Automatically strips `data:image/...;base64,` prefixes
- **Consistent format**: Both input and output use base64-encoded images
- **Better testability**: Easier to write and debug API tests

## Migration Examples

### Process Image Endpoint

#### Before (Multipart Upload)

```python
import requests

with open("image.tif", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process-image",
        files={"image": ("image.tif", f, "image/tiff")},
        data={
            "roi_id": "0050",
            "pixel_size": "0.065789"
        }
    )
```

#### After (JSON with Base64)

```python
import base64
import requests

# Load and encode image
with open("image.tif", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

# Send JSON request
response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "image": b64_image,
        "roi_id": "0050",
        "pixel_size": 0.065789
    }
)

result = response.json()
if result["success"]:
    # Decode outputs
    cropped = base64.b64decode(result["cropped_image"])
    mask = base64.b64decode(result["mask"])

    # Save to files
    with open("cropped.png", "wb") as f:
        f.write(cropped)
    with open("mask.png", "wb") as f:
        f.write(mask)
```

### Calibrate Endpoint

#### Before (Multipart Upload)

```python
import json
import requests

config = {
    "chip_name": "SAK",
    "calibration_images": [
        {"roi_id": "0000", "stage_position": {"x": 0, "y": 0}},
        {"roi_id": "7000", "stage_position": {"x": 100, "y": 100}},
        {"roi_id": "7315", "stage_position": {"x": 200, "y": 200}}
    ],
    "pixel_size": 0.065789,
    "blueprint_map_path": "artifacts/sak_blueprint_map.csv"
}

files = [
    ("images", open("0000.tif", "rb")),
    ("images", open("7000.tif", "rb")),
    ("images", open("7315.tif", "rb"))
]

response = requests.post(
    "http://localhost:8000/calibrate",
    files=files,
    data={"config": json.dumps(config)}
)
```

#### After (JSON with Base64)

```python
import base64
import requests

# Load and encode images
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Build request
request_body = {
    "chip_name": "SAK",
    "calibration_images": [
        {
            "image": encode_image("0000.tif"),
            "roi_id": "0000",
            "stage_position": {"x": 0, "y": 0}
        },
        {
            "image": encode_image("7000.tif"),
            "roi_id": "7000",
            "stage_position": {"x": 100, "y": 100}
        },
        {
            "image": encode_image("7315.tif"),
            "roi_id": "7315",
            "stage_position": {"x": 200, "y": 200}
        }
    ],
    "pixel_size": 0.065789,
    "blueprint_map_path": "artifacts/sak_blueprint_map.csv"
}

# Send request
response = requests.post(
    "http://localhost:8000/calibrate",
    json=request_body
)

result = response.json()
if result["success"]:
    # Save calibrated map
    import json
    with open("calibrated_map.json", "w") as f:
        json.dump(result["calibrated_map"], f, indent=2)

    # Print statistics
    stats = result["statistics"]
    print(f"RMSE: {stats['rmse']:.4f}")
    print(f"Max Error: {stats['max_error']:.4f}")
```

## Data URI Support

The API automatically handles data URIs from browser-based applications:

```python
# This works (with data URI prefix)
data_uri = "data:image/tiff;base64,iVBORw0KGgoAAAANS..."

# This also works (plain base64)
plain_b64 = "iVBORw0KGgoAAAANS..."

response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "image": data_uri,  # or plain_b64
        "roi_id": "0050"
    }
)
```

## Error Handling

The new API provides better error messages for common issues:

```python
# Invalid base64
response = requests.post(
    "http://localhost:8000/process-image",
    json={"image": "not-valid-base64!!!", "roi_id": "0050"}
)
# Returns: 422 Unprocessable Entity
# {"detail": [{"msg": "Invalid base64: ..."}]}

# Corrupted image data
response = requests.post(
    "http://localhost:8000/process-image",
    json={"image": "YWJjZGVm", "roi_id": "0050"}
)
# Returns: 200 OK
# {"success": false, "error_message": "Failed to decode base64 image: ..."}

# Invalid ROI ID
response = requests.post(
    "http://localhost:8000/process-image",
    json={"image": valid_b64, "roi_id": "invalid"}
)
# Returns: 200 OK
# {"success": false, "error_message": "Invalid ROI ID 'invalid': ..."}
```

## Testing

Run the test suite to verify the API works correctly:

```bash
conda run -n dmc-masking-claude pytest tests/test_api.py -v
```

Test visualizations are saved to `tests/test_output/api_visualizations/` for manual inspection.

## Browser Usage (JavaScript)

```javascript
// Convert File to base64
async function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Process image
async function processImage(file, roiId) {
    const imageBase64 = await fileToBase64(file);

    const response = await fetch('http://localhost:8000/process-image', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: imageBase64,
            roi_id: roiId,
            pixel_size: 0.065789
        })
    });

    const result = await response.json();

    if (result.success) {
        // Display cropped image
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.cropped_image}`;
        document.body.appendChild(img);
    }
}
```

## Performance Considerations

- Base64 encoding increases payload size by ~33% compared to raw binary
- For large batches, consider compression or streaming (future enhancement)
- The API server handles base64 decoding efficiently using optimized libraries

## Backward Compatibility

The old multipart/form-data endpoints have been removed. All clients must migrate to the new JSON API.

If you need to support legacy clients, you can deploy the previous API version alongside the new one on different ports or paths.

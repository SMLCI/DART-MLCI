# DART API -- Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd dmc-masking

# Install the package with API dependencies
pip install -e ".[api]"
```

Or with conda:

```bash
conda create -n dart python=3.10
conda activate dart
pip install -e ".[api]"
```

## Start the API Server

### Local

```bash
uvicorn dart_mlci.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up -d
```

Or build and run manually:

```bash
docker build -t dart-mlci .
docker run -d -p 8000:8000 --name dart-api dart-mlci
```

Visit http://localhost:8000/docs for interactive Swagger UI documentation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check service health and loaded resources |
| `/available-chips` | GET | List loaded chip configurations |
| `/chamber-types` | GET | List available chamber types |
| `/process-image` | POST | Process a single image (JSON with base64) |
| `/process-image-preview` | POST | Get HTML preview of processing result |
| `/calibrate` | POST | Calibrate microscope map from images |
| `/docs` | GET | Interactive Swagger API documentation |

---

## Python Client Examples

### Setup

```bash
pip install requests
```

### Health Check

```python
import requests

response = requests.get("http://localhost:8000/health")
health = response.json()
print(f"Status: {health['status']}")
print(f"Model loaded: {health['model_loaded']}")
print(f"Device: {health['device']}")
```

### Image Loading

The API accepts raw image bytes encoded as base64. The server handles
normalization internally (quantile-based for uint16 TIFFs, same as
`dart_mlci.io.load_image()`). Simply read the file bytes and encode:

```python
import base64

def load_image_b64(path):
    """Load an image file as a base64 string for the API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

> **16-bit images:** The API fully supports uint16 TIFFs (common in microscopy).
> The server applies quantile-based normalization (1st–99th percentile) to
> preserve contrast. Just send the raw file bytes — no client-side conversion needed.

> **Note:** If you have `dart_mlci` installed locally and want to inspect or
> preprocess images before sending, use `dart_mlci.io.load_image()` which
> returns a normalized HxWx3 uint8 array.

### Process a Single Image

```python
import base64
import requests
from pathlib import Path

# Load and encode image (raw file bytes — server normalizes automatically)
image_path = Path("my_image.tif")
image_b64 = load_image_b64(image_path)

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
    print(f"Rotation angle: {result['rotation_angle']:.2f}")

    # Save outputs
    with open("cropped.png", "wb") as f:
        f.write(base64.b64decode(result["cropped_image"]))
    with open("mask.png", "wb") as f:
        f.write(base64.b64decode(result["mask"]))
else:
    print(f"Error: {result['error_message']}")
```

### Calibrate from Images

```python
import requests

# Prepare calibration request (using load_image_b64 from above)
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
    "pixel_size": 0.065789
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

### Get HTML Preview

```python
import base64
import requests

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

### Chip Selection (Multi-Chip)

When multiple chip configs are loaded (via `DART_CHIP_CONFIGS_DIR`), specify
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

### Helper Function

```python
def process_image_file(image_path: str, roi_id: str, api_url: str = "http://localhost:8000"):
    """Process an image file and return results."""
    import base64
    import requests

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"{api_url}/process-image",
        json={"image": image_b64, "roi_id": roi_id}
    )
    return response.json()

# Usage
result = process_image_file("image.tif", "0050")
if result["success"]:
    print(f"Success! Angle: {result['rotation_angle']:.2f}")
```

---

## Java Client Examples (Java 11+)

These examples use `java.net.http.HttpClient` (built-in since Java 11) and
`org.json` for JSON parsing. No framework dependencies required.

> **Image loading:** Simply read the raw file bytes and Base64-encode them.
> The server handles normalization (quantile-based for uint16 TIFFs).

### Setup

Add `org.json` to your project (Maven):

```xml
<dependency>
    <groupId>org.json</groupId>
    <artifactId>json</artifactId>
    <version>20240303</version>
</dependency>
```

Or Gradle:

```groovy
implementation 'org.json:json:20240303'
```

> **Large requests:** Microscopy images encoded as base64 can produce requests
> of 50 MB+. The API server has no body size limit. However:
> - If behind **nginx**, set `client_max_body_size 100m;` (default is 1 MB)
> - In **Java**, increase the JVM heap if needed: `java -Xmx512m ...`
>   (a single base64-encoded TIFF can be 50 MB+; calibration sends 3+ images)

### Health Check

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import org.json.JSONObject;

public class DartHealthCheck {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:8000/health"))
                .GET()
                .build();

        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString());

        JSONObject health = new JSONObject(response.body());
        System.out.println("Status: " + health.getString("status"));
        System.out.println("Model loaded: " + health.getBoolean("model_loaded"));
        System.out.println("Device: " + health.getString("device"));
    }
}
```

### Process a Single Image

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import org.json.JSONObject;

public class DartProcessImage {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        // Load and encode image
        byte[] imageBytes = Files.readAllBytes(Path.of("my_image.tif"));
        String imageB64 = Base64.getEncoder().encodeToString(imageBytes);

        // Build JSON request
        JSONObject requestBody = new JSONObject();
        requestBody.put("image", imageB64);
        requestBody.put("roi_id", "0050");
        requestBody.put("pixel_size", 0.065789);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:8000/process-image"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
                .build();

        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString());

        JSONObject result = new JSONObject(response.body());

        if (result.getBoolean("success")) {
            System.out.println("Chamber type: " + result.getString("chamber_type"));
            System.out.println("Rotation angle: " + result.getDouble("rotation_angle"));

            // Save cropped image
            byte[] croppedBytes = Base64.getDecoder().decode(
                    result.getString("cropped_image"));
            Files.write(Path.of("cropped.png"), croppedBytes);

            // Save mask
            byte[] maskBytes = Base64.getDecoder().decode(
                    result.getString("mask"));
            Files.write(Path.of("mask.png"), maskBytes);

            System.out.println("Saved cropped.png and mask.png");
        } else {
            System.err.println("Error: " + result.getString("error_message"));
        }
    }
}
```

### Calibrate from Images

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import org.json.JSONArray;
import org.json.JSONObject;

public class DartCalibrate {
    static String loadImageB64(String path) throws Exception {
        byte[] bytes = Files.readAllBytes(Path.of(path));
        return Base64.getEncoder().encodeToString(bytes);
    }

    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();

        // Build calibration images array
        JSONArray calibrationImages = new JSONArray();

        calibrationImages.put(new JSONObject()
                .put("image", loadImageB64("cal_0000.tif"))
                .put("roi_id", "0000")
                .put("stage_position", new JSONObject()
                        .put("x", 0.0).put("y", 0.0)));

        calibrationImages.put(new JSONObject()
                .put("image", loadImageB64("cal_7000.tif"))
                .put("roi_id", "7000")
                .put("stage_position", new JSONObject()
                        .put("x", 461.25).put("y", 0.0)));

        calibrationImages.put(new JSONObject()
                .put("image", loadImageB64("cal_7315.tif"))
                .put("roi_id", "7315")
                .put("stage_position", new JSONObject()
                        .put("x", 480.94).put("y", 20.69)));

        // Build request body
        JSONObject requestBody = new JSONObject();
        requestBody.put("chip_name", "SAK");
        requestBody.put("calibration_images", calibrationImages);
        requestBody.put("pixel_size", 0.065789);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:8000/calibrate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
                .build();

        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString());

        JSONObject result = new JSONObject(response.body());

        if (result.getBoolean("success")) {
            // Save calibrated map
            JSONArray calibratedMap = result.getJSONArray("calibrated_map");
            Files.writeString(
                    Path.of("calibrated_map.json"),
                    calibratedMap.toString(2));

            JSONObject stats = result.getJSONObject("statistics");
            System.out.printf("Calibration complete:%n");
            System.out.printf("  RMSE: %.4f%n", stats.getDouble("rmse"));
            System.out.printf("  Max error: %.4f%n", stats.getDouble("max_error"));
            System.out.printf("  Points: %d%n", stats.getInt("n_points"));
        } else {
            System.err.println("Error: " + result.getString("error_message"));
        }
    }
}
```

### Helper Class

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import org.json.JSONObject;

public class DartClient {
    private final HttpClient client;
    private final String baseUrl;

    public DartClient(String baseUrl) {
        this.client = HttpClient.newHttpClient();
        this.baseUrl = baseUrl;
    }

    public DartClient() {
        this("http://localhost:8000");
    }

    public JSONObject processImage(String imagePath, String roiId) throws Exception {
        byte[] imageBytes = Files.readAllBytes(Path.of(imagePath));
        String imageB64 = Base64.getEncoder().encodeToString(imageBytes);

        JSONObject body = new JSONObject();
        body.put("image", imageB64);
        body.put("roi_id", roiId);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/process-image"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                .build();

        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString());

        return new JSONObject(response.body());
    }

    // Usage
    public static void main(String[] args) throws Exception {
        DartClient dart = new DartClient();
        JSONObject result = dart.processImage("image.tif", "0050");

        if (result.getBoolean("success")) {
            System.out.printf("Success! Angle: %.2f%n",
                    result.getDouble("rotation_angle"));
        }
    }
}
```

---

## Troubleshooting

### Import Error
```bash
# Verify environment
conda activate dmc-masking-claude
python -c "from dart_mlci.api.main import app; print('OK')"
```

### Port Already in Use
```bash
# Use different port
uvicorn dart_mlci.api.main:app --port 8888
```

### Invalid Base64
```python
# The API auto-strips data URI prefixes
image_b64 = "data:image/tiff;base64,iVBORw..."  # Works!
image_b64 = "iVBORw..."  # Also works!
```

### Docker Container Won't Start
```bash
# Check logs
docker logs dart-api

# Verify the image built correctly
docker run --rm dart-mlci python -c "from dart_mlci.api.main import app; print('OK')"
```

## More Information

- **Interactive Docs**: http://localhost:8000/docs (Swagger UI, when server is running)
- **Migration Guide**: `docs/API_BASE64_MIGRATION.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`

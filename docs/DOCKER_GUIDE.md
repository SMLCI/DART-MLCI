# Docker Guide for DMC Masking API

## Quick Start with Docker Compose

The easiest way to run the API in Docker:

```bash
# Build and start the API
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the API
docker-compose down
```

The API will be available at http://localhost:8000

## Manual Docker Commands

### Build the Image

```bash
# Build the Docker image
docker build -t dmc-masking:latest .

# Build with a specific tag
docker build -t dmc-masking:v1.0 .
```

### Run the Container

#### CPU Mode (Default)
```bash
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  -e DMC_MODEL_PATH=/app/artifacts/models/v26_detect_s_imgsz1280.pt \
  -e DMC_STRUCTURE_LIBRARY_PATH=/app/artifacts/chamber_structure.json \
  dmc-masking:latest
```

#### GPU Mode (Requires nvidia-docker)
```bash
docker run -d \
  --name dmc-masking-api \
  --gpus all \
  -p 8000:8000 \
  -e DMC_DEVICE=cuda:0 \
  dmc-masking:latest
```

#### Custom Port
```bash
docker run -d \
  --name dmc-masking-api \
  -p 8888:8000 \
  dmc-masking:latest
```

### Container Management

```bash
# View logs
docker logs -f dmc-masking-api

# Stop the container
docker stop dmc-masking-api

# Start the container
docker start dmc-masking-api

# Restart the container
docker restart dmc-masking-api

# Remove the container
docker rm -f dmc-masking-api

# Execute commands inside container
docker exec -it dmc-masking-api bash

# Check container health
docker inspect --format='{{.State.Health.Status}}' dmc-masking-api
```

## Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `DMC_MODEL_PATH` | `/app/artifacts/models/v26_detect_s_imgsz1280.pt` | Path to YOLO model |
| `DMC_STRUCTURE_LIBRARY_PATH` | `/app/artifacts/chamber_structure.json` | Path to structure library |
| `DMC_BLUEPRINT_MAP_PATH` | `/app/artifacts/sak_blueprint_map.csv` | Path to blueprint map |
| `DMC_CHIP_CONFIGS_DIR` | `/app/artifacts/chips/` | Directory of chip config JSONs (multi-chip support) |
| `DMC_CHIP_CONFIG_PATH` | Not set | Single chip config path (overrides directory scan) |
| `DMC_PIXEL_SIZE` | `0.065789` | Default pixel size in microns |
| `DMC_DEVICE` | Auto-detected | Device to use (`cpu` or `cuda:0`) |

### Multi-Chip Support

Place chip config JSON files in the `DMC_CHIP_CONFIGS_DIR` directory. Each file's
stem becomes the chip name (e.g., `sak.json` → chip name `sak`). Clients select
a chip via the `chip_name` field in API requests. List loaded chips at
`GET /available-chips`.

Example with custom environment variables:

```bash
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  -e DMC_PIXEL_SIZE=0.05 \
  -e DMC_DEVICE=cpu \
  dmc-masking:latest
```

## Volume Mounts

### Custom Model or Structure Library

```bash
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  -v /path/to/custom/model.pt:/app/custom_model.pt \
  -e DMC_MODEL_PATH=/app/custom_model.pt \
  dmc-masking:latest
```

### Persistent Logs

```bash
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  -v /path/to/logs:/app/logs \
  dmc-masking:latest
```

## Testing the Dockerized API

### Test from Host Machine

```bash
# Health check
curl http://localhost:8000/health

# Process an image (base64 encoded)
python3 << 'EOF'
import base64
import requests

# Load and encode test image
with open("tests/fixtures/calibration_image_0000.tif", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send request to Docker container
response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "image": image_b64,
        "roi_id": "0000",
        "pixel_size": 0.065789
    }
)

result = response.json()
print(f"Success: {result['success']}")
if result['success']:
    print(f"Chamber type: {result['chamber_type']}")
    print(f"Rotation angle: {result['rotation_angle']:.2f}°")
EOF
```

### Test from Inside Container

```bash
# Enter the container
docker exec -it dmc-masking-api bash

# Inside container - test health
curl http://localhost:8000/health

# Exit container
exit
```

## GPU Support

### Prerequisites

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Verify GPU is available:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### Run with GPU

Using Docker Compose (uncomment GPU section in `docker-compose.yml`):

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then:
```bash
docker-compose up --build
```

Or using Docker CLI:
```bash
docker run -d \
  --name dmc-masking-api \
  --gpus all \
  -p 8000:8000 \
  -e DMC_DEVICE=cuda:0 \
  dmc-masking:latest
```

### Verify GPU Usage

```bash
# Check if GPU is detected
curl http://localhost:8000/health | jq '.gpu_available'

# Should return: true

# Monitor GPU usage
nvidia-smi -l 1
```

## Multi-Architecture Builds

Build for multiple platforms (ARM64, AMD64):

```bash
# Create builder
docker buildx create --name multiarch --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t dmc-masking:latest \
  --push \
  .
```

## Troubleshooting

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Use a different port
docker run -d -p 8888:8000 dmc-masking:latest
```

### Container Won't Start

```bash
# Check logs
docker logs dmc-masking-api

# Common issues:
# - Model file not found: Check DMC_MODEL_PATH
# - Permission issues: Check file permissions in artifacts/
# - Port conflict: Use different port with -p flag
```

### Model Not Loading

```bash
# Verify model file exists in container
docker exec dmc-masking-api ls -lh /app/artifacts/models/

# Check environment variables
docker exec dmc-masking-api env | grep DMC_

# Test health endpoint
curl http://localhost:8000/health
```

### Out of Memory

```bash
# Limit container memory
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  --memory=4g \
  --memory-swap=4g \
  dmc-masking:latest
```

### Rebuild Without Cache

```bash
# Force complete rebuild
docker build --no-cache -t dmc-masking:latest .

# Or with docker-compose
docker-compose build --no-cache
```

## Production Deployment

### Using Docker Compose (Recommended)

```bash
# Production docker-compose.yml
version: '3.8'

services:
  dmc-masking-api:
    image: dmc-masking:latest
    container_name: dmc-masking-api
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DMC_MODEL_PATH=/app/artifacts/models/v26_detect_s_imgsz1280.pt
      - DMC_STRUCTURE_LIBRARY_PATH=/app/artifacts/chamber_structure.json
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Behind Nginx Reverse Proxy

```nginx
upstream dmc_api {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://dmc_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for large base64 payloads
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### Resource Limits

```bash
docker run -d \
  --name dmc-masking-api \
  -p 8000:8000 \
  --memory=4g \
  --memory-swap=4g \
  --cpus=2 \
  --restart=unless-stopped \
  dmc-masking:latest
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: dmc-masking:latest
```

## Additional Resources

- **Interactive API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Migration Guide**: `docs/API_BASE64_MIGRATION.md`
- **Quick Start**: `docs/API_QUICK_START.md`

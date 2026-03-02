# dmc-masking

Real-time microfluidic chamber image processing library with two core capabilities:

1. **Masking pipeline** — detect YOLO markers, match pairs, correct rotation, apply polygon mask, and crop the region of interest.
2. **Map calibration** — align a chip blueprint with microscope stage coordinates via affine transform so every chamber can be revisited automatically.

## Installation

```bash
# Core library
pip install .

# With REST API support
pip install ".[api]"

# Development (tests, linting)
pip install ".[dev]"
```

## Quick Start

### Python

```python
from dmc_masking import MarkerDetectionModel, ChipStructureLibrary

# Load chip config and detection model
lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json")
model = MarkerDetectionModel()

# Detect markers
markers = model.predict_markers(image)
```

### Pipeline steps

```python
from dmc_masking import (
    MarkerDetectionStep,
    MarkerMatchingStep,
    ImageRotationStep,
    RoIMaskingStep,
)

# Build pipeline
detect = MarkerDetectionStep()
match  = MarkerMatchingStep(marker_group_pixel=mgp)
rotate = ImageRotationStep()
mask   = RoIMaskingStep(marker_group_pixels=mgp, roi_polygon=polygon)

# Run
data = detect(image)
data = match(data)
data = rotate(data)
data = mask(data)

cropped_image = data["image"]
cropped_mask  = data["mask"]
```

### CLI — calibrate a map

```bash
python scripts/calibrate_map.py --config calibration.json
```

### CLI — process images

```bash
python scripts/process_image.py --image image.tif --roi-id 0050
```

### REST API

```bash
uvicorn dmc_masking.api.main:app --host 0.0.0.0 --port 8000
# Interactive docs at http://localhost:8000/docs
```

```python
import base64, requests

with open("image.tif", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8000/process-image",
    json={"image": b64, "roi_id": "0050"},
)
print(resp.json()["chamber_type"])
```

### Docker

```bash
docker-compose up --build
# API available at http://localhost:8000
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture overview](docs/index.md) | Module map, pipeline diagrams |
| [API Quick Start](docs/API_QUICK_START.md) | REST API usage examples |
| [Docker Guide](docs/DOCKER_GUIDE.md) | Container deployment |
| [API Migration](docs/API_BASE64_MIGRATION.md) | Base64 encoding migration notes |
| [Contributing](CONTRIBUTING.md) | Dev setup, tests, adding chip designs |
| [Changelog](CHANGELOG.md) | Release history |

## Project Structure

```
dmc_masking/
  __init__.py          # Public API re-exports
  constants.py         # DEFAULT_MODEL_PATH, pixel sizes, tolerances
  detection.py         # MarkerDetectionModel, extract_data
  masker.py            # RoIMasker, SingleStructureRoIMasker
  pipeline.py          # Step classes (detect → match → rotate → mask)
  mask.py              # RoIPolygon, apply_mask
  map.py               # Map, calibration transforms
  chip.py              # ChipStructureLibrary (unified chip config)
  config.py            # DMCConfig dataclasses
  io.py                # Image / structure file loading
  rotation.py          # Image and marker rotation
  match.py             # Marker pair matching
  registration.py      # Phase-correlation & timelapse registration
  utils.py             # normalize_image, helpers
  visualization/       # Plotting, OpenCV drawing, video generation
  api/                 # FastAPI REST endpoints
artifacts/
  models/              # YOLO weights
  chips/               # Chip config JSONs (sak.json, …)
scripts/               # CLI tools (calibrate_map, process_image, …)
tests/                 # Pytest suite
```

## License

[MIT](LICENSE) — Copyright (c) 2025 Johannes Seiffarth, Forschungszentrum Juelich GmbH

## Citation

If you use dmc-masking in your research, please cite:

```
@software{dmc_masking,
  author  = {Seiffarth, Johannes},
  title   = {dmc-masking: Real-time microfluidic chamber image processing},
  year    = {2025},
  url     = {https://github.com/JojoDevel/dmc-masking}
}
```

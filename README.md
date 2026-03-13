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

## Reproducing the DART Experiment

Step-by-step instructions to reproduce the automated analysis of all seven DART chamber types.

### Quick Validation

To verify that all steps work end-to-end, use the automated script:

```bash
bash reproduce.sh            # smoke test (1 stack per folder)
bash reproduce.sh --full     # full experiment (all stacks)
```

The script installs dependencies, downloads the dataset, runs processing into
`dart_experiment/output_reproduce/`, and validates that expected outputs exist.

### Prerequisites

- Python >= 3.10
- GPU with CUDA support (recommended for segmentation)

### Setup

```bash
pip install .
pip install acia    # segmentation library (Cellpose-SAM)
```

### Download Dataset

```bash
wget -O DART_Experiment.zip "https://fz-juelich.sciebo.de/s/Tq5SW76WG9zqMJi/download"
unzip DART_Experiment.zip -d dart_experiment/
```

### Expected Data Structure

After downloading and unzipping, the directory should look like:

```
dart_experiment/
├── folder_config.json                # Omnipose config (with registration)
├── folder_config_cellpose_sam.json   # Cellpose-SAM config (default)
└── DART_Experiment/
    ├── Small Chambers/               # .tif stacks
    ├── Big Chambers/
    ├── Big Chambers + Pillars/
    ├── Open Chambers/
    ├── Open Chambers + Structures/
    ├── Mother Machines/
    └── Small Chambers + Pillar/
```

Each subfolder contains one or more time-lapse TIFF stacks (T x H x W, uint16).

### Run Analysis

The default configuration uses **Cellpose-SAM** for segmentation with registration disabled:

```bash
python scripts/process_folder.py \
    --config dart_experiment/folder_config_cellpose_sam.json \
    --save-cropped \
    --render-stacks \
    --verbose
```

Key flags:

| Flag | Description |
|------|-------------|
| `--save-cropped` | Save cropped chamber images as `stack_cropped.tif` |
| `--render-stacks` | Generate per-frame PNGs and an MP4 timelapse video |
| `--verbose` | Print detailed per-frame progress |
| `--max-files N` | Process only the first N stacks (useful for testing) |
| `--skip-existing` | Skip stacks that already have output |

### Configuration

Both config files use the same JSON format. Key fields:

| Field | Description |
|-------|-------------|
| `input_dir` | Path to the unzipped dataset |
| `output_dir` | Where results are written |
| `pixel_size` | Microscope pixel size in µm/px |
| `chip_config` | Path to the SAK chip geometry JSON |
| `model_path` | Path to the YOLO marker detection weights |
| `flip` | Vertical flip before processing (microscope-dependent) |
| `registration` | Enable timelapse registration (NCC-based) |
| `segmenter` | Segmentation method: `"cellpose-sam"` or `"omnipose"` |
| `folders` | Mapping of subfolder names to chamber type identifiers |

### Output Structure

**Per-stack directory** (`output/<folder>/<stack_name>/`):

| File | Description |
|------|-------------|
| `stack.tif` | Segmentation masks (T x H x W, uint16 labeled instances) |
| `stack_chamber.tif` | Chamber boundary masks (T x H x W, uint8) |
| `stack_cropped.tif` | Cropped chamber images (with `--save-cropped`) |
| `meta.csv` | Per-frame metadata (timings, registration offsets, cell counts) |
| `cells.csv` | Per-cell measurements (area in pixels and µm²) |

**Batch-level summary** (in the output root directory):

| File | Description |
|------|-------------|
| `results.csv` | Per-stack success/failure and cell counts |
| `timings.csv` | All frame-level pipeline step timings |
| `summary.md` | Human-readable report with per-folder breakdown |
| `timings_table.tex` | LaTeX table of pipeline timings |
| `pipeline_timings.png` | Waterfall timing chart |

### Alternative: Omnipose Segmenter

To use **Omnipose** (with timelapse registration enabled):

```bash
pip install cellpose_omni
python scripts/process_folder.py --config dart_experiment/folder_config.json --save-cropped --verbose
```

### Verify Pixel Calibration (Optional)

To verify that the configured pixel size matches the physical marker distances:

```bash
python scripts/calibrate_pixel_scale.py --config dart_experiment/folder_config_cellpose_sam.json
```

This detects markers in the first frame of each subfolder and compares the detected pixel distance to the expected physical distance, reporting any calibration error.

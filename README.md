![pipeline](https://jugit.fz-juelich.de/emsig/dmc-masking/badges/main/pipeline.svg)
![coverage](https://jugit.fz-juelich.de/emsig/dmc-masking/badges/main/coverage.svg)
<!-- [![codecov](https://codecov.io/gh/OWNER/REPO/graph/badge.svg)](https://codecov.io/gh/OWNER/REPO) -->

# DART

Real-time microfluidic chamber image processing library with two core capabilities:
masking pipeline (marker detection, rotation correction, polygon masking, and cropping)
and map calibration (affine alignment of chip blueprints with microscope stage coordinates).

## Installation

```bash
# Core library
pip install .

# With REST API support
pip install ".[api]"

# Development (tests, linting)
pip install ".[dev]"
```

## Usage

### Python API

```python
from dart_mlci import MarkerDetectionModel, ChipStructureLibrary

# Load chip config and detection model
lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json")
model = MarkerDetectionModel()

# Detect markers
markers = model.predict_markers(image)
```

### Pipeline Steps

```python
from dart_mlci import (
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

### CLI — Calibrate a Map

```bash
python scripts/calibrate_map.py --config calibration.json
```

### CLI — Process Images

```bash
python scripts/process_image.py --image image.tif --roi-id 0050
```

### REST API

```bash
uvicorn dart_mlci.api.main:app --host 0.0.0.0 --port 8000
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

## Reproducible Experiments

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

### Map Calibration

The second core contribution is **map calibration**: aligning the chip blueprint
with microscope stage coordinates so every chamber can be revisited automatically.
The `reproduce.sh` script includes these steps (steps 4–6), but they can also be
run manually.

#### 1. Download calibration data

The calibration dataset contains microscope images with known stage positions.
A **subset** (~250 MB, 23 images) is used for the smoke test; the **full** dataset
(~9 GB, 1164 images) is used with `--full`.

#### 2. Run map calibration

```bash
python scripts/calibrate_map.py \
    --config dart_experiment/calibration_data/calibration_config.json \
    --output dart_experiment/output_reproduce/calibrated_map.csv \
    --output-dir dart_experiment/output_reproduce/calibration \
    --verbose
```

This uses 3 calibration images to compute an affine transform from the chip
blueprint (in `artifacts/chips/sak.json`) to microscope coordinates, producing
`calibrated_map.csv` with positions for all 1164 chambers.

#### 3. Validate calibrated map

```bash
python scripts/validate_map.py \
    --config dart_experiment/output_reproduce/validation_config.json \
    --output-dir dart_experiment/output_reproduce/validation \
    --verbose
```

This compares the calibrated map predictions against measured positions from
independent validation images, producing `validation_results.csv`,
`error_histogram.png`, and `error_map.png`.

### Verify Pixel Calibration (Optional)

To verify that the configured pixel size matches the physical marker distances:

```bash
python scripts/calibrate_pixel_scale.py --config dart_experiment/folder_config_cellpose_sam.json
```

This detects markers in the first frame of each subfolder and compares the detected pixel distance to the expected physical distance, reporting any calibration error.

## License

[MIT](LICENSE) — Copyright (c) 2025 Johannes Seiffarth, Forschungszentrum Juelich GmbH

## Citation

If you use DART in your research, please cite:

```
@software{dart,
  author  = {Seiffarth, Johannes},
  title   = {DART: Real-time microfluidic chamber image processing},
  year    = {2025},
  url     = {https://github.com/JojoDevel/dart-mlci}
}
```

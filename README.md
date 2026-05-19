[![CI](https://github.com/SMLCI/DART-MLCI/actions/workflows/ci.yml/badge.svg)](https://github.com/SMLCI/DART-MLCI/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SMLCI/DART-MLCI/branch/main/graph/badge.svg)](https://codecov.io/gh/SMLCI/DART-MLCI)
[![PyPI](https://img.shields.io/pypi/v/dart-mlci.svg)](https://pypi.org/project/dart-mlci/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# DART-MLCI: Aligning Blueprint and Physical Microfluidic Chip for Design-Aware and Real-Time Capable Live-Cell Image Analysis

**Real-time microfluidic RoI image processing.** DART-MLCI takes a raw
microscopy frame, detects alignment markers, rotates and crops the RoI,
masks the region of interest, and segments cells — automatically, for any chip
design described by a single JSON config (fine alignment).

**Time-constant microfluidic chip mapping.** DART-MLCI allows to record several
RoI positions on the microfluidic chip when its on the microscopy stage and aligns
the microfluidic blueprint providing all RoI positions for any chip design described
by a single JSON config (coarse alignment).

<p align="center">
  <img src="https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/pipeline_teaser.gif" alt="DART pipeline: detect → match → rotate → mask → segment" width="640" />
</p>

The repository covers two complementary capabilities:

- **Fine algiment** (Masking pipeline) — marker detection, pair matching, rotation correction,
  polygon masking, and ROI cropping.
- **Coarse alignment** (Map calibration) — affine alignment of the chip blueprint with microscope
  stage coordinates, so every chamber can be revisited automatically.

## Installation

```bash
# From PyPI
pip install dart-mlci

# Or from source
git clone https://github.com/SMLCI/DART-MLCI.git
cd DART-MLCI
pip install ".[dev]"        # core + dev extras

# Download YOLO marker-detection weights (not bundled in PyPI package)
bash scripts/download_artifacts.sh
```

Optional extras: `pip install "dart-mlci[api]"` for the FastAPI service,
`pip install "dart-mlci[segmentation]"` to pull in Cellpose-SAM / Omnipose.

## Quickstart: Python API

Run the marker detector on a bundled sample image:

```python
import cv2
from dart_mlci import MarkerDetectionModel, ChipStructureLibrary

lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json", pixel_size=0.065789)
model = MarkerDetectionModel()

image = cv2.imread("artifacts/images/sak/0007.png")
markers = model.predict_markers(image)
print(f"Detected {len(markers)} markers")  # e.g. "Detected 4 markers"
```

Compose the full pipeline as steps. `lib(roi_id)` returns the chamber's
polygon and the expected pixel-space positions of its cross/circle markers
(`mgp`):

```python
import cv2
from dart_mlci import (
    ChipStructureLibrary,
    MarkerDetectionStep, MarkerMatchingStep, ImageRotationStep, RoIMaskingStep,
)

lib = ChipStructureLibrary.from_file("artifacts/chips/sak.json", pixel_size=0.065789)
_, polygon, mgp = lib("0000")  # any NormaleBox-inner ROI matches 0007.png

image  = cv2.imread("artifacts/images/sak/0007.png")
detect = MarkerDetectionStep()
match  = MarkerMatchingStep(marker_group_pixel=mgp)
rotate = ImageRotationStep()
mask   = RoIMaskingStep(marker_group_pixels=mgp, roi_polygon=polygon)

data = mask(rotate(match(detect(image))))
cropped, chamber_mask = data["image"], data["mask"]
```

## Process Your Own TIFF Stack

`scripts/process_folder.py` is the CLI for batch-processing time-lapse TIFF
stacks. It works for a single stack just as well as a full experiment — point
its `folders` dict at one subfolder.

### 1. Lay out your data

```
my_data/
├── my_chamber/
│   ├── stack_001.tif         # T x H x W, uint16
│   └── stack_002.tif
```

### 2. Write a minimal `folder_config.json`

```json
{
  "input_dir": "my_data",
  "output_dir": "my_output",
  "pixel_size": 0.0928,
  "chip_config": "artifacts/chips/sak.json",
  "model_path": "artifacts/models/v26_detect_s_imgsz1280.pt",
  "segmenter": "cellpose-sam",
  "folders": {
    "my_chamber": "NormaleBox-inner"
  }
}
```

Set `pixel_size` to your microscope's µm/px. Pick the `chamber_types` key from
your chip JSON (run `python -c "from dart_mlci import ChipStructureLibrary;
print(ChipStructureLibrary.from_file('artifacts/chips/sak.json').chamber_types.keys())"`
to list them).

### 3. Run

```bash
python scripts/process_folder.py \
    --config folder_config.json \
    --save-cropped \
    --render-stacks \
    --verbose
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--render-stacks` | Emit `timelapse.mp4` (segmentation overlay) and `timelapse_raw.mp4` (rotated raw) per stack. |
| `--save-cropped` | Save the rotated+cropped raw frames as `stack_cropped.tif`. |
| `--min-area-um2`, `--max-area-um2` | Drop segmentation artifacts outside the size range. |
| `--max-files N` | Process only the first N stacks (smoke testing). |
| `--skip-existing` | Resume a partially complete run. |

### 4. What you get

Each stack produces `stack.tif` (instance labels), `cells.csv` (per-cell
measurements), `meta.csv` (per-frame metadata), and — with `--render-stacks` —
two MP4s. The full output schema is in
[`docs/configuration.md`](docs/configuration.md).

> Using a chip other than SAK? Author a new chip JSON — see
> [Adapting to a new chip design](#adapting-to-a-new-chip-design).

## Reproduce the Paper Experiment

A single command runs the full pipeline against the public DART dataset:

```bash
bash reproduce.sh            # smoke test (1 stack per folder, ~250 MB)
bash reproduce.sh --full     # full experiment (all stacks, ~9 GB)
```

The script creates a conda env, downloads the dataset and calibration data
from Sciebo, calibrates the map, processes all seven chamber types, and
validates the outputs. Per-step details are in
[`docs/configuration.md`](docs/configuration.md).

## Map Calibration

The second core contribution: aligning the chip blueprint with microscope
stage coordinates so every chamber can be revisited automatically.

```bash
# Compute affine transform from 3 calibration images
python scripts/calibrate_map.py \
    --config calibration_config.json \
    --output calibrated_map.csv \
    --output-dir calibration_output/ \
    --verbose

# Validate against independent images
python scripts/validate_map.py \
    --config validation_config.json \
    --output-dir validation_output/ \
    --verbose
```

The calibration step in `reproduce.sh` invokes both. Validation reports
per-point error in microns and pixels and produces a histogram + spatial
error map.

## REST API

```bash
pip install "dart-mlci[api]"
uvicorn dart_mlci.api.main:app --host 0.0.0.0 --port 8000
# Interactive docs at http://localhost:8000/docs
```

```python
import base64, requests
b64 = base64.b64encode(open("image.tif", "rb").read()).decode()
resp = requests.post("http://localhost:8000/process-image",
                     json={"image": b64, "roi_id": "0050"})
print(resp.json()["chamber_type"])
```

Or via Docker: `docker-compose up --build`.

Visit `http://localhost:8000/docs` for interactive Swagger documentation of
every endpoint. See [`docs/DOCKER_GUIDE.md`](docs/DOCKER_GUIDE.md) for
container deployment.

## Demo Gallery

Pipeline walkthrough videos for each of the seven SAK chamber types. Click a
thumbnail to download the full-resolution MP4 from the
[v0.1.0 release](https://github.com/SMLCI/DART-MLCI/releases/tag/v0.1.0).

| Chamber Type | Preview |
|---|---|
| Normal Box | [![NormaleBox-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/NormaleBox-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/NormaleBox-inner.mp4) |
| Normal Box + Pillar | [![NormaleBox-pillar-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/NormaleBox-pillar-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/NormaleBox-pillar-inner.mp4) |
| Big Box | [![BigBox-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/BigBox-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/BigBox-inner.mp4) |
| Big Box + Pillar | [![BigBox-pillar-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/BigBox-pillar-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/BigBox-pillar-inner.mp4) |
| Open Box | [![OpenBox-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/OpenBox-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/OpenBox-inner.mp4) |
| Open Box + Collector | [![OpenBox-collector-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/OpenBox-collector-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/OpenBox-collector-inner.mp4) |
| Mothermachine | [![Mothermachine-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/Mothermachine-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/Mothermachine-inner.mp4) |
| Mothermachine 2x | [![Mothermachine-2x-inner](https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/thumbnails/Mothermachine-2x-inner.png)](https://github.com/SMLCI/DART-MLCI/releases/download/v0.1.0/Mothermachine-2x-inner.mp4) |

Generate fresh demo videos for your own chip with
`python scripts/generate_sak_videos.py` (writes to
`scripts/output/sak_videos/`).

## Adapting to a New Chip Design

The pipeline is chip-agnostic: it only needs one JSON describing chamber
polygons, marker positions, and a blueprint map. Start from `artifacts/chips/sak.json` as a
template, then follow [`docs/CHIP_CONFIG.md`](docs/CHIP_CONFIG.md) for the full
schema, polygon conventions, marker-detection trade-offs (reuse the bundled
YOLO weights vs. retrain on new fiducials), and a validation checklist.

## License

[MIT](LICENSE) — Copyright (c) 2026 Johannes Seiffarth, Forschungszentrum Jülich GmbH.

## Citation

If you use DART-MLCI in your research, please cite:

```bibtex
@software{dart-mlci,
  author  = {Seiffarth, Johannes},
  title   = {DART-MLCI: Real-time microfluidic chamber image processing},
  year    = {2026},
  url     = {https://github.com/SMLCI/DART-MLCI}
}
```

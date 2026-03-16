# DART Architecture

## Overview

DART provides two core capabilities for microfluidic chip microscopy:

1. **Masking pipeline** — detect alignment markers, match pairs, correct image rotation, apply a polygon mask, and crop the region of interest (ROI).
2. **Map calibration** — compute an affine transform between chip blueprint coordinates and microscope stage coordinates so every chamber can be automatically revisited.

## Masking Pipeline

```
Image → MarkerDetectionStep → MarkerMatchingStep → ImageRotationStep → RoIMaskingStep → Cropped ROI + Mask
```

| Step | Module | Description |
|------|--------|-------------|
| Detection | `detection.py` | YOLO model locates cross / circle markers |
| Matching | `match.py` | Pairs cross+circle markers by proximity |
| Rotation | `rotation.py` | Rotates image+markers so chip is axis-aligned |
| Masking | `mask.py` | Translates ROI polygon, rasterizes mask, crops |

The step classes in `pipeline.py` wrap these into a composable chain where each step receives and returns a `data` dict.

Higher-level wrappers (`RoIMasker`, `SingleStructureRoIMasker` in `masker.py`) combine all steps into a single call for batch processing.

## Map Calibration Pipeline

```
Blueprint map + Calibration images → detect markers → measure stage offsets → fit affine → Calibrated map
```

The calibration script (`scripts/calibrate_map.py`) takes a JSON config listing images with known stage positions and ROI IDs. For each image, it runs the masking pipeline to locate the ROI center, computes the offset from the expected blueprint position, and fits an affine transform to map blueprint coordinates to stage coordinates.

## Chip Configuration

Chip designs are described in JSON files under `artifacts/chips/`. Each file defines:

- Chamber type polygons (GeoJSON)
- Marker positions in microns
- Blueprint map (all ROI positions)

The `ChipStructureLibrary` class (`chip.py`) loads these configs and provides lookup by ROI ID.

## Module Map

| Module | Purpose |
|--------|---------|
| `constants.py` | Default model path, pixel size, tolerance |
| `detection.py` | `MarkerDetectionModel`, `extract_data` |
| `masker.py` | `RoIMasker`, `SingleStructureRoIMasker` |
| `pipeline.py` | Step classes for composable pipelines |
| `mask.py` | `RoIPolygon`, `apply_mask` |
| `map.py` | `Map` class, affine transform calibration |
| `chip.py` | `ChipStructureLibrary`, chip config loading |
| `config.py` | `DARTConfig` dataclass system |
| `io.py` | Image / structure file I/O |
| `rotation.py` | GPU-accelerated image rotation |
| `match.py` | Marker pair matching algorithms |
| `registration.py` | Phase-correlation and timelapse registration |
| `utils.py` | `normalize_image`, helpers |
| `visualization/` | Matplotlib plots, OpenCV drawing, video |
| `api/` | FastAPI REST endpoints |

## Guides

- [API Quick Start](API_QUICK_START.md) — REST API usage examples
- [Docker Guide](DOCKER_GUIDE.md) — Container deployment
- [API Migration](API_BASE64_MIGRATION.md) — Base64 encoding notes

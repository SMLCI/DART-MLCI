# Contributing

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/JojoDevel/dart-mlci.git
cd dart-mlci
pip install -e ".[dev]"

# Download model weights and test images
bash scripts/download_artifacts.sh
```

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_chip.py -v

# With coverage
pytest tests/ --cov=dart_mlci --cov-report=html
```

## Linting

```bash
ruff check dart_mlci/ tests/
ruff format dart_mlci/ tests/
```

## Adding a Chip Design

1. Create a JSON file in `artifacts/chips/` (see `sak.json` as a template).
2. Define chamber types with GeoJSON polygons and marker positions in microns.
3. Include the full blueprint map with ROI IDs and structure types.
4. Add tests in `tests/test_chip.py` to validate the config loads correctly.

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
dart_mlci/
  __init__.py          # Public API re-exports
  constants.py         # DEFAULT_MODEL_PATH, pixel sizes, tolerances
  detection.py         # MarkerDetectionModel, extract_data
  masker.py            # RoIMasker, SingleStructureRoIMasker
  pipeline.py          # Step classes (detect → match → rotate → mask)
  mask.py              # RoIPolygon, apply_mask
  map.py               # Map, calibration transforms
  chip.py              # ChipStructureLibrary (unified chip config)
  config.py            # DARTConfig dataclasses
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

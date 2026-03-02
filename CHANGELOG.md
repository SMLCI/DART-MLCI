# Changelog

## 0.1.0

First public release.

### Features

- **Masking pipeline**: YOLO marker detection, pair matching, rotation correction, polygon masking, and ROI cropping.
- **Map calibration**: Affine transform fitting between chip blueprint and microscope stage coordinates.
- **Unified chip config**: Single JSON file per chip design (`artifacts/chips/`) replaces scattered config files.
- **Multi-chip API support**: `DMC_CHIP_CONFIGS_DIR` env var and `chip_name` request parameter for serving multiple chip designs.
- **REST API**: FastAPI endpoints for image processing (`/process-image`), calibration (`/calibrate`), and health checks.
- **Docker deployment**: Dockerfile and docker-compose.yml for containerized operation.
- **GPU acceleration**: Optional GPU-accelerated rotation via kornia.
- **Phase-correlation registration**: Sub-pixel timelapse alignment.

### Code Organization

- Split monolithic `__init__.py` into `detection.py`, `masker.py`, `pipeline.py`, `constants.py`.
- Split `visualization.py` into `visualization/` subpackage (plotting, drawing, video).
- Moved test utilities from `dmc_masking/test_utils/` to `tests/utils/`.
- Added `__all__` to public API.
- Fixed placeholder docstrings.

### Infrastructure

- MIT license added.
- GitLab CI release stage for version tags.
- Updated `.gitignore` to exclude development debris.
- Removed unused `albumentations` dependency.
- Removed unreferenced legacy model weights (`best34.pt`).

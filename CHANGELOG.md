# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-12

First public release on GitHub and PyPI.

### Features

- **Masking pipeline**: YOLO marker detection, pair matching, rotation
  correction, polygon masking, and ROI cropping.
- **Map calibration**: Affine transform fitting between chip blueprint and
  microscope stage coordinates; validation tooling reports per-point error in
  microns and pixels.
- **Unified chip config**: Single JSON file per chip design
  (`artifacts/chips/`) replaces scattered config files.
- **Multi-chip API support**: `DART_CHIP_CONFIGS_DIR` env var and `chip_name`
  request parameter for serving multiple chip designs.
- **Experiment helpers** (`dart_mlci.experiment`): frame selection and path
  resolution for both metadata-driven and TIFF-stack datasets.
- **Area-based segmentation filtering** (`filter_segmentation_by_area`) to
  drop speckles and fused-cell artifacts.
- **Pipeline walkthrough videos**: `scripts/generate_sak_videos.py` and the
  single-frame variant `scripts/generate_dart_frame_video.py`.
- **Reproducible-experiment script** (`reproduce.sh`) covering download →
  calibrate → process → analyze for all seven DART chamber types.
- **REST API**: FastAPI endpoints for image processing (`/process-image`),
  calibration (`/calibrate`), and health checks; Docker deployment via
  `docker-compose.yml`.
- **GPU acceleration**: Optional GPU-accelerated rotation via kornia.
- **Phase-correlation registration**: Sub-pixel timelapse alignment.

### Documentation

- Configuration reference (`docs/configuration.md`).
- New-chip onboarding tutorial (`docs/new_chip_tutorial.md`).
- README quickstart for single-TIFF-stack processing.

### Infrastructure

- MIT license.
- Trusted-publishing release workflow (`.github/workflows/release.yml`)
  publishes to TestPyPI then PyPI on `v*` tags.
- GitLab CI for internal development; GitHub Actions for public CI.

### Notes

- Model weights are **not** bundled in the PyPI package. Run
  `bash scripts/download_artifacts.sh` after install to fetch the YOLO marker
  detector and example images from Sciebo.

[Unreleased]: https://github.com/SMLCI/DART-MLCI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SMLCI/DART-MLCI/releases/tag/v0.1.0

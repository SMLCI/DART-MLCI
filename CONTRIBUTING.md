# Contributing

## Development Setup

```bash
# Clone and install in development mode
git clone <repo-url>
cd dmc-masking
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_chip.py -v

# With coverage
pytest tests/ --cov=dmc_masking --cov-report=html
```

## Linting

```bash
ruff check dmc_masking/ tests/
ruff format dmc_masking/ tests/
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

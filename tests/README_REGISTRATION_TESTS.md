# Registration Tests - Quick Start Guide

This directory contains comprehensive tests for the `TimelapseRegistration` class.

## Quick Start

### Run All Registration Tests
```bash
pytest tests/test_registration.py tests/test_registration_integration.py -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_registration.py -v

# Integration tests only
pytest tests/test_registration_integration.py -v

# Tests for a specific method
pytest tests/test_registration.py::TestTranslationComputation -v

# Single test
pytest tests/test_registration.py::TestTranslationComputation::test_compute_translation_known_shift -v
```

### Code Coverage
```bash
# Generate coverage report
coverage run -m pytest tests/test_registration.py tests/test_registration_integration.py
coverage report --include="dmc_masking/registration.py"

# HTML coverage report
coverage html
# Open htmlcov/index.html in browser
```

## Test Structure

### Unit Tests (`test_registration.py`)
- **TestTimelapseRegistrationInit** - Initialization and configuration
- **TestMarkerRegionExtraction** - Marker region extraction
- **TestTranslationComputation** - Translation detection accuracy
- **TestApplyTranslationToImage** - Image transformation
- **TestApplyTranslationToMask** - Mask transformation
- **TestFormatCompatibility** - Format conversions
- **TestEdgeCases** - Error handling and edge cases
- **TestParametricTests** - Configuration parameters

### Integration Tests (`test_registration_integration.py`)
- **TestSyntheticTimelapse** - End-to-end time-lapse registration
- **TestAccuracyStatistics** - Statistical accuracy validation
- **TestPerformance** - Speed and memory benchmarks
- **TestRobustness** - Challenging scenarios

## Test Utilities

### Synthetic Data Generation
Located in `tests/fixtures/synthetic_markers.py`:

```python
from tests.fixtures.synthetic_markers import (
    create_synthetic_marker_image,
    apply_known_translation,
    create_marker_group_pixel,
    create_synthetic_timelapse,
)

# Create test image with markers
image = create_synthetic_marker_image(
    width=640,
    height=480,
    marker_positions={"cross": (200, 200), "circle": (300, 250)},
    background="uniform"
)

# Apply known translation for testing
shifted = apply_known_translation(image, dx=5, dy=-3)
```

### Pytest Fixtures
Available in `tests/conftest.py`:

```python
# Use in your tests via function parameters
def test_something(marker_group_fixture, registration_instance_fixture):
    # marker_group_fixture: Standard marker configuration
    # registration_instance_fixture: Pre-initialized TimelapseRegistration
    pass
```

## Performance Benchmarks

Run performance tests with output:
```bash
# CPU benchmark
pytest tests/test_registration_integration.py::TestPerformance::test_registration_speed_benchmark -v -s

# GPU benchmark (requires CUDA)
pytest tests/test_registration_integration.py::TestPerformance::test_registration_speed_benchmark_gpu -v -s

# Memory efficiency test
pytest tests/test_registration_integration.py::TestPerformance::test_memory_efficiency -v -s
```

Expected performance (640×480 images):
- **CPU:** 50-100ms per frame pair
- **GPU:** < 100ms per frame pair (typically 20-50ms)

## Adding New Tests

### Example Test Template
```python
def test_my_feature(registration_instance_fixture):
    """Test description."""
    # Arrange
    marker_positions = {"cross": (200, 200), "circle": (300, 250)}
    image = create_synthetic_marker_image(640, 480, marker_positions)

    # Act
    result = registration_instance_fixture.some_method(image)

    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

### Parametrized Tests
```python
@pytest.mark.parametrize("shift", [(5, 0), (0, 5), (-3, 4)])
def test_multiple_shifts(registration_instance_fixture, shift):
    """Test with multiple shift values."""
    dx, dy = shift
    # Your test code here
```

## Troubleshooting

### CUDA Tests Skipped
If you see "CUDA not available" messages, CUDA tests are automatically skipped. This is normal on CPU-only systems.

### Import Errors
Make sure the package is installed:
```bash
pip install -e .
```

### Coverage Tool Issues
If coverage fails with torch import errors, run tests without coverage:
```bash
pytest tests/test_registration.py tests/test_registration_integration.py -v
```

## Test Files

- `test_registration.py` - Unit tests (52 tests)
- `test_registration_integration.py` - Integration tests (9 tests)
- `fixtures/synthetic_markers.py` - Test data generation utilities
- `conftest.py` - Pytest fixtures and configuration
- `TEST_REGISTRATION_SUMMARY.md` - Detailed test documentation

## Current Status

✅ **61 tests passing**
✅ **98% code coverage**
✅ **~5.5 second execution time**
✅ **All accuracy criteria met**
✅ **Performance requirements exceeded**

## See Also

- `TEST_REGISTRATION_SUMMARY.md` - Comprehensive test suite documentation
- `dmc_masking/registration.py` - Implementation under test

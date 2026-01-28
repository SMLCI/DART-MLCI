"""Pytest fixtures for DMC masking tests."""

import pytest
import torch

from tests.fixtures.synthetic_markers import (
    apply_known_translation,
    create_marker_group_pixel,
    create_synthetic_marker_image,
)


@pytest.fixture
def marker_group_fixture():
    """Standard marker configuration for tests."""
    marker_positions = {
        "cross": (200, 200),
        "circle": (300, 250),
    }
    return create_marker_group_pixel(marker_positions)


@pytest.fixture
def synthetic_image_pair_fixture():
    """Reference and target images with known translation."""
    marker_positions = {
        "cross": (200, 200),
        "circle": (300, 250),
    }

    # Create reference image
    reference = create_synthetic_marker_image(
        width=640,
        height=480,
        marker_positions=marker_positions,
        background="uniform",
    )

    # Create target with known shift
    true_dx, true_dy = 5, -3
    target = apply_known_translation(reference, true_dx, true_dy)

    return reference, target, true_dx, true_dy


@pytest.fixture
def registration_instance_fixture(marker_group_fixture):
    """Pre-initialized TimelapseRegistration instance."""
    from dmc_masking.registration import TimelapseRegistration

    return TimelapseRegistration(
        marker_group_pixel=marker_group_fixture,
        max_translation=20,
        padding=50,
        device="cpu",  # Use CPU for consistent testing
    )


@pytest.fixture
def device_fixture():
    """Return available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

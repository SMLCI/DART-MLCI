"""Pytest fixtures for DMC masking tests."""

import pytest
import torch


@pytest.fixture
def device_fixture():
    """Return available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

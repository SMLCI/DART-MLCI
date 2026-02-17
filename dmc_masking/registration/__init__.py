"""Translation-based image registration for time-lapse microscopy stacks.

Public API
----------
BaseRegistration               — Abstract base class
TimelapseRegistration          — NCC-based registration (GPU-accelerated via kornia)
PhaseCorrelationRegistration   — FFT-based registration (OpenCV)
"""

from dmc_masking.registration._base import BaseRegistration
from dmc_masking.registration._ncc import TimelapseRegistration
from dmc_masking.registration._phase_corr import PhaseCorrelationRegistration

__all__ = [
    "BaseRegistration",
    "PhaseCorrelationRegistration",
    "TimelapseRegistration",
]

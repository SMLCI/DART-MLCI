"""Backward-compatibility shim — prefer ``tests.utils`` for new code."""

import warnings

warnings.warn(
    "dmc_masking.test_utils is deprecated. Use tests.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from tests.utils import *  # noqa: F403, E402
from tests.utils import __all__  # noqa: F401, E402

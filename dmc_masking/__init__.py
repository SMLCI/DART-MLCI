"""DMC Masking — real-time microfluidic chamber image processing.

Two core capabilities:
  1. **Masking pipeline** — detect markers, match pairs, correct rotation,
     apply polygon mask, crop ROI.
  2. **Map calibration** — align chip blueprint coordinates with microscope
     stage coordinates via affine transform.
"""

# Constants
from dmc_masking.chip import ChipConfig as ChipConfig
from dmc_masking.chip import ChipStructureLibrary as ChipStructureLibrary
from dmc_masking.chip import load_chip_config as load_chip_config
from dmc_masking.constants import DEFAULT_MARKER_TOLERANCE_PX as DEFAULT_MARKER_TOLERANCE_PX
from dmc_masking.constants import DEFAULT_MODEL_PATH as DEFAULT_MODEL_PATH
from dmc_masking.constants import DEFAULT_PIXEL_SIZE_UM as DEFAULT_PIXEL_SIZE_UM
from dmc_masking.detection import MarkerDetectionModel as MarkerDetectionModel
from dmc_masking.detection import extract_data as extract_data
from dmc_masking.mask import RoIPolygon as RoIPolygon
from dmc_masking.mask import SingleRoIStructureLibrary as SingleRoIStructureLibrary
from dmc_masking.mask import apply_mask as apply_mask
from dmc_masking.masker import RoIMasker as RoIMasker
from dmc_masking.masker import SingleStructureRoIMasker as SingleStructureRoIMasker
from dmc_masking.masker import compute_marker_angles as compute_marker_angles
from dmc_masking.pipeline import ImageRotationStep as ImageRotationStep
from dmc_masking.pipeline import MarkerDetectionStep as MarkerDetectionStep
from dmc_masking.pipeline import MarkerMatchingStep as MarkerMatchingStep
from dmc_masking.pipeline import RoIMaskingStep as RoIMaskingStep
from dmc_masking.registration import PhaseCorrelationRegistration as PhaseCorrelationRegistration
from dmc_masking.registration import TimelapseRegistration as TimelapseRegistration

__all__ = [
    "DEFAULT_MARKER_TOLERANCE_PX",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PIXEL_SIZE_UM",
    "ChipConfig",
    "ChipStructureLibrary",
    "ImageRotationStep",
    "MarkerDetectionModel",
    "MarkerDetectionStep",
    "MarkerMatchingStep",
    "PhaseCorrelationRegistration",
    "RoIMasker",
    "RoIMaskingStep",
    "RoIPolygon",
    "SingleStructureRoIMasker",
    "TimelapseRegistration",
    "apply_mask",
    "compute_marker_angles",
    "extract_data",
    "load_chip_config",
]

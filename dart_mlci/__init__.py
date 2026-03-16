"""DART MLCI — real-time microfluidic chamber image processing.

Two core capabilities:
  1. **Masking pipeline** — detect markers, match pairs, correct rotation,
     apply polygon mask, crop ROI.
  2. **Map calibration** — align chip blueprint coordinates with microscope
     stage coordinates via affine transform.
"""

# Constants
from dart_mlci.chip import ChipConfig as ChipConfig
from dart_mlci.chip import ChipStructureLibrary as ChipStructureLibrary
from dart_mlci.chip import load_chip_config as load_chip_config
from dart_mlci.constants import DEFAULT_MARKER_TOLERANCE_PX as DEFAULT_MARKER_TOLERANCE_PX
from dart_mlci.constants import DEFAULT_MODEL_PATH as DEFAULT_MODEL_PATH
from dart_mlci.constants import DEFAULT_PIXEL_SIZE_UM as DEFAULT_PIXEL_SIZE_UM
from dart_mlci.detection import MarkerDetectionModel as MarkerDetectionModel
from dart_mlci.detection import extract_data as extract_data
from dart_mlci.mask import RoIPolygon as RoIPolygon
from dart_mlci.mask import SingleRoIStructureLibrary as SingleRoIStructureLibrary
from dart_mlci.mask import apply_mask as apply_mask
from dart_mlci.mask import filter_segmentation_by_mask as filter_segmentation_by_mask
from dart_mlci.masker import RoIMasker as RoIMasker
from dart_mlci.masker import SingleStructureRoIMasker as SingleStructureRoIMasker
from dart_mlci.masker import compute_marker_angles as compute_marker_angles
from dart_mlci.pipeline import ImageRotationStep as ImageRotationStep
from dart_mlci.pipeline import MarkerDetectionStep as MarkerDetectionStep
from dart_mlci.pipeline import MarkerMatchingStep as MarkerMatchingStep
from dart_mlci.pipeline import RoIMaskingStep as RoIMaskingStep
from dart_mlci.registration import PhaseCorrelationRegistration as PhaseCorrelationRegistration
from dart_mlci.registration import TimelapseRegistration as TimelapseRegistration

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
    "filter_segmentation_by_mask",
    "load_chip_config",
]

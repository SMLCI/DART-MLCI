"""DART MLCI — real-time microfluidic chamber image processing.

Two core capabilities:
  1. **Masking pipeline** — detect markers, match pairs, correct rotation,
     apply polygon mask, crop ROI.
  2. **Map calibration** — align chip blueprint coordinates with microscope
     stage coordinates via affine transform.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dart-mlci")
except PackageNotFoundError:
    __version__ = "0.2.1"

# Constants
# Analysis
from dart_mlci.analysis import ExponentialFitResult as ExponentialFitResult
from dart_mlci.analysis import LogisticFitResult as LogisticFitResult
from dart_mlci.analysis import compute_growth_stats as compute_growth_stats
from dart_mlci.analysis import discover_cells_csvs as discover_cells_csvs
from dart_mlci.analysis import filter_cells_by_area as filter_cells_by_area
from dart_mlci.analysis import fit_exponential_growth as fit_exponential_growth
from dart_mlci.analysis import fit_logistic_growth as fit_logistic_growth
from dart_mlci.analysis import load_cells_data as load_cells_data
from dart_mlci.chip import ChipConfig as ChipConfig
from dart_mlci.chip import ChipStructureLibrary as ChipStructureLibrary
from dart_mlci.chip import create_structure_library as create_structure_library
from dart_mlci.chip import load_chip_config as load_chip_config
from dart_mlci.constants import ARTIFACTS_DIR as ARTIFACTS_DIR
from dart_mlci.constants import DEFAULT_CHIP_CONFIG_PATH as DEFAULT_CHIP_CONFIG_PATH
from dart_mlci.constants import DEFAULT_MARKER_TOLERANCE_PX as DEFAULT_MARKER_TOLERANCE_PX
from dart_mlci.constants import DEFAULT_MODEL_PATH as DEFAULT_MODEL_PATH
from dart_mlci.constants import DEFAULT_PIXEL_SIZE_UM as DEFAULT_PIXEL_SIZE_UM
from dart_mlci.constants import DEFAULT_STRUCTURE_LIBRARY_PATH as DEFAULT_STRUCTURE_LIBRARY_PATH
from dart_mlci.detection import MarkerDetectionModel as MarkerDetectionModel
from dart_mlci.detection import extract_data as extract_data
from dart_mlci.experiment import absolutize_image_paths as absolutize_image_paths
from dart_mlci.experiment import load_tif_frame as load_tif_frame
from dart_mlci.experiment import (
    resolve_chamber_type_from_folder_config as resolve_chamber_type_from_folder_config,
)
from dart_mlci.experiment import resolve_time_column as resolve_time_column
from dart_mlci.experiment import select_timelapse_frame as select_timelapse_frame
from dart_mlci.mask import RoIPolygon as RoIPolygon
from dart_mlci.mask import SingleRoIStructureLibrary as SingleRoIStructureLibrary
from dart_mlci.mask import apply_mask as apply_mask
from dart_mlci.mask import filter_segmentation_by_area as filter_segmentation_by_area
from dart_mlci.mask import filter_segmentation_by_mask as filter_segmentation_by_mask
from dart_mlci.masker import RoIMasker as RoIMasker
from dart_mlci.masker import SingleStructureRoIMasker as SingleStructureRoIMasker
from dart_mlci.masker import compute_marker_angles as compute_marker_angles
from dart_mlci.pipeline import ChamberPipelineCache as ChamberPipelineCache
from dart_mlci.pipeline import ImageRotationStep as ImageRotationStep
from dart_mlci.pipeline import MarkerDetectionStep as MarkerDetectionStep
from dart_mlci.pipeline import MarkerMatchingStep as MarkerMatchingStep
from dart_mlci.pipeline import RoIMaskingStep as RoIMaskingStep
from dart_mlci.registration import PhaseCorrelationRegistration as PhaseCorrelationRegistration
from dart_mlci.registration import TimelapseRegistration as TimelapseRegistration
from dart_mlci.timelapse import TimelapseProcessor as TimelapseProcessor
from dart_mlci.timelapse import TimelapseResult as TimelapseResult
from dart_mlci.timelapse import create_segmenter as create_segmenter
from dart_mlci.types import FrameResult as FrameResult
from dart_mlci.types import PipelineError as PipelineError
from dart_mlci.types import PipelineTimings as PipelineTimings
from dart_mlci.types import StackResult as StackResult

__all__ = [
    "ARTIFACTS_DIR",
    "DEFAULT_CHIP_CONFIG_PATH",
    "DEFAULT_MARKER_TOLERANCE_PX",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_STRUCTURE_LIBRARY_PATH",
    "ChamberPipelineCache",
    "ChipConfig",
    "ChipStructureLibrary",
    "ExponentialFitResult",
    "FrameResult",
    "ImageRotationStep",
    "LogisticFitResult",
    "MarkerDetectionModel",
    "MarkerDetectionStep",
    "MarkerMatchingStep",
    "PhaseCorrelationRegistration",
    "PipelineError",
    "PipelineTimings",
    "RoIMasker",
    "RoIMaskingStep",
    "RoIPolygon",
    "SingleStructureRoIMasker",
    "StackResult",
    "TimelapseProcessor",
    "TimelapseRegistration",
    "TimelapseResult",
    "__version__",
    "absolutize_image_paths",
    "apply_mask",
    "compute_growth_stats",
    "compute_marker_angles",
    "create_segmenter",
    "create_structure_library",
    "discover_cells_csvs",
    "extract_data",
    "filter_cells_by_area",
    "filter_segmentation_by_area",
    "filter_segmentation_by_mask",
    "fit_exponential_growth",
    "fit_logistic_growth",
    "load_cells_data",
    "load_chip_config",
    "load_tif_frame",
    "resolve_chamber_type_from_folder_config",
    "resolve_time_column",
    "select_timelapse_frame",
]

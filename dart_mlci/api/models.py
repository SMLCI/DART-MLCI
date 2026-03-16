"""Pydantic request/response models for the DMC Masking API."""

import base64

from pydantic import BaseModel, Field, field_validator


class StagePosition(BaseModel):
    """Stage position in microscope coordinates."""

    x: float
    y: float
    z: float | None = None


class CalibrationImageMeta(BaseModel):
    """Metadata for a single calibration image."""

    roi_id: str
    stage_position: StagePosition


class CalibrateConfig(BaseModel):
    """Configuration for map calibration - same JSON format as scripts/calibrate_map.py."""

    chip_name: str = Field(default="SAK", description="Name of the chip design")
    calibration_images: list[CalibrationImageMeta] = Field(
        description="List of calibration image metadata (order must match uploaded images)"
    )
    pixel_size: float = Field(default=0.065789, description="Pixel size in microns")
    model_path: str | None = Field(
        default=None, description="Path to YOLO model (uses default if not set)"
    )


class ImageResultInfo(BaseModel):
    """Per-image result from calibration."""

    roi_id: str
    success: bool
    error_message: str | None = None
    microscope_position: list[float] | None = None


class CalibratedROIPosition(BaseModel):
    """A single calibrated ROI position."""

    roi_id: str = Field(description="ROI identifier")
    x: float = Field(description="Calibrated X position")
    y: float = Field(description="Calibrated Y position")
    z: float | None = Field(default=None, description="Calibrated Z position (if available)")


class CalibrationStatistics(BaseModel):
    """Statistics from the affine transform calibration."""

    rmse: float = Field(description="Root mean square error of the calibration")
    max_error: float = Field(description="Maximum residual error")
    n_points: int = Field(description="Number of calibration points used")
    residuals: list[float] = Field(description="Per-point residual distances")


class CalibrateResponse(BaseModel):
    """Response from the calibrate endpoint."""

    success: bool
    calibrated_map: list[CalibratedROIPosition] | None = Field(
        default=None,
        description="List of calibrated ROI positions",
    )
    statistics: CalibrationStatistics | None = Field(
        default=None, description="Calibration statistics"
    )
    image_results: list[ImageResultInfo] = Field(
        default_factory=list, description="Per-image processing results"
    )
    error_message: str | None = None


class ProcessImageResponse(BaseModel):
    """Response from the process-image endpoint."""

    success: bool
    cropped_image: str | None = Field(
        default=None, description="Base64-encoded PNG of cropped image"
    )
    mask: str | None = Field(default=None, description="Base64-encoded PNG of binary mask")
    rotation_angle: float | None = None
    chamber_type: str | None = None
    error_message: str | None = None


class HealthResponse(BaseModel):
    """Response from the health endpoint."""

    status: str = Field(description="'healthy' or 'unhealthy'")
    model_loaded: bool
    default_structure_library: str
    gpu_available: bool
    device: str = Field(description="e.g., 'cuda:0' or 'cpu'")
    segmenter_loaded: bool = False
    segmenter: str | None = None


class ChamberType(BaseModel):
    """Information about a chamber type."""

    name: str
    roi_pattern: str | None = Field(
        default=None, description="Regex pattern for matching ROI IDs (deprecated)"
    )


class ProcessImageRequest(BaseModel):
    """JSON request for image processing with base64-encoded image."""

    image: str = Field(description="Base64-encoded image (optionally with data URI prefix)")
    roi_id: str = Field(description="ROI identifier (e.g., '0050')")
    pixel_size: float = Field(default=0.065789, description="Pixel size in microns")
    chip_name: str | None = Field(
        default=None,
        description="Name of the chip design (e.g., 'sak'). Selects from loaded chip configs.",
    )
    structure_library_path: str | None = Field(
        default=None,
        description="Optional custom structure library path (deprecated, use chip_name)",
    )
    chip_config_path: str | None = Field(
        default=None,
        description="Optional path to unified chip config JSON (deprecated, use chip_name)",
    )
    return_uncropped: bool = Field(default=False, description="Return full-size mask")

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Strip data URI prefix and validate base64."""
        # Strip data URI prefix if present
        if v.startswith("data:"):
            v = v.split(",", 1)[1] if "," in v else v

        # Validate base64 encoding
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}") from e

        return v


class CalibrationImageData(BaseModel):
    """Single calibration image with base64-encoded data."""

    image: str = Field(description="Base64-encoded image (optionally with data URI prefix)")
    roi_id: str = Field(description="ROI identifier")
    stage_position: StagePosition = Field(description="Microscope stage position")

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Strip data URI prefix and validate base64."""
        # Strip data URI prefix if present
        if v.startswith("data:"):
            v = v.split(",", 1)[1] if "," in v else v

        # Validate base64 encoding
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}") from e

        return v


class SegmentRequest(BaseModel):
    """JSON request for instance segmentation."""

    image: str = Field(description="Base64-encoded cropped image (HxWx3 uint8 PNG)")
    mask: str = Field(description="Base64-encoded chamber mask (HxW bool PNG)")
    filter_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of cell area in masked region to trigger removal",
    )
    relabel: bool = Field(default=True, description="Relabel remaining IDs to be contiguous")

    @field_validator("image")
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        """Strip data URI prefix and validate base64."""
        if v.startswith("data:"):
            v = v.split(",", 1)[1] if "," in v else v
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}") from e
        return v

    @field_validator("mask")
    @classmethod
    def validate_mask_base64(cls, v: str) -> str:
        """Strip data URI prefix and validate base64."""
        if v.startswith("data:"):
            v = v.split(",", 1)[1] if "," in v else v
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}") from e
        return v


class SegmentResponse(BaseModel):
    """Response from the segment endpoint."""

    success: bool
    segmentation_mask: str | None = Field(
        default=None,
        description="Base64-encoded 16-bit grayscale PNG (pixel value = instance ID, 0 = background)",
    )
    cell_count: int | None = None
    total_cell_area: int | None = Field(
        default=None, description="Total area of all cell instances in pixels"
    )
    error_message: str | None = None


class CalibrateRequest(BaseModel):
    """JSON request for calibration with base64-encoded images."""

    chip_name: str = Field(default="SAK", description="Name of the chip design")
    calibration_images: list[CalibrationImageData] = Field(
        min_length=3, description="List of calibration images (minimum 3 required)"
    )
    pixel_size: float = Field(default=0.065789, description="Pixel size in microns")
    model_path: str | None = Field(
        default=None, description="Path to YOLO model (uses default if not set)"
    )

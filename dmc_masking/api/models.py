"""Pydantic request/response models for the DMC Masking API."""

from pydantic import BaseModel, Field


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
    blueprint_map_path: str = Field(description="Path to the blueprint map CSV")
    structure_library_path: str | None = Field(
        default=None, description="Path to structure library JSON (uses default if not set)"
    )
    model_path: str | None = Field(
        default=None, description="Path to YOLO model (uses default if not set)"
    )


class ImageResultInfo(BaseModel):
    """Per-image result from calibration."""

    roi_id: str
    success: bool
    error_message: str | None = None
    microscope_position: list[float] | None = None


class CalibrateResponse(BaseModel):
    """Response from the calibrate endpoint."""

    success: bool
    calibrated_map_csv: str | None = Field(default=None, description="CSV content as string")
    statistics: dict | None = Field(
        default=None, description="Calibration statistics (rmse, max_error, n_points)"
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


class ChamberType(BaseModel):
    """Information about a chamber type."""

    name: str
    roi_pattern: str = Field(description="Regex pattern for matching ROI IDs")

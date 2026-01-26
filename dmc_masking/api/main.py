"""FastAPI application for DMC Masking calibration and image processing."""

import base64
import io
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from dmc_masking import MarkerDetectionStep, MarkerMatchingStep
from dmc_masking.api.models import (
    CalibrateConfig,
    CalibrateResponse,
    ChamberType,
    HealthResponse,
    ImageResultInfo,
    ProcessImageResponse,
)
from dmc_masking.api.settings import get_settings, resolve_path
from dmc_masking.mask import SAKRoIStructureLibrary, apply_mask_rotation_free
from dmc_masking.rotation import compute_marker_group_angles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    settings = get_settings()

    # Load marker detection model
    model_path = resolve_path(settings.model_path)
    if model_path.exists():
        app.state.detection_step = MarkerDetectionStep(
            model_path=str(model_path),
            device=settings.device,
            verbose=False,
        )
        app.state.model_loaded = True
    else:
        app.state.detection_step = None
        app.state.model_loaded = False

    # Load default structure library
    structure_library_path = resolve_path(settings.structure_library_path)
    if structure_library_path.exists():
        app.state.structure_library = SAKRoIStructureLibrary(
            lookup_path=str(structure_library_path),
            pixel_size=settings.default_pixel_size,
        )
    else:
        app.state.structure_library = None

    # Detect GPU availability
    app.state.gpu_available = torch.cuda.is_available()
    if settings.device:
        app.state.device = settings.device
    elif app.state.gpu_available:
        app.state.device = "cuda:0"
    else:
        app.state.device = "cpu"

    yield

    # Cleanup (if needed)
    app.state.detection_step = None
    app.state.structure_library = None


app = FastAPI(
    title="DMC Masking API",
    description="API for microfluidic chamber calibration and image processing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and loaded resources."""
    settings = get_settings()
    model_loaded = getattr(app.state, "model_loaded", False)

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        default_structure_library=settings.structure_library_path,
        gpu_available=getattr(app.state, "gpu_available", False),
        device=getattr(app.state, "device", "cpu"),
    )


@app.get("/chamber-types", response_model=list[ChamberType])
async def list_chamber_types() -> list[ChamberType]:
    """List available chamber structure types and their ROI ID patterns."""
    structure_library = getattr(app.state, "structure_library", None)
    if structure_library is None:
        raise HTTPException(status_code=503, detail="Structure library not loaded")

    chamber_types = []
    for name, pattern in structure_library.patterns.items():
        chamber_types.append(ChamberType(name=name, roi_pattern=pattern.pattern))

    return chamber_types


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(
    image: UploadFile = File(...),
    roi_id: str = Form(...),
    pixel_size: float = Form(default=0.065789),
    structure_library_path: str | None = Form(default=None),
    return_uncropped: bool = Form(default=False),
) -> ProcessImageResponse:
    """Process a single image to extract and mask the chamber region.

    Args:
        image: Image file upload (TIFF, PNG, etc.)
        roi_id: ROI identifier (e.g., "0050") - auto-detects chamber type
        pixel_size: Pixel size in microns (default: 0.065789)
        structure_library_path: Optional path to structure JSON for different chip designs
        return_uncropped: If True, return full-size mask instead of cropped

    Returns:
        Cropped image and mask as base64-encoded PNGs
    """
    # Check if detection model is loaded
    detection_step = getattr(app.state, "detection_step", None)
    if detection_step is None:
        return ProcessImageResponse(
            success=False,
            error_message="Model not loaded - check DMC_MODEL_PATH configuration",
        )

    # Load structure library (custom or default)
    if structure_library_path:
        lib_path = resolve_path(structure_library_path)
        if not lib_path.exists():
            return ProcessImageResponse(
                success=False,
                error_message=f"Structure library not found: {structure_library_path}",
            )
        structure_library = SAKRoIStructureLibrary(
            lookup_path=str(lib_path),
            pixel_size=pixel_size,
        )
    else:
        structure_library = getattr(app.state, "structure_library", None)
        if structure_library is None:
            return ProcessImageResponse(
                success=False,
                error_message="Default structure library not loaded",
            )

    # Load image from upload
    try:
        contents = await image.read()
        # Try to load with PIL first, then tifffile for TIFF
        if image.filename and image.filename.lower().endswith((".tif", ".tiff")):
            import tifffile

            img_array = tifffile.imread(io.BytesIO(contents))
        else:
            pil_img = Image.open(io.BytesIO(contents))
            img_array = np.array(pil_img)
    except Exception as e:
        return ProcessImageResponse(
            success=False,
            error_message=f"Failed to load image: {e}",
        )

    # Ensure image is in correct format (C, H, W) for processing
    if img_array.ndim == 2:
        # Grayscale - add channel dimension
        img_array = img_array[np.newaxis, :, :]
    elif img_array.ndim == 3 and img_array.shape[2] in (3, 4):
        # (H, W, C) format -> transpose to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    # else assume already (C, H, W)

    # Get structure info for this ROI
    try:
        structure_name, roi_polygon, marker_group_configs = structure_library(roi_id)
    except Exception as e:
        return ProcessImageResponse(
            success=False,
            error_message=f"Invalid ROI ID '{roi_id}': {e}",
        )

    # Run marker detection
    detection_result = detection_step(img_array)
    markers = detection_result["markers"]

    if len(markers) < 2:
        return ProcessImageResponse(
            success=False,
            error_message=f"Insufficient markers detected: {len(markers)} (need at least 2)",
        )

    # Get marker group pixel positions for this structure
    marker_group_pixels = marker_group_configs

    # Match markers
    matching_step = MarkerMatchingStep(
        marker_group_pixel=marker_group_pixels,
        tolerance=60,
    )
    matched_result = matching_step(detection_result)
    matched_indices = matched_result.get("matched_marker_indices", [])

    if len(matched_indices) == 0:
        return ProcessImageResponse(
            success=False,
            error_message="No marker pairs could be matched",
        )

    # Compute rotation angle
    angles = compute_marker_group_angles(
        markers=markers,
        matched_marker_indices=matched_indices,
        marker_group=marker_group_pixels,
        on="bbox_center",
        signed=True,
    )
    rotation_angle = np.median(angles) if angles else 0.0

    # Apply mask
    try:
        cropped_img, cropped_mask = apply_mask_rotation_free(
            matched_marker_indices=matched_indices,
            markers=markers,
            marker_group_pixels=marker_group_pixels,
            roi_polygon=roi_polygon,
            image=img_array,
            rotation_angle=rotation_angle,
            return_uncropped=return_uncropped,
        )
    except Exception as e:
        return ProcessImageResponse(
            success=False,
            error_message=f"Masking failed: {e}",
        )

    # Convert to base64 PNG
    def array_to_base64_png(arr: np.ndarray, is_mask: bool = False) -> str:
        # Handle different array shapes
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        # Normalize to uint8 if needed
        if arr.dtype != np.uint8:
            if is_mask:
                arr = (arr > 0).astype(np.uint8) * 255
            else:
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

        # Encode as PNG
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L")
        elif arr.shape[2] == 3:
            pil_img = Image.fromarray(arr, mode="RGB")
        else:
            pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ProcessImageResponse(
        success=True,
        cropped_image=array_to_base64_png(cropped_img),
        mask=array_to_base64_png(cropped_mask, is_mask=True),
        rotation_angle=float(rotation_angle),
        chamber_type=structure_name,
    )


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_map_endpoint(
    images: list[UploadFile] = File(...),
    config: str = Form(...),
) -> CalibrateResponse:
    """Calibrate a map from calibration images with known stage positions.

    Args:
        images: Multiple image file uploads (order must match config)
        config: JSON string with calibration configuration

    Returns:
        Calibrated map CSV and statistics
    """
    # Import here to avoid circular imports and allow lazy loading
    import sys

    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    from scripts.calibrate_map import calibrate_map

    # Parse config
    try:
        config_dict = json.loads(config)
        calibrate_config = CalibrateConfig(**config_dict)
    except json.JSONDecodeError as e:
        return CalibrateResponse(
            success=False,
            error_message=f"Invalid JSON config: {e}",
        )
    except Exception as e:
        return CalibrateResponse(
            success=False,
            error_message=f"Invalid config: {e}",
        )

    # Validate number of images
    n_images = len(images)
    n_config = len(calibrate_config.calibration_images)
    if n_images != n_config:
        return CalibrateResponse(
            success=False,
            error_message=f"Number of images ({n_images}) does not match config ({n_config})",
        )

    if n_images < 3:
        return CalibrateResponse(
            success=False,
            error_message=f"At least 3 calibration images required, got {n_images}",
        )

    # Save uploaded images to temp files and build config
    settings = get_settings()
    calibration_images = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (upload, meta) in enumerate(
            zip(images, calibrate_config.calibration_images, strict=False)
        ):
            # Save image to temp file
            suffix = Path(upload.filename).suffix if upload.filename else ".tif"
            img_path = Path(tmpdir) / f"image_{i}{suffix}"
            contents = await upload.read()
            img_path.write_bytes(contents)

            # Build calibration image entry
            calibration_images.append(
                {
                    "image_path": str(img_path),
                    "roi_id": meta.roi_id,
                    "stage_position": {
                        "x": meta.stage_position.x,
                        "y": meta.stage_position.y,
                        "z": meta.stage_position.z,
                    },
                }
            )

        # Build full config dict
        full_config = {
            "calibration_images": calibration_images,
            "pixel_size": calibrate_config.pixel_size,
            "blueprint_map_path": calibrate_config.blueprint_map_path,
        }

        # Add optional fields
        if calibrate_config.structure_library_path:
            full_config["structure_library_path"] = calibrate_config.structure_library_path
        else:
            full_config["structure_library_path"] = settings.structure_library_path

        if calibrate_config.model_path:
            full_config["model_path"] = calibrate_config.model_path
        else:
            full_config["model_path"] = settings.model_path

        if settings.device:
            full_config["device"] = settings.device

        # Run calibration
        try:
            result, blueprint_map = calibrate_map(
                config=full_config,
                verbose=False,
            )
        except Exception as e:
            return CalibrateResponse(
                success=False,
                error_message=f"Calibration failed: {e}",
            )

    # Convert calibrated map to CSV string
    csv_lines = ["roi_id,x,y"]
    for roi_id, roi_pos in result.calibrated_map.roi_positions.items():
        x, y = roi_pos.position[:2]
        csv_lines.append(f"{roi_id},{x:.6f},{y:.6f}")

    # Add z positions if available
    if result.z_positions:
        csv_lines = ["roi_id,x,y,z"]
        for roi_id, roi_pos in result.calibrated_map.roi_positions.items():
            x, y = roi_pos.position[:2]
            z = result.z_positions.get(roi_id)
            if z is not None:
                csv_lines.append(f"{roi_id},{x:.6f},{y:.6f},{z:.6f}")
            else:
                csv_lines.append(f"{roi_id},{x:.6f},{y:.6f},")

    calibrated_map_csv = "\n".join(csv_lines)

    # Build per-image results
    image_results = []
    for img_result in result.image_results:
        pos = None
        if img_result.microscope_position is not None:
            pos = img_result.microscope_position.tolist()
        image_results.append(
            ImageResultInfo(
                roi_id=img_result.roi_id,
                success=img_result.success,
                error_message=img_result.error_message,
                microscope_position=pos,
            )
        )

    # Build statistics
    statistics = {
        "rmse": float(result.transform_result.rmse),
        "max_error": float(result.transform_result.max_error),
        "n_points": len(result.measured_map.roi_positions),
        "residuals": result.transform_result.residuals.tolist(),
    }

    return CalibrateResponse(
        success=True,
        calibrated_map_csv=calibrated_map_csv,
        statistics=statistics,
        image_results=image_results,
    )


@app.post("/validate", response_model=dict)
async def validate_calibration(
    calibrated_map: UploadFile = File(...),
    images: list[UploadFile] = File(...),
    config: str = Form(...),
) -> dict:
    """Validate a calibrated map against validation images.

    This endpoint is optional and can be used to verify calibration quality.

    Args:
        calibrated_map: CSV file of the calibrated map
        images: Validation images
        config: JSON config with validation image metadata

    Returns:
        Validation metrics (mean, median, max error)
    """
    # This is a simplified validation - just process images and compare
    # to expected positions from the calibrated map

    return {
        "status": "not_implemented",
        "message": "Validation endpoint is a placeholder for future implementation",
    }

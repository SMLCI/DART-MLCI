"""FastAPI application for DMC Masking calibration and image processing."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from dart_mlci import (
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
)
from dart_mlci.api.models import (
    CalibratedROIPosition,
    CalibrateRequest,
    CalibrateResponse,
    CalibrationStatistics,
    ChamberType,
    HealthResponse,
    ImageResultInfo,
    ProcessImageRequest,
    ProcessImageResponse,
    SegmentRequest,
    SegmentResponse,
)
from dart_mlci.api.settings import get_settings, resolve_path
from dart_mlci.artifacts import ensure_artifact
from dart_mlci.chip import ChipStructureLibrary
from dart_mlci.mask import SAKRoIStructureLibrary

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    settings = get_settings()

    # Load marker detection model. Configured path wins; otherwise fall back
    # to the auto-downloading artifact cache. Downloads happen on first use
    # (one-time, then cached); the API stays up even if the network is gone
    # — endpoints that need the model just report model_loaded=false.
    model_path = resolve_path(settings.model_path)
    if not model_path.exists():
        try:
            model_path = ensure_artifact("models/v26_detect_s_imgsz1280.pt")
        except Exception as exc:
            logger.warning("Could not obtain detection model: %s", exc)
            model_path = None

    if model_path is not None:
        app.state.detection_step = MarkerDetectionStep(
            model_path=str(model_path),
            device=settings.device,
            verbose=False,
        )
        app.state.model_loaded = True
    else:
        app.state.detection_step = None
        app.state.model_loaded = False

    # Build chip registry from chip_configs_dir; fall back to bundled chips/
    # via ensure_artifact when the configured dir is missing.
    app.state.chip_registry: dict[str, ChipStructureLibrary] = {}

    configs_dir: Path | None = None
    if settings.chip_configs_dir:
        configs_dir = resolve_path(settings.chip_configs_dir)
        if not configs_dir.is_dir():
            configs_dir = None
    if configs_dir is None:
        try:
            # ensure_artifact returns a file path, so resolve sak.json then
            # use its parent dir to scan for any sibling chip configs.
            sak_json = ensure_artifact("chips/sak.json")
            configs_dir = sak_json.parent
        except Exception as exc:
            logger.warning("Could not obtain chip configs: %s", exc)
            configs_dir = None

    if configs_dir is not None and configs_dir.is_dir():
        for json_file in sorted(configs_dir.glob("*.json")):
            try:
                lib = ChipStructureLibrary.from_file(
                    json_file,
                    pixel_size=settings.default_pixel_size,
                )
                chip_key = json_file.stem.lower()
                app.state.chip_registry[chip_key] = lib
            except Exception:
                pass  # skip malformed configs

    # Load structure library: prefer chip config, then registry default, then legacy
    app.state.structure_library = None
    if settings.chip_config_path:
        chip_config_path = resolve_path(settings.chip_config_path)
        if chip_config_path.exists():
            app.state.structure_library = ChipStructureLibrary.from_file(
                chip_config_path,
                pixel_size=settings.default_pixel_size,
            )

    if app.state.structure_library is None and app.state.chip_registry:
        # Use first loaded chip config as default
        app.state.structure_library = next(iter(app.state.chip_registry.values()))

    if app.state.structure_library is None:
        structure_library_path = resolve_path(settings.structure_library_path)
        if not structure_library_path.exists():
            try:
                structure_library_path = ensure_artifact("chamber_structure.json")
            except Exception:
                structure_library_path = None
        if structure_library_path is not None and structure_library_path.exists():
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                app.state.structure_library = SAKRoIStructureLibrary(
                    lookup_path=str(structure_library_path),
                    pixel_size=settings.default_pixel_size,
                )

    # Detect GPU availability
    app.state.gpu_available = torch.cuda.is_available()
    if settings.device:
        app.state.device = settings.device
    elif app.state.gpu_available:
        app.state.device = "cuda:0"
    else:
        app.state.device = "cpu"

    # Load segmentation model (optional)
    app.state.segmenter = None
    app.state.segmenter_type = None
    if settings.segmenter:
        try:
            if settings.segmenter == "cellpose-sam":
                from acia.segm.processor.cellpose_sam import CellposeSAMSegmenter

                app.state.segmenter = CellposeSAMSegmenter()
            elif settings.segmenter == "omnipose":
                from acia.segm.processor.omnipose import OmniposeSegmenter

                app.state.segmenter = OmniposeSegmenter()
            app.state.segmenter_type = settings.segmenter
        except ImportError:
            pass  # segmenter stays None

    yield

    # Cleanup (if needed)
    app.state.detection_step = None
    app.state.structure_library = None
    app.state.chip_registry = {}
    app.state.segmenter = None
    app.state.segmenter_type = None


app = FastAPI(
    title="DART API",
    description="API for microfluidic chamber calibration and image processing",
    version="0.2.0",
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
        segmenter_loaded=getattr(app.state, "segmenter", None) is not None,
        segmenter=getattr(app.state, "segmenter_type", None),
    )


@app.get("/available-chips", response_model=list[str])
async def list_available_chips() -> list[str]:
    """List the names of all loaded chip configurations."""
    registry = getattr(app.state, "chip_registry", {})
    return sorted(registry.keys())


@app.get("/chamber-types", response_model=list[ChamberType])
async def list_chamber_types() -> list[ChamberType]:
    """List available chamber structure types and their ROI ID patterns."""
    structure_library = getattr(app.state, "structure_library", None)
    if structure_library is None:
        raise HTTPException(status_code=503, detail="Structure library not loaded")

    chamber_types = []
    for name in structure_library.polygon_library:
        chamber_types.append(ChamberType(name=name))

    return chamber_types


async def _process_image_from_array(
    img_array_hwc: np.ndarray,
    roi_id: str,
    pixel_size: float = 0.065789,
    structure_library_path: str | None = None,
    return_uncropped: bool = False,
    chip_config_path: str | None = None,
    chip_name: str | None = None,
) -> dict:
    """
    Core processing pipeline accepting numpy array.

    Args:
        img_array_hwc: HxWx3 numpy array (uint8)
        roi_id: ROI identifier
        pixel_size: Pixel size in microns
        structure_library_path: Optional custom structure library
        return_uncropped: Return full-size mask
        chip_config_path: Optional path to chip config JSON
        chip_name: Optional chip name to select from registry

    Returns:
        Dict with success, cropped_img, cropped_mask, rotation_angle,
        chamber_type, or error_message
    """
    # Check if detection model is loaded
    detection_step = getattr(app.state, "detection_step", None)
    if detection_step is None:
        return {
            "success": False,
            "error_message": "Model not loaded - check DART_MODEL_PATH configuration",
        }

    # Load structure library: chip_name → chip_config_path → structure_library_path → default
    structure_library = None
    if chip_name:
        registry = getattr(app.state, "chip_registry", {})
        key = chip_name.lower()
        if key in registry:
            structure_library = registry[key]
        else:
            available = sorted(registry.keys())
            return {
                "success": False,
                "error_message": f"Unknown chip '{chip_name}'. Available: {available}",
            }
    elif chip_config_path:
        ccp = resolve_path(chip_config_path)
        if not ccp.exists():
            return {
                "success": False,
                "error_message": f"Chip config not found: {chip_config_path}",
            }
        structure_library = ChipStructureLibrary.from_file(ccp, pixel_size=pixel_size)
    elif structure_library_path:
        lib_path = resolve_path(structure_library_path)
        if not lib_path.exists():
            return {
                "success": False,
                "error_message": f"Structure library not found: {structure_library_path}",
            }
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            structure_library = SAKRoIStructureLibrary(
                lookup_path=str(lib_path),
                pixel_size=pixel_size,
            )
    else:
        structure_library = getattr(app.state, "structure_library", None)

    if structure_library is None:
        return {
            "success": False,
            "error_message": "No structure library available (set chip_config_path or structure_library_path)",
        }

    # Get structure info for this ROI
    try:
        structure_name, roi_polygon, marker_group_configs = structure_library(roi_id)
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Invalid ROI ID '{roi_id}': {e}",
        }

    # Run the full pipeline
    # Step 1: Detection
    try:
        data = detection_step(img_array_hwc)
        markers = data.get("markers", [])
        if len(markers) < 2:
            return {
                "success": False,
                "error_message": f"Insufficient markers detected: {len(markers)} (need at least 2)",
            }
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Detection failed: {e}",
        }

    # Step 2: Matching
    try:
        matching_step = MarkerMatchingStep(marker_group_pixel=marker_group_configs, tolerance=60)
        data = matching_step(data)
        matched_indices = data.get("matched_marker_indices", [])
        if len(matched_indices) == 0:
            return {
                "success": False,
                "error_message": "No marker pairs could be matched",
            }
        rotation_angle = data.get("angle", 0.0)
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Matching failed: {e}",
        }

    # Step 3: Rotation
    try:
        rotation_step = ImageRotationStep()
        data = rotation_step(data)
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Rotation failed: {e}",
        }

    # Step 4: Masking
    try:
        masking_step = RoIMaskingStep(marker_group_configs, roi_polygon)
        data = masking_step(data)
        cropped_img = data["image"]
        cropped_mask = data["mask"]
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Masking failed: {e}",
        }

    return {
        "success": True,
        "cropped_img": cropped_img,
        "cropped_mask": cropped_mask,
        "rotation_angle": float(rotation_angle),
        "chamber_type": structure_name,
        "roi_id": roi_id,
        "pixel_size": pixel_size,
    }


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(request: ProcessImageRequest) -> ProcessImageResponse:
    """
    Process image from base64-encoded JSON request.

    This endpoint accepts a JSON request with a base64-encoded image and
    returns the cropped chamber image and mask as base64-encoded PNGs.

    Args:
        request: JSON request with base64 image and parameters

    Returns:
        Base64-encoded cropped image and mask with metadata
    """
    from dart_mlci.api.utils import array_to_base64_png, base64_to_array

    # Decode base64 to array
    try:
        img_array = base64_to_array(request.image)
    except Exception as e:
        return ProcessImageResponse(
            success=False,
            error_message=f"Failed to decode base64 image: {e}",
        )

    # Process image
    result = await _process_image_from_array(
        img_array,
        request.roi_id,
        request.pixel_size,
        request.structure_library_path,
        request.return_uncropped,
        chip_config_path=getattr(request, "chip_config_path", None),
        chip_name=getattr(request, "chip_name", None),
    )

    if not result["success"]:
        return ProcessImageResponse(
            success=False,
            error_message=result["error_message"],
        )

    # Encode outputs to base64
    return ProcessImageResponse(
        success=True,
        cropped_image=array_to_base64_png(result["cropped_img"]),
        mask=array_to_base64_png(result["cropped_mask"], is_mask=True),
        rotation_angle=result["rotation_angle"],
        chamber_type=result["chamber_type"],
    )


@app.post("/process-image-preview", response_class=HTMLResponse)
async def process_image_preview(request: ProcessImageRequest) -> HTMLResponse:
    """
    Process image and return HTML preview with side-by-side visualization.

    This endpoint provides a visual preview of the processing results,
    displaying the cropped image and mask side-by-side in an HTML page.

    Args:
        request: JSON request with base64 image and parameters

    Returns:
        HTML page with embedded base64 images
    """
    from dart_mlci.api.utils import array_to_base64_png, base64_to_array

    # Decode base64 to array
    try:
        img_array = base64_to_array(request.image)
    except Exception as e:
        result = {
            "success": False,
            "error_message": f"Failed to decode base64 image: {e}",
        }
    else:
        # Run the pipeline
        result = await _process_image_from_array(
            img_array,
            request.roi_id,
            request.pixel_size,
            request.structure_library_path,
            return_uncropped=False,
            chip_config_path=getattr(request, "chip_config_path", None),
            chip_name=getattr(request, "chip_name", None),
        )

    # Handle errors - return error HTML
    if not result["success"]:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DMC Masking - Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .error {{
            background: #fee;
            border: 2px solid #c00;
            color: #c00;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        h1 {{ color: #c00; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Processing Error</h1>
        <div class="error">
            <strong>Error:</strong> {result["error_message"]}
        </div>
        <p><a href="/docs">← Back to API Documentation</a></p>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=html_content, status_code=422)

    # Convert images to base64
    cropped_image_b64 = array_to_base64_png(result["cropped_img"])
    mask_b64 = array_to_base64_png(result["cropped_mask"], is_mask=True)

    # Generate success HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DMC Masking - Image Preview</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f9f9f9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
        }}
        .metadata p {{
            margin: 8px 0;
            font-size: 14px;
        }}
        .metadata strong {{
            display: inline-block;
            width: 150px;
            color: #555;
        }}
        .images {{
            display: flex;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .image-box {{
            flex: 1;
            min-width: 400px;
            background: #fafafa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .image-box h2 {{
            margin-top: 0;
            color: #444;
            font-size: 18px;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 8px;
        }}
        .image-box img {{
            max-width: 100%;
            border: 2px solid #ccc;
            border-radius: 3px;
            display: block;
            margin-top: 10px;
            background: white;
        }}
        .back-link {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .back-link a {{
            color: #0066cc;
            text-decoration: none;
            font-weight: bold;
        }}
        .back-link a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DMC Masking Result</h1>
        <div class="metadata">
            <p><strong>ROI ID:</strong> {result["roi_id"]}</p>
            <p><strong>Chamber Type:</strong> {result["chamber_type"]}</p>
            <p><strong>Rotation Angle:</strong> {result["rotation_angle"]:.2f}°</p>
            <p><strong>Pixel Size:</strong> {result["pixel_size"]:.6f} μm</p>
        </div>
        <div class="images">
            <div class="image-box">
                <h2>Cropped Image</h2>
                <img src="data:image/png;base64,{cropped_image_b64}" alt="Cropped Image">
            </div>
            <div class="image-box">
                <h2>Mask</h2>
                <img src="data:image/png;base64,{mask_b64}" alt="Mask">
            </div>
        </div>
        <div class="back-link">
            <a href="/docs">← Back to API Documentation</a>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.post("/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest) -> SegmentResponse:
    """
    Run instance segmentation on a cropped chamber image.

    Accepts the cropped image and mask from /process-image and returns
    a labeled instance segmentation mask using cellpose/omnipose via acia.

    Args:
        request: JSON request with base64 cropped image and chamber mask

    Returns:
        Base64-encoded 16-bit grayscale PNG with instance labels
    """
    from dart_mlci.api.utils import (
        array_to_base64_uint16_png,
        base64_to_array,
        base64_to_mask,
    )
    from dart_mlci.mask import filter_segmentation_by_mask

    segmenter = getattr(app.state, "segmenter", None)
    if segmenter is None:
        return SegmentResponse(
            success=False,
            error_message="Segmentation model not loaded. Set DART_SEGMENTER env var.",
        )

    # Decode inputs
    try:
        img_array = base64_to_array(request.image)
    except Exception as e:
        return SegmentResponse(
            success=False,
            error_message=f"Failed to decode base64 image: {e}",
        )

    try:
        chamber_mask = base64_to_mask(request.mask)
    except Exception as e:
        return SegmentResponse(
            success=False,
            error_message=f"Failed to decode base64 mask: {e}",
        )

    # Run segmentation
    try:
        from acia.segm.local import THWCSequenceSource

        height, width = img_array.shape[:2]
        segm_input = img_array[None, :, :, :].astype(np.uint8)  # 1xHxWxC
        source = THWCSequenceSource(segm_input)

        with torch.no_grad():
            seg_result = segmenter(source.to_channel(0))

        masks = seg_result.toMasks(height, width, binary_mask=False)
        labeled_mask = masks[0].astype(np.uint16)
    except Exception as e:
        return SegmentResponse(
            success=False,
            error_message=f"Segmentation failed: {e}",
        )

    # Filter by chamber mask
    try:
        labeled_mask = filter_segmentation_by_mask(
            labeled_mask,
            chamber_mask,
            threshold=request.filter_threshold,
            relabel=request.relabel,
        )
    except Exception as e:
        return SegmentResponse(
            success=False,
            error_message=f"Mask filtering failed: {e}",
        )

    return SegmentResponse(
        success=True,
        segmentation_mask=array_to_base64_uint16_png(labeled_mask),
        cell_count=int(labeled_mask.max()),
        total_cell_area=int(np.sum(labeled_mask > 0)),
    )


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_map_endpoint(request: CalibrateRequest) -> CalibrateResponse:
    """
    Calibrate microscope map from JSON with base64 images.

    This endpoint accepts a JSON request with multiple base64-encoded calibration
    images and their associated stage positions, then returns a calibrated map.

    Args:
        request: JSON request with base64 images and calibration config

    Returns:
        Calibrated CSV map and statistics
    """
    from dart_mlci.api.utils import base64_to_array
    from dart_mlci.calibration import CalibrationError, run_calibration

    # Validate number of images (Pydantic should already enforce min_length=3)
    n_images = len(request.calibration_images)
    if n_images < 3:
        return CalibrateResponse(
            success=False,
            error_message=f"At least 3 calibration images required, got {n_images}",
        )

    # Decode all images to numpy arrays (no temp files)
    images = []
    roi_ids = []
    stage_positions = []

    for i, img_data in enumerate(request.calibration_images):
        try:
            images.append(base64_to_array(img_data.image))
        except Exception as e:
            return CalibrateResponse(
                success=False,
                error_message=f"Failed to decode calibration image {i}: {e}",
            )
        roi_ids.append(img_data.roi_id)
        stage_positions.append(
            {
                "x": img_data.stage_position.x,
                "y": img_data.stage_position.y,
                "z": img_data.stage_position.z,
            }
        )

    # Resolve chip config from registry
    registry = getattr(app.state, "chip_registry", {})
    chip_key = request.chip_name.lower()
    if chip_key not in registry:
        available = sorted(registry.keys())
        return CalibrateResponse(
            success=False,
            error_message=f"Unknown chip '{request.chip_name}'. Available: {available}",
        )
    chip_lib = registry[chip_key]

    # Load blueprint map from chip config
    blueprint_map = chip_lib.get_blueprint_map()

    # Get detection step
    detection_step = getattr(app.state, "detection_step", None)
    if detection_step is None:
        return CalibrateResponse(
            success=False,
            error_message="Model not loaded - check DART_MODEL_PATH configuration",
        )

    # Run calibration directly with numpy arrays
    try:
        result = run_calibration(
            images=images,
            roi_ids=roi_ids,
            stage_positions=stage_positions,
            detection_step=detection_step,
            structure_library=chip_lib,
            blueprint_map=blueprint_map,
            pixel_size=request.pixel_size,
        )
    except (CalibrationError, ValueError) as e:
        # Extract per-image results from CalibrationError
        error_image_results = []
        raw_results = getattr(e, "image_results", None)
        if raw_results:
            for img_result in raw_results:
                pos = None
                if img_result.microscope_position is not None:
                    pos = img_result.microscope_position.tolist()
                error_image_results.append(
                    ImageResultInfo(
                        roi_id=img_result.roi_id,
                        success=img_result.success,
                        error_message=img_result.error_message,
                        microscope_position=pos,
                    )
                )
        return CalibrateResponse(
            success=False,
            error_message=f"Calibration failed: {e}",
            image_results=error_image_results if error_image_results else None,
        )
    except Exception as e:
        return CalibrateResponse(
            success=False,
            error_message=f"Calibration failed: {e}",
        )

    # Convert calibrated map to typed objects
    calibrated_map = []
    for roi_id, roi_pos in result.calibrated_map.roi_positions.items():
        x, y = roi_pos.position[:2]
        z = None
        if result.z_positions:
            z_val = result.z_positions.get(roi_id)
            if z_val is not None:
                z = round(float(z_val), 6)
        calibrated_map.append(
            CalibratedROIPosition(roi_id=roi_id, x=round(float(x), 6), y=round(float(y), 6), z=z)
        )

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
    statistics = CalibrationStatistics(
        rmse=float(result.transform_result.rmse),
        max_error=float(result.transform_result.max_error),
        n_points=len(result.measured_map.roi_positions),
        residuals=result.transform_result.residuals.tolist(),
    )

    return CalibrateResponse(
        success=True,
        calibrated_map=calibrated_map,
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

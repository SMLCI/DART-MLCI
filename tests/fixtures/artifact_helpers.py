"""Utilities for loading and working with real artifact images for testing."""

from pathlib import Path

import cv2
import numpy as np
import tifffile

import dart_mlci

# Artifact directory paths
ARTIFACTS_DIR = Path(dart_mlci.__file__).parent.parent / "artifacts"
SAK_DIR = ARTIFACTS_DIR / "images" / "sak"
IMAGE_STACK_PATH = ARTIFACTS_DIR / "images" / "image_stack.tif"


def load_sak_sequence(max_frames: int | None = None) -> tuple[list[np.ndarray], list[int]]:
    """Load SAK sequence from artifacts directory.

    The SAK sequence contains 10 frames (0000-0010) with frame 0002 missing.
    Images are high-resolution (2160x2560) real microscopy images with
    cross and circle markers.

    Args:
        max_frames: Maximum number of frames to load (None for all)

    Returns:
        Tuple of:
            - frames: List of numpy arrays in HWC format (uint8 or uint16)
            - frame_numbers: List of actual frame numbers [0, 1, 3, 4, ...]
    """
    if not SAK_DIR.exists():
        raise FileNotFoundError(f"SAK directory not found: {SAK_DIR}")

    # Available frame numbers (0002 is missing)
    available_frames = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

    if max_frames is not None:
        available_frames = available_frames[:max_frames]

    frames = []
    frame_numbers = []

    for frame_num in available_frames:
        # Try PNG first, then TIF
        png_path = SAK_DIR / f"{frame_num:04d}.png"
        tif_path = SAK_DIR / f"{frame_num:04d}.tif"

        if png_path.exists():
            # Load PNG with OpenCV (returns HWC in BGR)
            img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Ensure RGB format
                if img.ndim == 2:
                    # Grayscale - convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif tif_path.exists():
            # Load TIF with tifffile
            img = tifffile.imread(tif_path)
            # Ensure HWC format
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
                # CHW to HWC
                img = np.moveaxis(img, 0, -1)
        else:
            raise FileNotFoundError(f"Frame {frame_num:04d} not found in {SAK_DIR}")

        if img is None:
            raise ValueError(f"Failed to load frame {frame_num:04d}")

        frames.append(img)
        frame_numbers.append(frame_num)

    return frames, frame_numbers


def load_image_stack(
    channel: int = 0,
    max_frames: int | None = None,
    frame_step: int = 1,
) -> list[np.ndarray]:
    """Load frames from image_stack.tif artifact.

    The image stack is a 163-frame multi-channel time-lapse sequence (1.1GB).
    Each frame is 2160x2560 pixels.

    Args:
        channel: Channel index to load (0 or 1)
        max_frames: Maximum number of frames to load (None for all)
        frame_step: Step size for frame sampling (1 for consecutive, >1 for subsampling)

    Returns:
        List of numpy arrays in HWC format (uint16 or uint8)
    """
    if not IMAGE_STACK_PATH.exists():
        raise FileNotFoundError(f"Image stack not found: {IMAGE_STACK_PATH}")

    # Load with tifffile for efficient multi-page TIFF reading
    with tifffile.TiffFile(IMAGE_STACK_PATH) as tif:
        num_pages = len(tif.pages)

        # Determine frames to load
        if max_frames is None:
            frame_indices = list(range(0, num_pages, frame_step))
        else:
            frame_indices = list(range(0, min(max_frames * frame_step, num_pages), frame_step))

        frames = []
        for idx in frame_indices:
            # Load single page
            img = tif.pages[idx].asarray()

            # Extract channel if multi-channel
            if img.ndim == 3 and img.shape[0] > 1:
                # CHW format
                img = img[channel]

            # Ensure HWC format
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
                # CHW to HWC
                img = np.moveaxis(img, 0, -1)

            frames.append(img)

    return frames


def detect_markers_in_image(
    image: np.ndarray,
    model_path: Path | None = None,
) -> list[dict]:
    """Detect cross/circle markers in real image using existing detection pipeline.

    Args:
        image: Input image in HWC format (uint8 or uint16)
        model_path: Path to YOLO model (None uses default)

    Returns:
        List of detected markers with keys:
            - 'bbox_center': np.ndarray([x, y])
            - 'label': 'cross' or 'circle'
            - 'conf': confidence score
            - 'mask_center': np.ndarray([x, y]) (if available)
    """
    from dart_mlci import MarkerDetectionModel

    # Initialize detection model
    mdm = MarkerDetectionModel(model_path=model_path, verbose=False)

    # Ensure uint8 RGB format
    if image.dtype == np.uint16:
        # Normalize to uint8
        image_norm = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    else:
        image_norm = image

    # Detect markers
    markers = mdm.predict_markers(image_norm)

    return markers


def create_marker_group_from_detection(
    markers: list[dict],
    expected_labels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Create marker_group_pixel dict from detected markers.

    Args:
        markers: List of detected markers from detect_markers_in_image
        expected_labels: Expected marker labels (default: ['cross', 'circle'])

    Returns:
        Dict mapping marker labels to positions:
            {'cross': np.ndarray([x, y]), 'circle': np.ndarray([x, y])}
    """
    if expected_labels is None:
        expected_labels = ["cross", "circle"]

    marker_group = {}

    for label in expected_labels:
        # Find first marker with this label
        for marker in markers:
            if marker["label"] == label:
                # Use mask_center if available, otherwise bbox_center
                if "mask_center" in marker:
                    marker_group[label] = marker["mask_center"]
                else:
                    marker_group[label] = marker["bbox_center"]
                break

    return marker_group


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 range [0, 255].

    Args:
        image: Input image (uint8 or uint16)

    Returns:
        Image normalized to uint8
    """
    if image.dtype == np.uint8:
        return image
    elif image.dtype == np.uint16:
        # Normalize to [0, 255]
        return (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        # Assume [0, 1] range
        return (np.clip(image, 0, 1) * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

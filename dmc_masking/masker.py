"""High-level masking classes that compose the detection/matching/rotation/masking steps."""

from pathlib import Path

import numpy as np

from dmc_masking.constants import DEFAULT_MODEL_PATH
from dmc_masking.detection import MarkerDetectionModel
from dmc_masking.mask import RoIPolygon, apply_mask
from dmc_masking.match import match_markers
from dmc_masking.rotation import compute_marker_group_angles, rotate_image_and_markers
from dmc_masking.utils import normalize_image

from .mask import SingleRoIStructureLibrary


class RoIMasker:
    """Perform the complete masking pipeline on an image stack."""

    def __init__(
        self,
        model_path: Path | None = None,
        roi_polygon: RoIPolygon | None = None,
        marker_group_pixel: dict[str, np.ndarray] | None = None,
    ):
        """Create new masking instance.

        Args:
            model_path: Path to the YOLO model weights. If None, uses DEFAULT_MODEL_PATH.
            roi_polygon: Polygon information for the RoI shape.
            marker_group_pixel: Marker placement relative to the RoI shape in pixel coordinates.
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model_path = model_path
        self.roi_polygon = roi_polygon
        self.marker_group_pixel = marker_group_pixel

        self.detection_model = MarkerDetectionModel(self.model_path)

    def __call__(
        self,
        image_stack: np.ndarray,
        roi_polygon: RoIPolygon | None = None,
        marker_group_pixel=None,
        return_uncropped=False,
    ):
        """Run the full masking pipeline on an image stack.

        Args:
            image_stack: The raw image stack (TxCxHxW). First channel should be phase contrast.
            roi_polygon: Chamber shape polygon. If None, uses the instance default.
            marker_group_pixel: Marker group information in pixel coordinates. If None, uses the instance default.
            return_uncropped: If True, return the full-size image and mask without cropping.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cropped image stack and mask stack.
        """

        if roi_polygon is None:
            roi_polygon = self.roi_polygon

        if marker_group_pixel is None:
            marker_group_pixel = self.marker_group_pixel

        result_images = []
        result_masks = []

        for image in image_stack:
            ph_image = image[0]

            if ph_image.dtype == np.uint16:
                ph_image = normalize_image(ph_image)

            if len(ph_image.shape) == 2:
                ph_image = np.stack((ph_image,) * 3, axis=-1)

            # 1. detect markers

            markers = self.detection_model.predict_markers(ph_image)

            # 2. match markers

            matched_marker_indices = match_markers(
                markers, marker_group=marker_group_pixel, tolerance=60
            )

            # 3. compute angle
            angles = compute_marker_group_angles(
                markers, matched_marker_indices, marker_group_pixel
            )
            mean_angle = np.mean(angles)

            # 4. Rotate image

            rotated_image, rotated_markers = rotate_image_and_markers(image, markers, mean_angle)

            # 5. Apply mask
            cropped_image, cropped_mask = apply_mask(
                matched_marker_indices=matched_marker_indices,
                rotated_markers=rotated_markers,
                marker_group_pixels=marker_group_pixel,
                roi_polygon=roi_polygon,
                rotated_image=rotated_image,
                return_uncropped=return_uncropped,
            )

            result_images.append(cropped_image)
            result_masks.append(cropped_mask)

        # Homogenize cropped image sizes (should be not more than 1 pixel)
        shapes = np.stack([np.array(im.shape) for im in result_images], axis=0)
        max_height = np.max(shapes[:, 1])
        max_width = np.max(shapes[:, 2])

        for i, im in enumerate(result_images):
            im_height, im_width = im.shape[-2:]

            ph = max_height - im_height
            pw = max_width - im_width

            result_images[i] = np.pad(im, [(0, 0), (0, ph), (0, pw)])
            result_masks[i] = np.pad(result_masks[i], [(0, ph), (0, pw)])

        return np.stack(result_images, axis=0), np.stack(result_masks, axis=0)


def compute_marker_angles(markers, marker_group_pixel):
    """Compute the mean rotation angle from detected markers.

    Args:
        markers: List of detected marker dicts.
        marker_group_pixel: Expected marker positions in pixel coordinates.

    Returns:
        float: Mean rotation angle in degrees.
    """
    matched_marker_indices = match_markers(markers, marker_group=marker_group_pixel, tolerance=60)

    angles = compute_marker_group_angles(markers, matched_marker_indices, marker_group_pixel)
    mean_angle = np.mean(angles)

    return mean_angle


class SingleStructureRoIMasker:
    """Masker for a chip with a single structure type."""

    def __init__(
        self,
        model_path: Path | None = None,
        structure_library: Path | None = None,
        structure_name="OpenBox-inner",
        pixel_size: float = 0.065789,
    ):
        """Initialize a single-structure masker.

        Args:
            model_path: Path to the YOLO model. Defaults to None (uses DEFAULT_MODEL_PATH).
            structure_library: Path to the structure library JSON. Defaults to None.
            structure_name: Name of the structure. Defaults to "OpenBox-inner".
            pixel_size: Size of a pixel in micrometers. Defaults to 0.065789.
        """

        if structure_library is None:
            structure_library = Path(__file__).parent.parent / "artifacts/chamber_structure.json"
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.rm = RoIMasker(model_path=model_path, roi_polygon=None, marker_group_pixel=None)

        self.structure_library = SingleRoIStructureLibrary(
            lookup_path=structure_library,
            structure_name=structure_name,
            pixel_size=pixel_size,
        )

    def __call__(self, image_stack: np.ndarray, roi_id: str):
        """Mask the structures.

        Args:
            image_stack: TxCxHxW image stack.
            roi_id: The id of the roi.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cropped image (TxCxH*xW*) and mask (TxCxH*xW*).
        """

        _, rp, mgp = self.structure_library(roi_id)

        cropped_image, cropped_mask = self.rm(image_stack, rp, mgp)

        return cropped_image, cropped_mask

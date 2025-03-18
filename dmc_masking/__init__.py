"""Main masking functionality"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from dmc_masking.mask import RoIPolygon, apply_mask
from dmc_masking.match import match_markers
from dmc_masking.rotation import (
    compute_marker_group_angles,
    rotate_image,
    rotate_markers,
)
from dmc_masking.utils import normalize_image

from .utils import center_of_mask_mass


def extract_data(result, image: np.ndarray):
    """Extracts marker information from yolo detection results on the image

    Args:
        result (_type_): Yolo detection results
        image (np.ndarray): the original image

    Returns:
        _type_: marker information
    """

    boxes_data = result.boxes.cpu().numpy()

    data = []

    for marker_detection, label in zip(boxes_data.xywh, boxes_data.cls):
        x, y, _, _ = marker_detection

        data.append({"bbox_center": np.array((x, y)), "label": result.names[label]})

    if result.masks is not None:
        mask_data = result.masks.cpu().numpy()
        for i, mask in enumerate(mask_data.data):
            x, y = center_of_mask_mass(mask.astype(np.uint8))

            _, mask_width = mask.shape
            _, image_width = image.shape[:2]

            # correct for image scaling
            scale = image_width / mask_width
            x *= scale
            y *= scale

            data[i]["mask_center"] = np.array((x, y))
            data[i]["mask_size"] = np.sum(mask.astype(np.uint8)) * scale

    return data


class MarkerDetectionModel:
    """Yolo model for detecting the markers"""

    def __init__(self, model_path: Path):
        """

        Args:
            model_path (Path): path to the model pt
        """
        self.model = YOLO(model_path)

    def predict_markers(self, image: np.ndarray):
        """Predict markers on the image

        Args:
            image (np.ndarray): the recorded image

        Returns:
            _type_: marker information
        """
        result = self.model(image)[0]

        return extract_data(result, image)


class RoIMasker:
    """Performing the complete masking"""

    def __init__(
        self,
        model_path: Path,
        roi_polygon: RoIPolygon,
        marker_group_pixel: dict[str, np.ndarray],
    ):
        """Create new masking instance

        Args:
            model_path (Path): path to the yolo pt
            roi_polygon (RoIPolygon): polygon information for the RoI shape
            marker_group_pixel (dict[str, np.ndarray]): Information on the marker placement relative to the RoI shape
        """

        self.model_path = model_path
        self.roi_polygon = roi_polygon
        self.marker_group_pixel = marker_group_pixel

        self.detection_model = MarkerDetectionModel(self.model_path)

    def __call__(
        self,
        image: np.ndarray,
        roi_polygon: RoIPolygon = None,
        marker_group_pixel=None,
        return_uncropped=False,
    ):
        """_summary_

        Args:
            image (np.ndarray): the raw image
            roi_polygon (RoIPolygon, optional): chamber shape polygon. If None, the initial shape polygon is used. Defaults to None.
            marker_group_pixel (_type_, optional): marker group information in pixel coordinates. If None, the intial marker group information is used. Defaults to None.

        Returns:
            _type_: cropped image and cropped mask
        """

        if roi_polygon is None:
            roi_polygon = self.roi_polygon

        if marker_group_pixel is None:
            marker_group_pixel = self.marker_group_pixel

        if image.dtype == np.uint16:
            image = normalize_image(image)

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # 1. detect markers

        markers = self.detection_model.predict_markers(image)

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

        rotated_image = rotate_image(image, mean_angle)
        rotated_markers = rotate_markers(markers, image, mean_angle)

        # 5. Apply mask

        cropped_image, cropped_mask = apply_mask(
            matched_marker_indices=matched_marker_indices,
            rotated_markers=rotated_markers,
            marker_group_pixels=marker_group_pixel,
            roi_polygon=roi_polygon,
            rotated_image=rotated_image,
            return_uncropped=return_uncropped,
        )

        return cropped_image, cropped_mask


def compute_marker_angles(markers, marker_group_pixel):
    # 2. match markers

    matched_marker_indices = match_markers(
        markers, marker_group=marker_group_pixel, tolerance=60
    )

    # 3. compute angle
    angles = compute_marker_group_angles(
        markers, matched_marker_indices, marker_group_pixel
    )
    mean_angle = np.mean(angles)

    return mean_angle

"""Main masking functionality"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from dmc_masking.mask import RoIPolygon, apply_mask
from dmc_masking.match import match_markers
from dmc_masking.rotation import (
    compute_marker_group_angles,
    rotate_image_and_markers,
)
from dmc_masking.utils import normalize_image

from .mask import SingleRoIStructureLibrary
from .utils import center_of_mask_mass


def extract_data(result, image: np.ndarray, label_mapping: dict[str, str] | None = None):
    """Extracts marker information from yolo detection results on the image

    Args:
        result (_type_): Yolo detection results
        image (np.ndarray): the original image
        label_mapping: Optional dict mapping model class names to desired labels
                       e.g., {"class_0": "cross", "class_1": "circle"}

    Returns:
        _type_: marker information
    """

    boxes_data = result.boxes.cpu().numpy()

    data = []

    for marker_detection, label, conf in zip(
        boxes_data.xywh, boxes_data.cls, boxes_data.conf, strict=False
    ):
        x, y, _, _ = marker_detection
        raw_label = result.names[label]
        mapped_label = label_mapping.get(raw_label, raw_label) if label_mapping else raw_label

        data.append({"bbox_center": np.array((x, y)), "label": mapped_label, "conf": conf})

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

    def __init__(
        self, model_path: Path, verbose=False, label_mapping: dict[str, str] | None = None
    ):
        """

        Args:
            model_path (Path): path to the model pt
            verbose: whether to print verbose output
            label_mapping: Optional dict mapping model class names to desired labels
                           e.g., {"class_0": "cross", "class_1": "circle"}
        """
        self.model = YOLO(model_path, verbose=verbose)
        self.label_mapping = label_mapping

    def predict_markers(self, image: np.ndarray):
        """Predict markers on the image

        Args:
            image (np.ndarray): the recorded image

        Returns:
            _type_: marker information
        """
        result = self.model(image)[0]

        return extract_data(result, image, self.label_mapping)


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
        image_stack: np.ndarray,
        roi_polygon: RoIPolygon = None,
        marker_group_pixel=None,
        return_uncropped=False,
    ):
        """_summary_

        Args:
            image (np.ndarray): the raw image (TxCxHxW). First channel should be phase contrast.
            roi_polygon (RoIPolygon, optional): chamber shape polygon. If None, the initial shape polygon is used. Defaults to None.
            marker_group_pixel (_type_, optional): marker group information in pixel coordinates. If None, the intial marker group information is used. Defaults to None.

        Returns:
            _type_: cropped image and cropped mask
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
    # 2. match markers

    matched_marker_indices = match_markers(markers, marker_group=marker_group_pixel, tolerance=60)

    # 3. compute angle
    angles = compute_marker_group_angles(markers, matched_marker_indices, marker_group_pixel)
    mean_angle = np.mean(angles)

    return mean_angle


class SingleStructureRoIMasker:
    """Masker for a chip with a single structure"""

    def __init__(
        self,
        model_path: Path | None = None,
        structure_library: Path | None = None,
        structure_name="OpenBox-inner",
        pixel_size: float = 0.065789,
    ):
        """_summary_

        Args:
            model_path (Path, optional): Path to the yolo model. Defaults to None.
            structure_library (Path, optional): Path to the structure library. Defaults to None.
            structure_name (str, optional): Name of the structure. Defaults to "OpenBox-inner".
            pixel_size (float, optional): size of a pixel in micrometer. Defaults to 0.065789.
        """

        if structure_library is None:
            structure_library = Path(__file__).parent.parent / "artifacts/chamber_structure.json"
        if model_path is None:
            model_path = Path(__file__).parent.parent / "artifacts/models/best34.pt"

        self.rm = RoIMasker(model_path=model_path, roi_polygon=None, marker_group_pixel=None)

        self.structure_library = SingleRoIStructureLibrary(
            lookup_path=structure_library,
            structure_name=structure_name,
            pixel_size=pixel_size,
        )

    def __call__(self, image_stack: np.ndarray, roi_id: str):
        """mask the structures

        Args:
            image_stack (np.ndarray): TxCxHxW image stack
            roi_id (str): the id of the roi

        Returns:
            tuple[np.ndarray, np.ndarray]: Cropped image (TxCxH*xW*) and mask (TxCxH*xW*)
        """

        _, rp, mgp = self.structure_library(roi_id)

        cropped_image, cropped_mask = self.rm(image_stack, rp, mgp)

        return cropped_image, cropped_mask


class MarkerDetectionStep:
    """Detect Markers"""

    def __init__(self, model_path: str):
        self.mdm = MarkerDetectionModel(model_path)
        self.mdm.model.conf = 0.6

    def __call__(self, image):
        markers = self.mdm.predict_markers(image)

        data = {"image": image, "markers": markers}

        return data


class MarkerMatchingStep:
    """Match markers into pairs"""

    def __init__(self, marker_group_pixel, tolerance=60):
        self.marker_group_pixel = marker_group_pixel
        self.tolerance = tolerance

    def __call__(self, data):
        markers = data["markers"]

        matched_marker_indices = match_markers(
            markers, marker_group=self.marker_group_pixel, tolerance=self.tolerance
        )

        data["matched_marker_indices"] = matched_marker_indices

        angles = compute_marker_group_angles(
            markers, matched_marker_indices, self.marker_group_pixel
        )
        mean_angle = np.mean(angles)

        data["angle"] = mean_angle

        return data


class ImageRotationStep:
    """Rotate images and markers"""

    def __call__(self, data):
        markers = data["markers"]
        mean_angle = data["angle"]
        image = data["image"]

        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])

        rotated_image, rotated_markers = rotate_image_and_markers(image, markers, mean_angle)

        rotated_image = np.moveaxis(rotated_image, [0, 1, 2], [2, 0, 1])

        data["image"] = rotated_image
        data["markers"] = rotated_markers

        return data


class RoIMaskingStep:
    """Masking RoI"""

    def __init__(self, marker_group_pixels, roi_polygon):
        super().__init__()

        self.marker_group_pixels = marker_group_pixels
        self.roi_polygon = roi_polygon

    def __call__(self, data, cropped=True):
        image = np.moveaxis(data["image"], [0, 1, 2], [1, 2, 0])

        cropped_image, cropped_mask = apply_mask(
            matched_marker_indices=data["matched_marker_indices"],
            rotated_markers=data["markers"],
            marker_group_pixels=self.marker_group_pixels,
            roi_polygon=self.roi_polygon,
            rotated_image=image,
            return_uncropped=not cropped,
        )

        cropped_image = np.moveaxis(cropped_image, [0, 1, 2], [2, 0, 1])

        data["image"] = cropped_image
        data["mask"] = cropped_mask

        return data

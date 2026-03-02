"""Marker detection using YOLO models."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from dmc_masking.constants import DEFAULT_MODEL_PATH
from dmc_masking.utils import center_of_mask_mass


def extract_data(result, image: np.ndarray, label_mapping: dict[str, str] | None = None):
    """Extract marker information from YOLO detection results.

    Args:
        result: YOLO detection result object
        image: The original image (HxWxC)
        label_mapping: Optional dict mapping model class names to desired labels
                       e.g., {"class_0": "cross", "class_1": "circle"}

    Returns:
        list[dict]: List of marker dicts with keys 'bbox_center', 'label', 'conf',
                    and optionally 'mask_center' and 'mask_size'.
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
    """YOLO model for detecting markers on microfluidic chip images."""

    def __init__(
        self,
        model_path: Path | None = None,
        verbose=False,
        label_mapping: dict[str, str] | None = None,
        device: str | None = None,
    ):
        """
        Args:
            model_path: Path to the YOLO model weights. If None, uses DEFAULT_MODEL_PATH.
            verbose: Whether to print verbose YOLO output.
            label_mapping: Optional dict mapping model class names to desired labels
                           e.g., {"class_0": "cross", "class_1": "circle"}
            device: Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). None for auto.
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model = YOLO(model_path, verbose=verbose)
        self.label_mapping = label_mapping
        self.device = device
        self.verbose = verbose

    def predict_markers(self, image: np.ndarray):
        """Predict markers on the image.

        Args:
            image: The recorded image in HxWxC format.

        Returns:
            list[dict]: Detected marker information.
        """
        result = self.model(image, device=self.device, verbose=self.verbose)[0]
        return extract_data(result, image, self.label_mapping)

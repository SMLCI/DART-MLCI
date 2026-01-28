"""Testcases for single-cell property extractors"""

import unittest

import numpy as np

from dmc_masking import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dmc_masking.utils import load_tiff


class TestMarkerDetection(unittest.TestCase):
    """Test cases for single-cell property extractors"""

    def test_marker_detection(self):
        """test marker detection using the yolo model"""

        # 1. Load yolo model

        model = MarkerDetectionModel(DEFAULT_MODEL_PATH)

        # 2. Load image
        image = load_tiff("./artifacts/images/00150d6e-cecf-48f9-8b2d-a37a687ec0db.tif")

        image = np.stack((image,) * 3, axis=-1)

        # 3. Detect markers
        markers = model.predict_markers(image)

        marker_gt = [
            {
                "bbox_center": np.array([201.58, 530.41], dtype=float),
                "label": "cross",
                "mask_center": np.array([203, 539]),
                "mask_size": np.float64(5578.0),
            },
            {
                "bbox_center": np.array([1864.3, 610.91], dtype=float),
                "label": "cross",
                "mask_center": np.array([1867, 623]),
                "mask_size": np.float64(5588.0),
            },
            {
                "bbox_center": np.array([170.09, 1317.2], dtype=float),
                "label": "circle",
                "mask_center": np.array([171, 1327]),
                "mask_size": np.float64(8290.0),
            },
            {
                "bbox_center": np.array([1829.3, 1391.6], dtype=float),
                "label": "circle",
                "mask_center": np.array([1831, 1403]),
                "mask_size": np.float64(8210.0),
            },
        ]

        pixel_accuracy = 5

        for pred_marker, gt_marker in zip(markers, marker_gt, strict=False):
            self.assertLess(
                np.linalg.norm(pred_marker["bbox_center"] - gt_marker["bbox_center"]),
                pixel_accuracy,
            )
            self.assertLess(
                np.linalg.norm(pred_marker["mask_center"] - gt_marker["mask_center"]),
                pixel_accuracy,
            )
            self.assertLess(np.abs(pred_marker["mask_size"] - gt_marker["mask_size"]), 100)
            self.assertEqual(pred_marker["label"], gt_marker["label"])


if __name__ == "__main__":
    unittest.main()

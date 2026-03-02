"""Testcases for marker detection."""

import unittest

import numpy as np

from dmc_masking import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dmc_masking.io import load_image


def _sort_markers(markers):
    """Sort markers by label then x-position for stable comparison."""
    return sorted(markers, key=lambda m: (m["label"], m["bbox_center"][0]))


class TestMarkerDetection(unittest.TestCase):
    """Test cases for marker detection using YOLO."""

    def test_marker_detection(self):
        """Test marker detection using the YOLO model."""

        # 1. Load yolo model
        model = MarkerDetectionModel(DEFAULT_MODEL_PATH)

        # 2. Load image
        image = load_image("./artifacts/images/00150d6e-cecf-48f9-8b2d-a37a687ec0db.tif")

        # 3. Detect markers
        markers = model.predict_markers(image)

        # Ground truth — bbox_center only (detection model, no masks)
        marker_gt = [
            {"bbox_center": np.array([170.09, 1317.2]), "label": "circle"},
            {"bbox_center": np.array([1829.3, 1391.6]), "label": "circle"},
            {"bbox_center": np.array([201.58, 530.41]), "label": "cross"},
            {"bbox_center": np.array([1864.3, 610.91]), "label": "cross"},
        ]

        self.assertEqual(len(markers), len(marker_gt))

        # Sort both by label + x-position for stable comparison
        markers_sorted = _sort_markers(markers)
        gt_sorted = _sort_markers(marker_gt)

        pixel_accuracy = 5

        for pred_marker, gt_marker in zip(markers_sorted, gt_sorted, strict=False):
            self.assertEqual(pred_marker["label"], gt_marker["label"])
            self.assertLess(
                np.linalg.norm(pred_marker["bbox_center"] - gt_marker["bbox_center"]),
                pixel_accuracy,
            )


if __name__ == "__main__":
    unittest.main()

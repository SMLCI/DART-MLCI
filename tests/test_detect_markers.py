"""Testcases for marker detection."""

import unittest

import numpy as np

from dart_mlci import DEFAULT_MODEL_PATH, MarkerDetectionModel
from dart_mlci.io import load_image


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

        # Ground truth — bbox_center only (detection model, no masks).
        # The four real markers are stable; YOLO may pick up 0-2 extra spurious
        # detections depending on hardware / dep versions. Accept the range and
        # only assert the ground truth markers are *present*.
        marker_gt = [
            {"bbox_center": np.array([170.09, 1317.2]), "label": "circle"},
            {"bbox_center": np.array([1829.3, 1391.6]), "label": "circle"},
            {"bbox_center": np.array([201.58, 530.41]), "label": "cross"},
            {"bbox_center": np.array([1864.3, 610.91]), "label": "cross"},
        ]

        self.assertGreaterEqual(len(markers), 4)
        self.assertLessEqual(len(markers), 6)

        pixel_accuracy = 5

        # Every ground truth marker must be matched by some detected marker
        # with the same label within pixel_accuracy. Extra detections are
        # tolerated (the matching/filtering steps downstream handle them).
        for gt_marker in marker_gt:
            same_label = [m for m in markers if m["label"] == gt_marker["label"]]
            distances = [
                np.linalg.norm(m["bbox_center"] - gt_marker["bbox_center"]) for m in same_label
            ]
            self.assertTrue(
                distances and min(distances) < pixel_accuracy,
                f"No {gt_marker['label']} detected within {pixel_accuracy}px of "
                f"{gt_marker['bbox_center'].tolist()} (closest: "
                f"{min(distances) if distances else 'n/a'})",
            )


if __name__ == "__main__":
    unittest.main()

"""Testcases for marker matching."""

import unittest

import numpy as np

from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers


class TestMatch(unittest.TestCase):
    """Test maker matching"""

    def test_match(self):
        """test marker matching"""

        pixel_size = 0.065789

        marker_group = {
            "cross": np.array((0, -4), dtype=float),
            "circle": np.array((52, -4), dtype=float),
        }

        marker_group_pixel = marker_group_to_pixel_coordinates(marker_group, pixel_size)

        markers = [
            {"label": "cross", "bbox_center": np.array([0, 0])},
            {"label": "circle", "bbox_center": np.array([52 * (1.0 / pixel_size), 0])},
        ]

        matched_indices = match_markers(markers, marker_group_pixel)

        # we have one match
        self.assertEqual(len(matched_indices), 1)

        # with specific indices
        self.assertEqual(matched_indices[0][0], 0)
        self.assertEqual(matched_indices[0][1], 1)

        # add some distance such that matching does not tolerate
        markers = [
            {"label": "cross", "bbox_center": np.array([0, 0])},
            {"label": "circle", "bbox_center": np.array([53 * (1.0 / pixel_size), 0])},
        ]

        matched_indices = match_markers(markers, marker_group_pixel)

        self.assertEqual(len(matched_indices), 0)

        # increase the tolerance such that it works again
        matched_indices = match_markers(markers, marker_group_pixel, tolerance=20)

        self.assertEqual(len(matched_indices), 1)

        # add some distance such that matching does not tolerate
        markers = [
            {"label": "cross", "bbox_center": np.array([0, 0])},
            {"label": "circle", "bbox_center": np.array([52 * (1.0 / pixel_size), 0])},
            {"label": "cross", "bbox_center": np.array([0, -400])},
            {
                "label": "circle",
                "bbox_center": np.array([52 * (1.0 / pixel_size), -400]),
            },
        ]

        # increase the tolerance such that it works again
        matched_indices = match_markers(markers, marker_group_pixel, tolerance=20)

        self.assertEqual(len(matched_indices), 2)

        # with specific indices
        self.assertEqual(matched_indices[0][0], 0)
        self.assertEqual(matched_indices[0][1], 1)
        self.assertEqual(matched_indices[1][0], 2)
        self.assertEqual(matched_indices[1][1], 3)


if __name__ == "__main__":
    unittest.main()

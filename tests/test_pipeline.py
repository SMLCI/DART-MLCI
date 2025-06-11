""" Testcases for full masking pipeline """

import unittest
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

import dmc_masking
from dmc_masking import (
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
)
from dmc_masking.mask import RoIPolygon
from dmc_masking.match import marker_group_to_pixel_coordinates
from dmc_masking.utils import plot_marker_paris, plot_markers


def build_polygon(pixel_size):
    chamber_width = 60
    chamber_height = 60

    interior = Point(30, 30).buffer(5)

    chamber_polygon = Polygon(
        [
            (0, 0),
            (chamber_width, 0),
            (chamber_width, chamber_height),
            (0, chamber_height),
        ]
    ).difference(interior)

    rp = RoIPolygon(chamber_polygon)

    rp = rp.scale(1.0 / pixel_size)
    xmin, ymin, _, _ = rp.roi_polygon.bounds

    # move polygon into positive coordinates
    rp = rp.translate(x=-xmin, y=-ymin)

    return rp


class TestFullPipeline(unittest.TestCase):
    """Test cases for full masking pipeline"""

    @staticmethod
    def test_all_pipeline_steps():
        """test the full pipeline step-by-step"""

        # config
        pixel_size = 0.065789
        marker_group = {
            "cross": np.array((4, 8), dtype=float),
            "circle": np.array((56, 8), dtype=float),
        }

        marker_group_pixels = marker_group_to_pixel_coordinates(
            marker_group, pixel_size
        )

        roi_polygon = build_polygon(pixel_size)

        step1 = MarkerDetectionStep(
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"
        )
        step2 = MarkerMatchingStep(marker_group_pixels, tolerance=60)
        step3 = ImageRotationStep()
        step4 = RoIMaskingStep(marker_group_pixels, roi_polygon)

        image = cv2.imread(
            Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0000.png"
        )

        # detect markers
        data_res_1 = step1(image)

        plot_markers(image, data_res_1["markers"])
        plt.savefig("test_pp_step_1.png")

        # match markers
        data_res_2 = step2(data_res_1)

        matched_marker_indices = data_res_2["matched_marker_indices"]
        markers = data_res_2["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig("test_pp_step_2.png")

        # rotate image
        data_res_3 = step3(data_res_2)

        image = data_res_3["image"]
        matched_marker_indices = data_res_3["matched_marker_indices"]
        markers = data_res_3["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig("test_pp_step_3.png")

        # apply mask
        data_res_4 = step4(data_res_3)

        plt.figure()
        plt.imshow(data_res_4["image"], cmap="gray")
        plt.savefig("test_pp_step_4.png")

        # data_res = step4(step3(step2(step1(image))))
        print(data_res_4)


if __name__ == "__main__":
    unittest.main()

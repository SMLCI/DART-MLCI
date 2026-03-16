"""Testcases for full masking pipeline"""

import unittest
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

import dart_mlci
from dart_mlci import (
    DEFAULT_MODEL_PATH,
    ImageRotationStep,
    MarkerDetectionStep,
    MarkerMatchingStep,
    RoIMaskingStep,
    SingleRoIStructureLibrary,
)
from dart_mlci.mask import RoIPolygon
from dart_mlci.visualization import plot_marker_paris, plot_markers

# Dedicated folder for test results
TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


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
        # Create subfolder for this test
        output_dir = TEST_RESULTS_DIR / "pipeline_steps"
        output_dir.mkdir(exist_ok=True)

        # config
        pixel_size = 0.065789

        # get the RoI structure information
        srsl = SingleRoIStructureLibrary(
            lookup_path=Path(dart_mlci.__file__).parent.parent / "artifacts/chamber_structure.json",
            structure_name="NormaleBox-inner",
            pixel_size=pixel_size,
        )

        _, roi_polygon, marker_group_pixels = srsl("0000")

        # build the pipeline
        step1 = MarkerDetectionStep(DEFAULT_MODEL_PATH)
        step2 = MarkerMatchingStep(marker_group_pixels, tolerance=60)
        step3 = ImageRotationStep()
        step4 = RoIMaskingStep(marker_group_pixels, roi_polygon)

        image = cv2.imread(Path(dart_mlci.__file__).parent.parent / "artifacts/images/sak/0000.png")

        ### Go through the pipeline steps

        # detect markers
        data_res_1 = step1(image)

        plot_markers(image, data_res_1["markers"])
        plt.savefig(output_dir / "step_1_detection.png")

        # match markers
        data_res_2 = step2(data_res_1)

        matched_marker_indices = data_res_2["matched_marker_indices"]
        markers = data_res_2["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig(output_dir / "step_2_matching.png")

        # rotate image
        data_res_3 = step3(data_res_2)

        image = data_res_3["image"]
        matched_marker_indices = data_res_3["matched_marker_indices"]
        markers = data_res_3["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig(output_dir / "step_3_rotation.png")

        # apply mask
        data_res_4 = step4(data_res_3)

        plt.figure()
        plt.imshow(data_res_4["image"], cmap="gray")
        plt.savefig(output_dir / "step_4_masking.png")

        # data_res = step4(step3(step2(step1(image))))
        print(data_res_4)

    @staticmethod
    def test_all_pipeline_steps_bright():
        """test the full pipeline step-by-step"""
        # Create subfolder for this test
        output_dir = TEST_RESULTS_DIR / "pipeline_steps_bright"
        output_dir.mkdir(exist_ok=True)

        # config
        pixel_size = 0.07220  # micrometer / pixel

        # get the RoI structure information
        srsl = SingleRoIStructureLibrary(
            lookup_path=Path(dart_mlci.__file__).parent.parent / "artifacts/chamber_structure.json",
            structure_name="OpenBox-inner",
            pixel_size=pixel_size,
        )

        _, roi_polygon, marker_group_pixels = srsl("0000")

        # build the pipeline (bright images have lower detection confidence)
        step1 = MarkerDetectionStep(DEFAULT_MODEL_PATH, conf_threshold=0.3)
        step2 = MarkerMatchingStep(marker_group_pixels, tolerance=60)
        step3 = ImageRotationStep()
        step4 = RoIMaskingStep(marker_group_pixels, roi_polygon)

        image = cv2.imread(
            Path(dart_mlci.__file__).parent.parent / "artifacts/images/bright/bright_chamber.png"
        )

        ### Go through the pipeline steps

        # detect markers
        data_res_1 = step1(image)

        plot_markers(image, data_res_1["markers"])
        plt.savefig(output_dir / "step_1_detection.png")

        # match markers
        data_res_2 = step2(data_res_1)

        matched_marker_indices = data_res_2["matched_marker_indices"]
        markers = data_res_2["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig(output_dir / "step_2_matching.png")

        # rotate image
        data_res_2["angle"] *= -1
        data_res_3 = step3(data_res_2)

        image = data_res_3["image"]
        matched_marker_indices = data_res_3["matched_marker_indices"]
        markers = data_res_3["markers"]

        plot_marker_paris(image, matched_marker_indices, markers)
        plt.savefig(output_dir / "step_3_rotation.png")

        # apply mask
        data_res_4 = step4(data_res_3)

        plt.figure()
        plt.imshow(data_res_4["image"], cmap="gray")
        plt.savefig(output_dir / "step_4_masking.png")

        # data_res = step4(step3(step2(step1(image))))
        print(data_res_4)


if __name__ == "__main__":
    unittest.main()

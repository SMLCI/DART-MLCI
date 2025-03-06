""" Testcases for full masking pipeline """

import unittest

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from shapely.geometry import Point, Polygon

from dmc_masking import MarkerDetectionModel, RoIMasker
from dmc_masking.mask import RoIPolygon
from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers
from dmc_masking.rotation import (
    compute_marker_group_angles,
    rotate_image,
    rotate_markers,
)
from dmc_masking.utils import load_tiff


def build_polygon():
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

    return RoIPolygon(chamber_polygon)


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

        roi_polygon = build_polygon().scale(1.0 / pixel_size)
        xmin, ymin, xmax, ymax = roi_polygon.roi_polygon.bounds

        # move polygon into positive coordinates
        roi_polygon = roi_polygon.translate(x=-xmin, y=-ymin)

        # 1. Load yolo model

        model = MarkerDetectionModel("./artifacts/models/best34.pt")

        # 2. Load image
        image = load_tiff("./artifacts/images/00150d6e-cecf-48f9-8b2d-a37a687ec0db.tif")

        image = np.stack((image,) * 3, axis=-1)

        # 3. Detect markers
        markers = model.predict_markers(image)

        print(markers)

        marker_group_pixels = marker_group_to_pixel_coordinates(
            marker_group, pixel_size
        )

        # 4. Match markers
        matched_marker_indices = match_markers(
            markers, marker_group=marker_group_pixels, tolerance=60
        )

        print(matched_marker_indices)

        # 5. Compute angle
        angles = compute_marker_group_angles(
            markers, matched_marker_indices, marker_group_pixels
        )
        mean_angle = np.mean(angles)

        print("angles", angles)

        # 6. Rotate image

        rotated_image = rotate_image(image, mean_angle)
        rotated_markers = rotate_markers(markers, image, mean_angle)

        # 7. Apply mask

        masks = []
        polygons = []
        im_height, im_width = rotated_image.shape[:2]

        for cross_index, circle_index in matched_marker_indices:

            cross_marker = rotated_markers[cross_index]
            circle_marker = rotated_markers[circle_index]

            print(cross_marker["bbox_center"][0])

            # correct for difference in expected width
            width = np.abs(
                cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0]
            )
            expected_width = np.abs(
                marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0]
            )
            diff = width - expected_width

            # translate roi polygon
            rp = roi_polygon.translate(
                x=cross_marker["bbox_center"][0]
                - marker_group_pixels["cross"][0]
                + diff,
                y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1],
            )

            # check whether roi polygon in image
            xmin, ymin, xmax, ymax = rp.roi_polygon.bounds

            if xmin < 0 or xmax > im_width or ymin < 0 or ymax > im_height:
                # roi is out of image bounds
                continue

            polygons.append(rp)
            masks.append(~rp.to_mask(height=im_height, width=im_width).astype(bool))
            break

        if len(masks) == 0:
            raise ValueError("No roi lies completely inside the image")

        mask = masks[
            0
        ]  # np.bitwise_or.reduce(np.stack(masks, axis=-1).astype(bool), axis=-1)
        polygon: RoIPolygon = polygons[0]

        # 8. Cropping

        minx, miny, maxx, maxy = tuple(
            map(int, map(np.round, polygon.roi_polygon.bounds))
        )
        cropped_image = rotated_image[miny:maxy, minx:maxx]
        cropped_mask = mask[miny:maxy, minx:maxx]

        _, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].imshow(rotated_image)
        axes[0].imshow(mask, alpha=0.5)

        axes[1].imshow(cropped_image)
        axes[1].imshow(cropped_mask, alpha=0.2)
        plt.savefig("test.jpg")

    @staticmethod
    def test_roi_masker():
        """testing the roi masker class."""

        # general information
        pixel_size = 0.065789
        marker_group = {
            "cross": np.array((4, 8), dtype=float),
            "circle": np.array((56, 8), dtype=float),
        }

        # convert info to pixel coordinates
        marker_group_pixel = marker_group_to_pixel_coordinates(marker_group, pixel_size)
        roi_polygon = build_polygon().scale(1.0 / pixel_size)

        # create the masker
        rm = RoIMasker(
            model_path="./artifacts/models/best34.pt",
            roi_polygon=roi_polygon,
            marker_group_pixel=marker_group_pixel,
        )

        # load the image
        image = tifffile.imread(
            "./artifacts/images/00150d6e-cecf-48f9-8b2d-a37a687ec0db.tif"
        )

        # apply the masker
        cropped_image, cropped_mask = rm(image)

        # plot
        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_image)
        plt.imshow(cropped_mask, alpha=0.25)

        plt.savefig("test2.jpg")


if __name__ == "__main__":
    unittest.main()

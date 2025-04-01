""" Testcases for full masking pipeline """

import unittest
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, shape
from tqdm.auto import tqdm

import dmc_masking
from dmc_masking import MarkerDetectionModel, RoIMasker, SingleStructureRoIMasker
from dmc_masking.io import load_roi_structures
from dmc_masking.mask import RoIPolygon, SAKRoIStructureLibrary
from dmc_masking.match import marker_group_to_pixel_coordinates, match_markers
from dmc_masking.rotation import (
    compute_marker_group_angles,
    rotate_image_and_markers,
)


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

        # 1. Load yolo model

        model = MarkerDetectionModel(
            Path(dmc_masking.__file__).parent.parent / "artifacts/models/best34.pt"
        )  # "./artifacts/models/best34.pt")

        # 2. Load image
        image = cv2.imread(
            Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0007.png"
        )

        # 3. Detect markers
        markers = model.predict_markers(image)

        print(markers)

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

        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])

        # 6. Rotate image

        rotated_image, rotated_markers = rotate_image_and_markers(
            image, markers, mean_angle
        )

        # rotated_image = np.stack([rotate_image(im, mean_angle) for im in image], axis=0)
        # rotated_markers = rotate_markers(markers, image, mean_angle)

        # 7. Apply mask

        masks = []
        polygons = []
        im_height, im_width = rotated_image.shape[-2:]

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
        cropped_image = rotated_image[..., miny:maxy, minx:maxx]
        cropped_mask = mask[miny:maxy, minx:maxx]

        _, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].imshow(np.moveaxis(rotated_image, [0, 1, 2], [2, 0, 1]))
        axes[0].imshow(mask, alpha=0.5)

        axes[1].imshow(np.moveaxis(cropped_image, [0, 1, 2], [2, 0, 1]))
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
        roi_polygon = build_polygon(pixel_size)

        # create the masker
        rm = RoIMasker(
            # model_path="./artifacts/models/best34.pt",
            model_path=Path(dmc_masking.__file__).parent.parent
            / "artifacts/models/best34.pt",
            roi_polygon=roi_polygon,
            marker_group_pixel=marker_group_pixel,
        )

        image = cv2.imread(
            Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0007.png"
        )

        # apply the masker
        cropped_image, cropped_mask = rm(np.moveaxis(image[None], [1, 2, 3], [2, 3, 1]))

        # plot
        _, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(np.moveaxis(cropped_image[0], [0, 1, 2], [2, 0, 1]))
        axes[0].imshow(cropped_mask[0], alpha=0.25)
        axes[1].imshow(image)

        plt.savefig("test2.jpg")

    @staticmethod
    def test_roi_masker_sak():
        """testing the roi masker class."""

        # general information
        pixel_size = 0.065789

        # pylint: disable=too-many-function-args
        sakl = SAKRoIStructureLibrary(
            Path(dmc_masking.__file__).parent.parent
            / "artifacts/chamber_structure.json",
            pixel_size,
        )

        # create the masker
        rm = RoIMasker(
            # model_path="./artifacts/models/best34.pt",
            model_path=Path(dmc_masking.__file__).parent.parent
            / "artifacts/models/best34.pt",
            roi_polygon=None,
            marker_group_pixel=None,
        )

        image = cv2.imread(
            Path(dmc_masking.__file__).parent.parent / "artifacts/images/sak/0007.png"
        )

        print(image.shape)

        # make it TxCxHxW
        image = np.moveaxis(image[None, ...], [1, 2, 3], [2, 3, 1])

        # pylint: disable=unbalanced-tuple-unpacking
        _, sp, sc = sakl("0000")

        # apply the masker
        cropped_image, cropped_mask = rm(image, roi_polygon=sp, marker_group_pixel=sc)

        # plot
        _, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(np.moveaxis(cropped_image[0], [0, 1, 2], [2, 0, 1]))
        axes[0].imshow(cropped_mask[0], alpha=0.25)
        axes[1].imshow(np.moveaxis(image[0], [0, 1, 2], [2, 0, 1]))

        plt.savefig("test_rm2.jpg")

    @staticmethod
    def test_sak_chip():
        configs = [
            {
                "file_name": "0000.png",
                "chamber_type": "NormaleBox-pillar-inner",
            },
            {
                "file_name": "0001.png",
                "chamber_type": "BigBox-pillar-inner",
            },
            {
                "file_name": "0003.png",
                "chamber_type": "OpenBox-inner",
            },
            {
                "file_name": "0004.png",
                "chamber_type": "NormaleBox-pillar-inner",
            },
            {
                "file_name": "0005.png",
                "chamber_type": "OpenBox-collector-inner",
            },
            {
                "file_name": "0006.png",
                "chamber_type": "BigBox-inner",
            },
            {
                "file_name": "0007.png",
                "chamber_type": "NormaleBox-inner",
            },
            {
                "file_name": "0008.png",
                "chamber_type": "Mothermachine-2x-inner",
            },
        ]

        marker_group_configs = {
            "NormaleBox-pillar-inner": {
                "cross": np.array((4, 8), dtype=float),
                "circle": np.array((56, 8), dtype=float),
            },
            "BigBox-pillar-inner": {
                "cross": np.array((4, 8), dtype=float),
                "circle": np.array((56, 8), dtype=float),
            },
            "OpenBox-inner": {
                "cross": np.array((14, 8), dtype=float),
                "circle": np.array((66, 8), dtype=float),
            },
            "OpenBox-collector-inner": {
                "cross": np.array((14, 8), dtype=float),
                "circle": np.array((66, 8), dtype=float),
            },
            "BigBox-inner": {
                "cross": np.array((4, 8), dtype=float),
                "circle": np.array((56, 8), dtype=float),
            },
            "NormaleBox-inner": {
                "cross": np.array((4, 8), dtype=float),
                "circle": np.array((56, 8), dtype=float),
            },
            "Mothermachine-2x-inner": {
                "cross": np.array((14, 8), dtype=float),
                "circle": np.array((66, 8), dtype=float),
            },
            "Mothermachine-inner": {
                "cross": np.array((14, 8), dtype=float),
                "circle": np.array((66, 8), dtype=float),
            },
        }

        # load structures
        roi_structures = load_roi_structures(
            Path(dmc_masking.__file__).parent.parent
            / "artifacts/chamber_structure.json"
        )

        # general information
        pixel_size = 0.065789

        # create the masker
        rm = RoIMasker(
            # model_path="./artifacts/models/best34.pt",
            model_path=Path(dmc_masking.__file__).parent.parent
            / "artifacts/models/best34.pt",
            roi_polygon=None,
            marker_group_pixel=None,
        )

        for i, conf in enumerate(tqdm(configs)):
            image_file = (
                Path(dmc_masking.__file__).parent.parent
                / "artifacts/images/sak"
                / conf["file_name"]
            )
            chamber_type = conf["chamber_type"]

            marker_group = marker_group_configs[chamber_type]
            marker_group_pixel = marker_group_to_pixel_coordinates(
                marker_group, pixel_size
            )

            rp = RoIPolygon(shape(roi_structures[chamber_type]))

            rp = rp.scale(1.0 / pixel_size)
            xmin, ymin, _, _ = rp.roi_polygon.bounds

            # move polygon into positive coordinates
            rp = rp.translate(x=-xmin, y=-ymin)

            image = cv2.imread(image_file)

            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])[None]

            # apply the masker
            cropped_image, cropped_mask = rm(
                image,
                roi_polygon=rp,
                marker_group_pixel=marker_group_pixel,
                return_uncropped=False,
            )
            rotated_image, rotated_mask = rm(
                image,
                roi_polygon=rp,
                marker_group_pixel=marker_group_pixel,
                return_uncropped=True,
            )

            # plot
            _, axes = plt.subplots(1, 3, figsize=(20, 10))
            axes[0].imshow(np.moveaxis(image[0], [0, 1, 2], [2, 0, 1]))
            axes[1].imshow(np.moveaxis(rotated_image[0], [0, 1, 2], [2, 0, 1]))
            axes[1].imshow(rotated_mask[0], alpha=0.2)
            axes[2].imshow(np.moveaxis(cropped_image[0], [0, 1, 2], [2, 0, 1]))
            axes[2].imshow(cropped_mask[0], alpha=0.2)

            plt.tight_layout()

            plt.savefig(f"test_{i}.jpg")

    def test_ssrm(self):
        """test cropping of an image stack with a single structure masker"""
        ssrm = SingleStructureRoIMasker()

        image_path = (
            Path(dmc_masking.__file__).parent.parent
            / "artifacts/images/sak"
            / "0003.png"
        )

        image = cv2.imread(image_path)

        image_stack = np.stack((np.moveaxis(image, [0, 1, 2], [1, 2, 0]),) * 10, axis=0)

        cropped_images, _ = ssrm(image_stack=image_stack, roi_id="0000")

        self.assertEqual(len(cropped_images), 10)

        cropped_images = np.moveaxis(cropped_images, [1, 2, 3], [3, 1, 2])

        # plot
        _, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].imshow(cropped_images[0])
        axes[1].imshow(cropped_images[1])
        axes[2].imshow(cropped_images[2])

        plt.tight_layout()

        plt.savefig("test_ssrm.jpg")


if __name__ == "__main__":
    unittest.main()

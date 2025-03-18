"""Implementation of masking operations"""

import re
from copy import deepcopy

import numpy as np
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import Polygon, shape

from .io import load_roi_structures


class RoIPolygon:
    """RoI polygon shape"""

    def __init__(self, roi_polygon: Polygon):
        self.roi_polygon = roi_polygon

    def copy(self) -> "RoIPolygon":
        return deepcopy(self)

    def scale(self, scale: float) -> "RoIPolygon":
        return RoIPolygon(affinity.scale(self.roi_polygon, xfact=scale, yfact=scale))

    def translate(self, x: float = 0, y: float = 0) -> "RoIPolygon":
        return RoIPolygon(affinity.translate(self.roi_polygon, xoff=x, yoff=y))

    def to_mask(self, height: int, width: int) -> np.ndarray:
        return rasterize([self.roi_polygon], out_shape=(height, width))

    @property
    def area(self) -> float:
        return self.roi_polygon.area

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.roi_polygon.centroid.coords)[0]

    def difference(self, other: "RoIPolygon") -> "RoIPolygon":
        return RoIPolygon(self.roi_polygon.difference(other.roi_polygon))

    def union(self, other: "RoIPolygon") -> "RoIPolygon":
        return RoIPolygon(self.roi_polygon.union(other.roi_polygon))


def apply_mask(
    matched_marker_indices,
    rotated_markers,
    marker_group_pixels,
    roi_polygon,
    rotated_image,
    return_uncropped=False,
):
    """Compute and apply mask to image

    Args:
        matched_marker_indices (_type_): the pair of matched marker indices
        rotated_markers (_type_): the rotated marker information
        marker_group_pixels (_type_): the marker group information in pixels
        roi_polygon (_type_): the shape of the roi polygon (in pixels)
        rotated_image (_type_): the rotated image

    Returns:
        _type_: tuple of cropped image and mask
    """

    polygons = []
    masks = []

    im_width, im_height = rotated_image.shape[:2]

    for cross_index, circle_index in matched_marker_indices:

        cross_marker = rotated_markers[cross_index]
        circle_marker = rotated_markers[circle_index]

        print(cross_marker["bbox_center"][0])

        # correct for difference in expected width
        width = np.abs(cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0])
        expected_width = np.abs(
            marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0]
        )
        diff = width - expected_width

        # translate roi polygon
        rp = roi_polygon.translate(
            x=cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
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

    mask = masks[0]
    polygon: RoIPolygon = polygons[0]

    if return_uncropped:
        # return uncropped image and mask
        return rotated_image, mask

    # 8. Cropping

    minx, miny, maxx, maxy = tuple(map(int, map(np.round, polygon.roi_polygon.bounds)))
    cropped_image = rotated_image[miny:maxy, minx:maxx]
    cropped_mask = mask[miny:maxy, minx:maxx]

    return cropped_image, cropped_mask


def gen_pattern(start_c: int, array: int):
    return "|".join([rf"({c}{array}\d\d)" for c in range(start_c, 8, 2)])


class SAKRoIStructureLibrary:
    """Library for SAK roi structures"""

    def __init__(self, lookup_path):

        # load structural information of the polygon library
        roi_structures = load_roi_structures(lookup_path)
        self.polygon_library = {}
        for structure_name, serialized_polygon in roi_structures.items():
            self.polygon_library[structure_name] = shape(serialized_polygon)

        # load pattern matchin of id to structure name
        self.patterns = {
            "NormaleBox-inner": gen_pattern(0, 0),
            "BigBox-inner": gen_pattern(0, 1),
            "OpenBox-inner": gen_pattern(0, 2),
            "Mothermachine-inner": gen_pattern(0, 3),
            "NormaleBox-pillar-inner": gen_pattern(1, 0),
            "BigBox-pillar-inner": gen_pattern(1, 1),
            "OpenBox-collector-inner": gen_pattern(1, 2),
            "Mothermachine-2x-inner": gen_pattern(1, 3),
        }

        self.marker_group_configs = {
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

    def __call__(self, roi_id: str) -> Polygon:

        # match id with structure patterns
        structure_name = None

        for sn, structure_pattern in self.patterns.items():
            if re.match(structure_pattern, roi_id) is not None:
                structure_name = sn

        if structure_name is None:
            raise ValueError(
                f"No structure found corresponding to the roi id {roi_id}!"
            )

        return structure_name, self.polygon_library[structure_name]

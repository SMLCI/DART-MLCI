"""Implementation of masking operations"""

import re
import warnings
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

    def rotate(self, angle: float, origin: tuple[float, float] | str = "center") -> "RoIPolygon":
        """Rotate polygon around a point.

        Args:
            angle: Rotation angle in degrees (counter-clockwise positive)
            origin: Either a (x, y) tuple or "center" for bounding box center

        Returns:
            New RoIPolygon with rotated geometry
        """
        if origin == "center":
            origin = tuple(self.center)
        return RoIPolygon(affinity.rotate(self.roi_polygon, angle, origin=origin))

    def to_mask(self, height: int, width: int) -> np.ndarray:
        return rasterize([self.roi_polygon], out_shape=(height, width))

    @property
    def area(self) -> float:
        return self.roi_polygon.area

    @property
    def center(self) -> np.ndarray:
        """Return the center of the polygon's bounding box.

        Uses the bounding box center (midpoint of axis-aligned bounds) rather than
        the geometric centroid, which provides a more intuitive center point for
        asymmetric polygons.

        Returns:
            np.ndarray: (x, y) coordinates of the bounding box center
        """
        xmin, ymin, xmax, ymax = self.roi_polygon.bounds
        return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

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
    return_bbox=False,
    allow_truncation=False,
):
    """Compute and apply mask to image.

    Args:
        matched_marker_indices (list[tuple[int, int]]): Pairs of (cross_idx, circle_idx).
        rotated_markers (list[dict]): Detected marker dicts after rotation.
        marker_group_pixels (dict[str, np.ndarray]): Expected marker positions in pixels.
        roi_polygon (RoIPolygon): The RoI polygon shape (in pixels).
        rotated_image (np.ndarray): The rotated image (CxHxW or HxW).
        return_uncropped: If True, return the full image and mask without cropping.
        return_bbox: If True, also return the crop bounding box (minx, miny, maxx, maxy).
        allow_truncation: If True, allow the ROI to extend beyond image boundaries
                          by clipping the crop region to the image. Default is False.

    Returns:
        tuple: (cropped_image, cropped_mask) or (image, mask) if return_uncropped=True.
               If return_bbox=True, returns (image, mask, bbox) tuple.
    """

    polygons = []
    masks = []
    min_dists = []

    im_height, im_width = rotated_image.shape[-2:]

    for cross_index, circle_index in matched_marker_indices:
        cross_marker = rotated_markers[cross_index]
        circle_marker = rotated_markers[circle_index]

        # correct for difference in expected width
        width = np.abs(cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0])
        expected_width = np.abs(marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0])
        diff = width - expected_width

        # translate roi polygon
        rp = roi_polygon.translate(
            x=cross_marker["bbox_center"][0] - marker_group_pixels["cross"][0] + diff,
            y=cross_marker["bbox_center"][1] + marker_group_pixels["cross"][1],
        )

        # check whether roi polygon in image
        xmin, ymin, xmax, ymax = rp.roi_polygon.bounds

        if not allow_truncation and (xmin < 0 or xmax > im_width or ymin < 0 or ymax > im_height):
            # roi is out of image bounds
            continue

        min_dist = np.min(
            list(map(np.abs, [0 - xmin, xmax - im_width, ymin - 0, ymax - im_height]))
        )

        min_dists.append(min_dist)
        polygons.append(rp)
        masks.append(~rp.to_mask(height=im_height, width=im_width).astype(bool))

    if len(masks) == 0:
        raise ValueError("No roi lies completely inside the image")

    # use the RoI with maximum margin to the image boundaries
    # (or least overflow when allow_truncation is True)
    index = np.argmax(min_dists)

    mask = masks[index]
    polygon: RoIPolygon = polygons[index]

    # Cropping (compute bbox even if not cropping, for return_bbox)
    minx, miny, maxx, maxy = tuple(map(int, map(np.round, polygon.roi_polygon.bounds)))

    # Clip to image bounds when truncation is allowed
    if allow_truncation:
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, im_width)
        maxy = min(maxy, im_height)

    bbox = (minx, miny, maxx, maxy)

    if return_uncropped:
        # return uncropped image and mask
        if return_bbox:
            return rotated_image, mask, bbox
        return rotated_image, mask

    cropped_image = rotated_image[..., miny:maxy, minx:maxx]
    cropped_mask = mask[miny:maxy, minx:maxx]

    if return_bbox:
        return cropped_image, cropped_mask, bbox
    return cropped_image, cropped_mask


def filter_segmentation_by_mask(
    labeled_mask: np.ndarray,
    chamber_mask: np.ndarray,
    threshold: float = 0.5,
    relabel: bool = True,
) -> np.ndarray:
    """Remove segmented objects that significantly overlap with masked-out regions.

    Args:
        labeled_mask: HxW uint16 instance mask (0=bg, 1..N=cells)
        chamber_mask: HxW bool mask (True=outside ROI / internal structures)
        threshold: Fraction of object area that must be in valid region to keep it
        relabel: If True (default), relabel remaining IDs to be contiguous.
            If False, keep original label IDs (removed cells are zeroed out).

    Returns:
        Filtered instance mask.
    """
    if labeled_mask.max() == 0:
        return labeled_mask

    filtered = labeled_mask.copy()
    for label_id in range(1, int(labeled_mask.max()) + 1):
        obj_pixels = labeled_mask == label_id
        n_total = np.sum(obj_pixels)
        if n_total == 0:
            continue
        n_masked = np.sum(obj_pixels & chamber_mask)
        if n_masked / n_total > threshold:
            filtered[obj_pixels] = 0

    if not relabel:
        return filtered

    # Relabel to keep IDs contiguous
    unique_labels = np.unique(filtered)
    unique_labels = unique_labels[unique_labels > 0]
    relabeled = np.zeros_like(filtered)
    for new_id, old_id in enumerate(unique_labels, 1):
        relabeled[filtered == old_id] = new_id

    return relabeled


def apply_mask_rotation_free(
    matched_marker_indices: list[tuple[int, int]],
    markers: list[dict],
    marker_group_pixels: dict[str, np.ndarray],
    roi_polygon: "RoIPolygon",
    image: np.ndarray,
    rotation_angle: float,
    return_uncropped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and apply mask to an unrotated image using polygon rotation.

    This function avoids the expensive image rotation step by rotating the RoI polygon
    instead. The polygon is rotated to match the detected orientation and applied
    directly to the unrotated image.

    Args:
        matched_marker_indices: List of (cross_idx, circle_idx) tuples for matched markers
        markers: List of detected marker dicts with bbox_center (NOT rotated)
        marker_group_pixels: Expected marker positions in pixels (cross, circle)
        roi_polygon: The RoI polygon shape (in pixels)
        image: The unrotated image, shape (H, W) or (C, H, W)
        rotation_angle: Detected rotation angle in degrees (counter-clockwise positive)
        return_uncropped: If True, return the full image and mask without cropping

    Returns:
        Tuple of (cropped_image, cropped_mask) or (image, mask) if return_uncropped=True
    """
    polygons = []
    masks = []
    min_dists = []

    im_height, im_width = image.shape[-2:]

    # Get the cross marker position in the polygon's local coordinate system
    cross_local = marker_group_pixels["cross"]

    for cross_index, circle_index in matched_marker_indices:
        cross_marker = markers[cross_index]
        circle_marker = markers[circle_index]

        # Correct for difference in expected width (scaling correction)
        width = np.abs(cross_marker["bbox_center"][0] - circle_marker["bbox_center"][0])
        expected_width = np.abs(marker_group_pixels["cross"][0] - marker_group_pixels["circle"][0])
        diff = width - expected_width

        # Step 1: Compute the rotation origin in the polygon's local coordinates
        # The cross marker is at cross_local relative to the polygon's origin (0,0)
        # We rotate around the cross marker position
        rotation_origin = (cross_local[0], -cross_local[1])

        # Step 2: Rotate the polygon around the cross marker position
        rp = roi_polygon.rotate(rotation_angle, origin=rotation_origin)

        # Step 3: Translate the rotated polygon so the cross marker aligns with detection
        # After rotation, the cross marker is still at rotation_origin in local coords
        # We need to move it to the detected cross marker position
        rp = rp.translate(
            x=cross_marker["bbox_center"][0] - rotation_origin[0] + diff,
            y=cross_marker["bbox_center"][1] - rotation_origin[1],
        )

        # Check whether RoI polygon is within image bounds
        xmin, ymin, xmax, ymax = rp.roi_polygon.bounds

        if xmin < 0 or xmax > im_width or ymin < 0 or ymax > im_height:
            # RoI is out of image bounds
            continue

        min_dist = np.min(
            list(map(np.abs, [0 - xmin, xmax - im_width, ymin - 0, ymax - im_height]))
        )

        min_dists.append(min_dist)
        polygons.append(rp)
        masks.append(~rp.to_mask(height=im_height, width=im_width).astype(bool))

    if len(masks) == 0:
        raise ValueError("No roi lies completely inside the image")

    # Use the RoI with maximum margin to the image boundaries
    index = np.argmax(min_dists)

    mask = masks[index]
    polygon: RoIPolygon = polygons[index]

    if return_uncropped:
        return image, mask

    # Crop to polygon bounds
    minx, miny, maxx, maxy = tuple(map(int, map(np.round, polygon.roi_polygon.bounds)))
    cropped_image = image[..., miny:maxy, minx:maxx]
    cropped_mask = mask[miny:maxy, minx:maxx]

    return cropped_image, cropped_mask


def _gen_pattern(start_c: int, array: int) -> str:
    """Generate ROI ID matching pattern.

    .. deprecated::
        This function is kept for internal backward compatibility only.
        Use ChipStructureLibrary with a chip config file instead.
    """
    return "|".join([rf"({c}{array}\d\d)" for c in range(start_c, 8, 2)])


# Legacy marker positions in microns (hardcoded for SAK chip)
_SAK_MARKER_CONFIGS = {
    "NormaleBox-inner": {"cross": (4.0, 8.0), "circle": (56.0, 8.0)},
    "BigBox-inner": {"cross": (4.0, 8.0), "circle": (56.0, 8.0)},
    "OpenBox-inner": {"cross": (14.0, 8.0), "circle": (66.0, 8.0)},
    "Mothermachine-inner": {"cross": (14.0, 8.0), "circle": (66.0, 8.0)},
    "NormaleBox-pillar-inner": {"cross": (4.0, 8.0), "circle": (56.0, 8.0)},
    "BigBox-pillar-inner": {"cross": (4.0, 8.0), "circle": (56.0, 8.0)},
    "OpenBox-collector-inner": {"cross": (14.0, 8.0), "circle": (66.0, 8.0)},
    "Mothermachine-2x-inner": {"cross": (14.0, 8.0), "circle": (66.0, 8.0)},
}

_SAK_ROI_PATTERNS = {
    "NormaleBox-inner": _gen_pattern(0, 0),
    "BigBox-inner": _gen_pattern(0, 1),
    "OpenBox-inner": _gen_pattern(0, 2),
    "Mothermachine-inner": _gen_pattern(0, 3),
    "NormaleBox-pillar-inner": _gen_pattern(1, 0),
    "BigBox-pillar-inner": _gen_pattern(1, 1),
    "OpenBox-collector-inner": _gen_pattern(1, 2),
    "Mothermachine-2x-inner": _gen_pattern(1, 3),
}


class SAKRoIStructureLibrary:
    """Library for SAK roi structures.

    .. deprecated::
        Use :class:`dmc_masking.chip.ChipStructureLibrary` instead.
        This class is maintained for backward compatibility and will be
        removed in a future release.
    """

    def __init__(self, lookup_path, pixel_size):
        warnings.warn(
            "SAKRoIStructureLibrary is deprecated. "
            "Use dmc_masking.chip.ChipStructureLibrary instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.pixel_size = pixel_size

        # Build from legacy files using the same logic as before
        roi_structures = load_roi_structures(lookup_path)
        self.polygon_library = {}
        for structure_name, serialized_polygon in roi_structures.items():
            self.polygon_library[structure_name] = shape(serialized_polygon)

        self.patterns = dict(_SAK_ROI_PATTERNS)

        self.marker_group_configs = {}
        for name, markers in _SAK_MARKER_CONFIGS.items():
            self.marker_group_configs[name] = {
                mk: np.array(mv, dtype=float) for mk, mv in markers.items()
            }

        for sn, sp in self.polygon_library.items():
            rp = RoIPolygon(sp)
            rp = rp.scale(1.0 / pixel_size)
            xmin, ymin, _, _ = rp.roi_polygon.bounds
            rp = rp.translate(x=-xmin, y=-ymin)
            self.polygon_library[sn] = rp

        for _sn, sc in self.marker_group_configs.items():
            for mn, mc in sc.items():
                sc[mn] = mc * 1.0 / pixel_size

    def _structure_name(self, roi_id: str) -> str:
        roi_id = str(roi_id)
        for sn, structure_pattern in self.patterns.items():
            if re.match(structure_pattern, roi_id) is not None:
                return sn

    def __call__(self, roi_id: str) -> tuple[str, RoIPolygon, dict]:
        structure_name = None
        for sn, structure_pattern in self.patterns.items():
            if re.match(structure_pattern, roi_id) is not None:
                structure_name = sn

        if structure_name is None:
            raise ValueError(f"No structure found corresponding to the roi id {roi_id}!")

        return (
            structure_name,
            self.polygon_library[structure_name],
            self.marker_group_configs[structure_name],
        )


class SingleRoIStructureLibrary(SAKRoIStructureLibrary):
    """Library for a single type of roi structure.

    .. deprecated::
        Use :class:`dmc_masking.chip.ChipStructureLibrary` instead.
        This class is maintained for backward compatibility and will be
        removed in a future release.
    """

    def __init__(self, lookup_path, pixel_size, structure_name: str):
        super().__init__(lookup_path, pixel_size)

        self.structure_name = structure_name

        if self.structure_name not in self.polygon_library:
            raise ValueError(
                f"Structure {self.structure_name} is not in the polygon libarary. Only {[self.polygon_library.keys()]} names are available!"
            )

    def __call__(self, roi_id: str | None = None) -> tuple[str, RoIPolygon, dict]:
        structure_name = self.structure_name

        return (
            structure_name,
            self.polygon_library[structure_name],
            self.marker_group_configs[structure_name],
        )

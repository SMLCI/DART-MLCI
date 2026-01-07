"""Implementation of marker matching."""

from collections import deque
from itertools import product

import numpy as np


def marker_group_to_pixel_coordinates(marker_group, pixel_size: float):
    new_marker_group = {}

    for label, pos in marker_group.items():
        new_marker_group[label] = pos * (1.0 / pixel_size)

    return new_marker_group


def match_markers(markers, marker_group: dict[str, np.ndarray], on="bbox_center", tolerance=5.0):
    if len(marker_group) != 2:
        raise ValueError("No more general implementation!")

    cross_pos = marker_group["cross"]
    circle_pos = marker_group["circle"]

    expected_distance = np.linalg.norm(cross_pos - circle_pos)

    data = deque()

    cross_marker_indices = filter(lambda i: markers[i]["label"] == "cross", range(len(markers)))
    circle_marker_indices = filter(lambda i: markers[i]["label"] == "circle", range(len(markers)))

    for iCross, iCircle in product(cross_marker_indices, circle_marker_indices):
        dist = np.linalg.norm(markers[iCross][on] - markers[iCircle][on])

        if np.abs(dist - expected_distance) < tolerance:
            data.append((iCross, iCircle, dist))

    sorted(data, key=lambda x: x[2])

    used_marker_indices = set()
    matches = []
    while len(data) > 0:
        iCross, iCircle, dist = data.popleft()

        if len(used_marker_indices.intersection({iCross, iCircle})) == 0:
            matches.append((iCross, iCircle))

        used_marker_indices.add(iCross)
        used_marker_indices.add(iCircle)

    return matches

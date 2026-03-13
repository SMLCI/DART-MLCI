"""Implementation of marker matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment


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

    cross_indices = [i for i in range(len(markers)) if markers[i]["label"] == "cross"]
    circle_indices = [i for i in range(len(markers)) if markers[i]["label"] == "circle"]

    if not cross_indices or not circle_indices:
        return []

    cost_matrix = np.full((len(cross_indices), len(circle_indices)), np.inf)

    for ci, iCross in enumerate(cross_indices):
        for cj, iCircle in enumerate(circle_indices):
            dist = np.linalg.norm(markers[iCross][on] - markers[iCircle][on])
            distance_error = abs(dist - expected_distance)

            if distance_error < tolerance:
                conf_cross = markers[iCross].get("conf", 1.0)
                conf_circle = markers[iCircle].get("conf", 1.0)
                cost_matrix[ci, cj] = distance_error / (conf_cross * conf_circle)

    # Replace inf with a large finite value for the solver, then filter after
    finite_mask = np.isfinite(cost_matrix)
    if not finite_mask.any():
        return []

    large_cost = cost_matrix[finite_mask].max() + 1e6
    solver_matrix = np.where(finite_mask, cost_matrix, large_cost)
    row_ind, col_ind = linear_sum_assignment(solver_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind, strict=False):
        if finite_mask[r, c]:
            matches.append((cross_indices[r], circle_indices[c]))

    return matches

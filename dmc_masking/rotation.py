"""Implementation of angle and rotation functions."""

import cv2
import numpy as np


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in degrees between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 57.29578


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center

    Args:
        image (np.ndarray): the image to rotate
        angle (float): the angle in degrees

    Returns:
        np.ndarray: the rotated image
    """

    height, width = image.shape[-2:]

    image_center = tuple(np.array(image.shape[-2:][::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result


def rotate_image_and_markers(
    image: np.ndarray, markers, angle: float, position_labels=None
) -> np.ndarray:
    """Rotate image around its center

    Args:
        image (np.ndarray): the image to rotate. CxHxW
        markers: Markers to rotate
        angle (float): the angle in degrees

    Returns:
        np.ndarray: the rotated image. CxHxW
    """

    height, width = image.shape[-2:]

    image_center = tuple(np.array(image.shape[-2:][::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate all channels
    result = np.stack(
        [cv2.warpAffine(im, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR) for im in image],
        axis=0,
    )

    if position_labels is None:
        position_labels = ["bbox_center"]

    new_markers = []

    for marker in markers:
        new_marker = {**marker}
        for pl in position_labels:
            p = np.array([*new_marker[pl], 1])

            # apply rotation matrix
            new_marker[pl] = np.dot(rot_mat, p.T)

        new_markers.append(new_marker)

    return result, new_markers


def rotate_point(p: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    """Clockwise rotation of a 2d point around an origin

    Args:
        p (np.ndarray): the point
        origin (np.ndarray): the origin
        angle (float): the angle in degrees for rotation

    Returns:
        np.ndarray: the rotated point
    """
    image_center = tuple(origin)  # tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = np.array(cv2.getRotationMatrix2D(image_center, angle, 1.0))

    p = np.array([*p, 1])

    return np.dot(rot_mat, p.T)


def rotate_markers(markers, image, angle: float, position_labels=None):
    if position_labels is None:
        position_labels = ["bbox_center"]

    new_markers = []

    image_center = tuple(np.array(image.shape[-2:][::-1]) / 2)

    for marker in markers:
        new_marker = {**marker}
        for pl in position_labels:
            new_marker[pl] = rotate_point(new_marker[pl], image_center, angle)

        new_markers.append(new_marker)

    return new_markers


def compute_marker_group_angles(markers, matched_marker_indices, marker_group, on="bbox_center"):
    angles = []
    blueprint_cross_to_circle = marker_group["circle"] - marker_group["cross"]

    for iCross, iCircle in matched_marker_indices:
        measured_cross_to_circle = markers[iCircle][on] - markers[iCross][on]
        angles.append(angle_between(blueprint_cross_to_circle, measured_cross_to_circle))

    return angles

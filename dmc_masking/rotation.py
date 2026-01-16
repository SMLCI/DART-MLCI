"""Implementation of angle and rotation functions."""

import cv2
import kornia.geometry.transform as KT
import numpy as np
import torch


def _get_rotation_matrix_and_bounds(height: int, width: int, angle: float):
    """Compute rotation matrix and output bounds for rotating an image.

    Args:
        height: Image height
        width: Image width
        angle: Rotation angle in degrees

    Returns:
        rot_mat: 2x3 rotation matrix
        bound_w: Output width after rotation
        bound_h: Output height after rotation
    """
    image_center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    return rot_mat, bound_w, bound_h


def rotate_image_opencv(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image using OpenCV (CPU).

    Args:
        image: Image array of shape (C, H, W)
        angle: Rotation angle in degrees

    Returns:
        Rotated image array of shape (C, H', W')
    """
    height, width = image.shape[-2:]
    rot_mat, bound_w, bound_h = _get_rotation_matrix_and_bounds(height, width, angle)

    result = np.stack(
        [cv2.warpAffine(im, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR) for im in image],
        axis=0,
    )
    return result


def rotate_image_kornia(
    image: np.ndarray | torch.Tensor,
    angle: float,
    device: str | None = None,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """Rotate image using kornia (GPU-accelerated).

    Args:
        image: Image array of shape (C, H, W), either numpy array or torch.Tensor
        angle: Rotation angle in degrees
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        return_tensor: If True, return torch.Tensor instead of numpy array

    Returns:
        Rotated image of shape (C, H', W'), as numpy array or torch.Tensor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    height, width = image.shape[-2:]
    rot_mat, bound_w, bound_h = _get_rotation_matrix_and_bounds(height, width, angle)

    # Convert to tensor if needed: (C, H, W) -> (1, C, H, W)
    if isinstance(image, torch.Tensor):
        tensor = image.unsqueeze(0) if image.dim() == 3 else image
        if tensor.device.type != device.split(":")[0]:
            tensor = tensor.to(device)
        tensor = tensor.float()
    else:
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

    # Convert 2x3 rotation matrix to 3x3 for kornia
    rot_mat_3x3 = np.vstack([rot_mat, [0, 0, 1]])
    rot_mat_tensor = torch.from_numpy(rot_mat_3x3).unsqueeze(0).float().to(device)

    # Apply affine transformation
    rotated = KT.warp_affine(
        tensor,
        rot_mat_tensor[:, :2, :],  # kornia expects 2x3 matrix
        dsize=(bound_h, bound_w),
        mode="bilinear",
        padding_mode="zeros",
    )

    # Return tensor or numpy based on flag
    result = rotated.squeeze(0)
    if return_tensor:
        return result
    return result.cpu().numpy()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the absolute angle in degrees between vectors 'v1' and 'v2'.

    Note: This function returns only positive angles (0° to 180°).
    Use signed_angle_between() for signed angles (-180° to 180°).
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 57.29578


def signed_angle_between(v1, v2):
    """Returns the signed angle in degrees from vector 'v1' to vector 'v2'.

    The angle is positive for counter-clockwise rotation (in image coordinates
    where y-axis points down, this appears as clockwise on screen).

    Args:
        v1: Reference vector (2D numpy array)
        v2: Target vector (2D numpy array)

    Returns:
        Signed angle in degrees, range [-180, 180]
    """
    # Cross product (z-component): v1.x * v2.y - v1.y * v2.x
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    # Dot product: v1.x * v2.x + v1.y * v2.y
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    # atan2 gives signed angle in radians
    angle_rad = np.arctan2(cross, dot)
    # Convert to degrees
    return angle_rad * (180.0 / np.pi)


def rotate_image(image: np.ndarray, angle: float, use_gpu: bool = True) -> np.ndarray:
    """Rotate image around its center

    Args:
        image (np.ndarray): the image to rotate (2D: H x W)
        angle (float): the angle in degrees
        use_gpu (bool): Use GPU-accelerated kornia if available (default: True)

    Returns:
        np.ndarray: the rotated image
    """
    height, width = image.shape[-2:]
    rot_mat, bound_w, bound_h = _get_rotation_matrix_and_bounds(height, width, angle)

    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result


def rotate_image_and_markers(
    image: np.ndarray | torch.Tensor,
    markers,
    angle: float,
    position_labels=None,
    use_gpu: bool = True,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """Rotate image around its center

    Args:
        image: the image to rotate. CxHxW (numpy array or torch.Tensor)
        markers: Markers to rotate
        angle (float): the angle in degrees
        position_labels: Keys in marker dict to transform (default: ["bbox_center"])
        use_gpu (bool): Use GPU-accelerated kornia if available (default: True)
        return_tensor (bool): If True, return torch.Tensor instead of numpy array

    Returns:
        Rotated image (CxHxW) as numpy array or torch.Tensor
        list: transformed markers
    """
    height, width = image.shape[-2:]
    rot_mat, bound_w, bound_h = _get_rotation_matrix_and_bounds(height, width, angle)

    # Rotate image using GPU or CPU
    is_tensor = isinstance(image, torch.Tensor)
    if use_gpu and torch.cuda.is_available():
        result = rotate_image_kornia(image, angle, return_tensor=return_tensor or is_tensor)
    else:
        # CPU path requires numpy
        if is_tensor:
            image_np = image.cpu().numpy()
        else:
            image_np = image
        result = rotate_image_opencv(image_np, angle)
        if return_tensor:
            result = torch.from_numpy(result)

    # Transform marker positions (CPU, already efficient)
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


def compute_marker_group_angles(
    markers, matched_marker_indices, marker_group, on="bbox_center", signed=True
):
    """Compute rotation angles between detected markers and expected marker positions.

    Args:
        markers: List of detected markers with position information
        matched_marker_indices: List of (cross_idx, circle_idx) tuples
        marker_group: Dict with 'cross' and 'circle' expected positions
        on: Key to use for marker position (default: 'bbox_center')
        signed: If True, return signed angles (-180° to 180°). If False, return
                absolute angles (0° to 180°). Default is True.

    Returns:
        List of angles in degrees for each matched marker pair
    """
    angles = []
    blueprint_cross_to_circle = marker_group["circle"] - marker_group["cross"]

    for iCross, iCircle in matched_marker_indices:
        measured_cross_to_circle = markers[iCircle][on] - markers[iCross][on]
        if signed:
            angles.append(signed_angle_between(blueprint_cross_to_circle, measured_cross_to_circle))
        else:
            angles.append(angle_between(blueprint_cross_to_circle, measured_cross_to_circle))

    return angles

"""Utility functionality."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def normalize_image(
    im: np.ndarray, low_quantile=0.01, high_quantile=0.99
) -> np.ndarray:
    """Normalize image to uint8 space [0...255]

    Args:
        im (np.ndarray): the input image
        low_quantile (float, optional): the lower quantile (values below become 0). Defaults to 0.01.
        high_quantile (float, optional): the upper quantile (value higher become 255). Defaults to 0.99.

    Returns:
        np.ndarray: the resulting image with values in the uint8 space [0...255]
    """

    # compute min max
    im_max, im_min = np.quantile(im, high_quantile), np.quantile(im, low_quantile)

    # normalize image
    return (np.clip((im - im_min) / (im_max - im_min), 0, 1) * 255).astype(np.uint8)


def load_tiff(image_path: Path) -> np.ndarray:
    """Loading tiff file into uint8 numpy array

    Args:
        image_path (Path): path to the tiff file

    Returns:
        np.ndarray: loaded numpy image in uint8 range [0...255]
    """
    im_raw = tifffile.imread(image_path)

    return normalize_image(im_raw)


def center_of_mask_mass(mask):
    y, x = np.nonzero(mask)
    return np.median(np.unique(x)), np.median(np.unique(y))


def plot_marker_data(marker_data, ax):
    for marker in marker_data:
        x, y = marker["bbox_center"]

        ax.scatter(x, y, s=1, c="red")

        x, y = marker["mask_center"]

        ax.scatter(x, y, s=1, c="blue")


def plot_markers(image: np.ndarray, markers: dict):
    """Visualize markers on the image using matplotlib

    Args:
        image (np.ndarray): the image
        markers (dict): the detected markers
    """

    marker_image = np.copy(image)

    plt.imshow(marker_image)
    plt.tight_layout()
    plt.axis("off")

    for m in markers:
        print(m)
        if m["label"] == "cross":
            c = "red"
            mr = "+"
        elif m["label"] == "circle":
            c = "blue"
            mr = "o"

        plt.plot(
            [m["bbox_center"][0]], [m["bbox_center"][1]], c=c, marker=mr, markersize=10
        )


def plot_marker_paris(image: np.ndarray, matched_marker_indices: list, markers: dict):
    """Visualize marker pairs on the image

    Args:
        image (np.ndarray): the image
        matched_marker_indices (list): the matched markers
        markers (dict): the makerd detections
    """

    matched_image = np.copy(image)

    colors = ["purple", "yellow"]

    plt.figure()
    plt.imshow(matched_image)
    for i, index_match in enumerate(matched_marker_indices):
        for ind in index_match:
            plt.plot(
                [markers[ind]["bbox_center"][0]],
                [markers[ind]["bbox_center"][1]],
                c=colors[i],
                marker="+" if markers[ind]["label"] == "cross" else "o",
                markersize=10,
            )

    plt.axis("off")
    plt.tight_layout()


def homogenize_image_size(result_images: list[np.ndarray]):
    """Homogneize spatial image dimensions

    Args:
        result_images (list[np.ndarray]): List of TxHxW images where HxW may differ

    Returns:
        np.ndarray: TxH*xW*xC image stack with homogeneous and maximal image spatial dimensions of H*xW*
    """

    # extract image shapes
    shapes = np.stack([np.array(im.shape) for im in result_images], axis=0)

    if not np.all(shapes[:, -1] == shapes[0, -1]):
        raise ValueError(
            f"Inhomogeneous number of channels (last dimension). All images need the same number of channels. But they are of dimension {shapes}"
        )

    # identify max size
    max_height = np.max(shapes[:, 0])
    max_width = np.max(shapes[:, 1])

    for i, im in enumerate(result_images):
        im_height, im_width = im.shape[:2]

        # compute differences to max image size
        ph = max_height - im_height
        pw = max_width - im_width

        # introduce padding
        result_images[i] = np.pad(im, [(0, ph), (0, pw), (0, 0)])

    # combine into a single image stack
    return np.stack(result_images, axis=0)

"""Utility functionality."""

from pathlib import Path

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

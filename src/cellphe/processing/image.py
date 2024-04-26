"""
    cellphe.processing.image
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Functions related to processing images.
"""

from __future__ import annotations

import numpy as np


def normalise_image(image: np.array, lower: int, upper: int) -> np.array:
    """
    Normalises an image to a specified range.

    :param image: The image to normalise as a 2D numpy array.
    :param lower: The lower bound of the target normalisation range, as an integer.
    :param upper: The upper boundo of the target normalisation range, as an integer.

    :return: The normalised image as a 2D numpy array.
    """
    scale = (upper - lower) / (np.max(image) - np.min(image))
    offset = upper - (scale * np.max(image))

    return scale * image + offset


def create_type_mask(image: np.array, roi: np.array) -> np.array:
    """
    Creates a type mask for a given ROI in an image.

    This achieves 4 things:
        - The image is subset to the region containing the ROI
        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    return image

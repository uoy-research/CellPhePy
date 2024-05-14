"""
    cellphe.processing.image
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Functions related to processing images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path


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
    # Initialise the mask
    image_mask = np.full(image.shape, -1)
    # ROI = 1, NB: ROI is [x,y] coordinates rather than [row,col]
    image_mask[roi[:, 1], roi[:, 0]] = 0

    # Get values that are inside the ROI using the matplotlib Path class
    path = Path(roi)
    # Get the indices of every coordinate in the image (row, col)
    image_indices = np.indices(image.shape).reshape(2, image.size).T
    # Convert into (x,y) coordinates
    image_indices = np.flip(image_indices, axis=1)
    # See if the ROI contains these values
    mask = path.contains_points(image_indices).reshape(image.shape)

    # Remove the ROI border itself
    mask[roi[:, 1], roi[:, 0]] = False
    # Set these values in the output to 0
    image_mask[mask] = 1

    return image_mask


@dataclass
class SubImage:
    sub_image: np.array
    type_mask: np.array
    centroid: float


def extract_subimage(image: np.array, roi: np.array) -> SubImage:
    """
    Extracts a sub-image and relevant statistics from a given image and ROI.

    :param image: The image as a 2D Numpy array.
    :param roi: The region of interest as an Mx2 Numpy array.

    :return: A SubImage instance.
    """
    mask = create_type_mask(image, roi)

    bbox = Path(roi).get_extents()
    mask_subset = mask[int(bbox.ymin) : (int(bbox.ymax) + 1), int(bbox.xmin) : (int(bbox.xmax) + 1)]
    image_subset = image[int(bbox.ymin) : (int(bbox.ymax) + 1), int(bbox.xmin) : (int(bbox.xmax) + 1)]

    centroid = roi.mean(axis=0)

    return SubImage(sub_image=image_subset, type_mask=mask_subset, centroid=centroid)

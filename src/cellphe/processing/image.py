"""
    cellphe.processing.image
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Functions related to processing images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from skimage.segmentation import flood_fill

from cellphe.processing.roi import roi_corners


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


@dataclass
class SubImage:
    sub_image: np.array
    type_mask: np.array
    centroid: float


def create_type_mask_matplotlib(roi: np.array) -> np.array:
    """
    Creates a type mask for a given ROI in an image.

    This returns a 2D integer array representing the sub-image of the cell
    that the ROI covers, where:

        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    This method uses matplotlib's Path module and 'contains_points' methods on
    all the points in the image subset to see if they reside within the ROI.

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    maxcol, maxrow = roi.max(axis=0) + 1
    mask = np.full((maxrow, maxcol), -1)
    # Get values that are inside the ROI using the matplotlib Path class
    path = Path(roi)
    # Get the indices of every coordinate in the image (row, col)
    image_indices = np.indices(mask.shape).reshape(2, mask.size).T
    # Convert into (x,y) coordinates
    image_indices = np.flip(image_indices, axis=1)
    # See if the ROI contains these values
    poly_mask = path.contains_points(image_indices).reshape(mask.shape)
    # Set these values in the output to 0
    mask[poly_mask] = 1
    # Set ROI boundaries as 0
    mask[roi[:, 1], roi[:, 0]] = 0
    return mask


def create_type_mask_fill_polygon(roi):
    """
    Creates a type mask for a given ROI in an image.

    This returns a 2D integer array representing the sub-image of the cell
    that the ROI covers, where:

        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    This implementation is based the following algorithm:
    https://www.alienryderflex.com/polygon_fill/

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    maxcol, maxrow = roi.max(axis=0) + 1
    node_mask = np.zeros((maxrow, maxcol)).astype(int)
    mask_vec = np.zeros((maxrow, maxcol)).astype(int)

    # get corners of the polygon, needed for the following algorithm.
    corners = roi_corners(roi)

    # Setup
    height = roi[:, 1].max()
    cornerY = corners[:, 1]
    cornerX = corners[:, 0]
    # The algorithm iterates through the corners in path order in pairs, so
    # keep a rotated copy to use as the 'previous' corner
    prevCornerY = np.roll(cornerY, 1)
    prevCornerX = np.roll(cornerX, 1)

    # The original algorithm iterates row-by-row. We want to vectorise this so
    # will create data structures to hold all the corners for each row
    prevCornerX = np.tile(prevCornerX, height)
    prevCornerY = np.tile(prevCornerY, height)
    cornerX = np.tile(cornerX, height)
    cornerY = np.tile(cornerY, height)
    pixelY = np.repeat(np.arange(height), corners.shape[0])

    # The crux of the algorithm. Find nodes, i.e. crossing points
    has_node = ((cornerY < pixelY) & (prevCornerY >= pixelY)) | ((prevCornerY < pixelY) & (cornerY >= pixelY))
    # Calculate the x-values at each node
    polyx = cornerX[has_node]
    polyy = cornerY[has_node]
    prevy = prevCornerY[has_node]
    prevx = prevCornerX[has_node]
    new_y = pixelY[has_node]
    new_x = (polyx + (new_y - polyy) / (prevy - polyy) * (prevx - polyx)).astype(int)

    # Save the number of crossings at each node
    for x, y in zip(new_x, new_y):
        node_mask[y, x] += 1

    # Interior locations are those with an odd number of boundary crossings
    mask_rows = node_mask.cumsum(axis=1)
    mask_vec[mask_rows % 2 == 1] = 1
    # Set anything else as outside the cell
    mask_vec[mask_vec == 0] = -1
    # Add ROI boundary
    mask_vec[roi[:, 1], roi[:, 0]] = 0
    return mask_vec


def create_type_mask_flood_fill(roi: np.array) -> np.array:
    """
    Creates a type mask for a given ROI in an image.

    This returns a 2D integer array representing the sub-image of the cell
    that the ROI covers, where:

        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    This method uses a floodfill algorithm, implemented in skimage.

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    # Initialise the mask
    maxcol, maxrow = roi.max(axis=0) + 1
    mask = np.full((maxrow, maxcol), -1)
    mask[roi[:, 1], roi[:, 0]] = 0

    # Fill values inside the ROI to 1 using floodfill
    # Using the median as a guaranteed point inside the ROI.
    # Not great, but given that the ROIs must all be 8x8 then should be
    # sufficient. Else can use Shapely's representative_point
    start_col, start_row = tuple(np.median(roi, axis=0).astype(int))
    mask = flood_fill(mask, (start_row, start_col), 1, connectivity=1)
    return mask


def extract_subimage(image: np.array, roi: np.array, method="fill_polygon") -> SubImage:
    """
    Extracts a sub-image and relevant statistics from a given image and ROI.

    :param image: The image as a 2D Numpy array.
    :param roi: The region of interest as an Mx2 Numpy array.

    :return: A SubImage instance.
    """
    # Calculate centroid before subsetting roi, as units are relative to the
    # entire frame
    centroid = roi.mean(axis=0)

    # Subset image to ROI
    xmin, ymin = roi.min(axis=0)
    xmax, ymax = roi.max(axis=0)
    image_subset = image[int(ymin) : (int(ymax) + 1), int(xmin) : (int(xmax) + 1)]
    roi = roi - roi.min(axis=0)
    algs = {
        "fill_polygon": create_type_mask_fill_polygon,
        "matplotlib": create_type_mask_matplotlib,
        "flood_fill": create_type_mask_flood_fill,
    }
    alg = algs[method]
    mask_subset = alg(roi)

    return SubImage(sub_image=image_subset, type_mask=mask_subset, centroid=centroid)

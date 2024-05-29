"""
    cellphe.processing.image
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Functions related to processing images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from skimage.measure import grid_points_in_poly
from skimage.segmentation import flood

from cellphe.processing.roi import roi_corners


def normalise_image(image: np.array, lower: int, upper: int) -> np.array:
    """
    Normalises an image to a specified range.

    :param image: The image to normalise as a 2D numpy array.
    :param lower: The lower bound of the target normalisation range, as an integer.
    :param upper: The upper boundo of the target normalisation range, as an integer.

    :return: The normalised image as a 2D numpy array.
    """
    return (image - image.min()) / (image.max() - image.min()) * (upper - lower) + lower


@dataclass
class SubImage:
    """
    Represents a sub-image, that is a portion of an image corresponding to a
    specific region-of-interest (ROI).

    Properties:
        - sub_image: A 2D array representing the image truncated to the bounds
        of the ROI.
        - type_mask: A 2D array the same size as the sub-image detailing which
        pixels are either inside the ROI (1), outside (-1), or on the ROI border
        (0).
        - centroid: A 1D array of length 2 containing the (x,y) points at the
        centre of the ROI.
    """

    sub_image: np.array
    type_mask: np.array
    centroid: np.array


def create_type_mask_skimage(roi: np.array) -> np.array:
    """
    Creates a type mask for a given ROI in an image.

    This returns a 2D integer array representing the sub-image of the cell
    that the ROI covers, where:

        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    This method uses skimage's grid_points_in_poly, a point in polygon algorithm
    similar to Matplotlib's.

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    maxcol, maxrow = roi.max(axis=0) + 1
    mask = np.full((maxrow, maxcol), -1)
    grid_mask = grid_points_in_poly((maxrow, maxcol), np.flip(roi, axis=1))
    mask[grid_mask] = 1
    # Set ROI boundaries as 0
    mask[roi[:, 1], roi[:, 0]] = 0
    return mask


def create_type_mask_ray_cast_4(roi: np.array) -> np.array:
    """
    Creates a type mask for a given ROI in an image.

    This returns a 2D integer array representing the sub-image of the cell
    that the ROI covers, where:

        - Pixels outside the cell are given a value of -1
        - Pixels on the ROI border are assigned 0
        - Pixels inside the ROI border are assigned 1

    This is a modified Ray Casting algorithm that rather than defining
    interior pixels as those that lie within an odd number of boundary crossings
    in 1 direction, it identifies them as lying within a relaxed boundary on
    all 4 sides. The relaxed boundary includes any points after the first
    boundary crossing. This is because there are known limitations with
    the odd-numbered approach with this dataset, which using all 4
    directions is attempting to resolve.

    :param image: 2D array of the image pixels.
    :param roi: 2D array of x,y coordinates.
    :return: Returns a 2D array representing the image where the values
    are either -1, 0, or 1.
    """
    maxcol, maxrow = roi.max(axis=0) + 1
    mask = np.full((maxrow, maxcol), -1)
    interior_mask = np.zeros((maxrow, maxcol))
    # Set ROI boundaries as 0
    interior_mask[roi[:, 1], roi[:, 0]] = 1

    # Get the number of crossings in 4 directions
    lr = interior_mask.cumsum(axis=1) > 0
    tb = interior_mask.cumsum(axis=0) > 0
    bt = np.cumsum(interior_mask[::-1, :], axis=0)[::-1, :] > 0
    rl = np.cumsum(interior_mask[:, ::-1], axis=1)[:, ::-1] > 0
    grid_mask = lr & tb & bt & rl
    mask[grid_mask] = 1
    mask[roi[:, 1], roi[:, 0]] = 0
    return mask


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
    # Get the indices of every coordinate in the image (row, col)
    image_indices = np.indices(mask.shape).reshape(2, mask.size).T
    # Convert into (x,y) coordinates
    image_indices = np.flip(image_indices, axis=1)
    # See if the ROI contains these values
    poly_mask = Path(roi).contains_points(image_indices).reshape(mask.shape)
    # Set these values in the output to 0
    mask[poly_mask] = 1
    # Set ROI boundaries as 0
    mask[roi[:, 1], roi[:, 0]] = 0
    return mask


def find_crossing_points(corners: np.array, shape: np.array) -> np.array:
    """
    Finds the crossing points for a ray entering a given polygon.

    This implementation is based the following algorithm:
    https://www.alienryderflex.com/polygon_fill/

    :param corners: The corners that define the polygon. Must be ordered either
    clockwise or anti-clockwise.
    :param shape: The maximum shape of the polygon in (height, width).
    :return: A 2D array of shape (height, width) of type integer, where each
    value corresponds to the number of boundary crossings at this pixel.
    """
    # Setup
    height = shape[1] - 1
    crossings = np.zeros((shape[1], shape[0])).astype(int)
    corner_y = corners[:, 1]
    corner_x = corners[:, 0]
    # The algorithm iterates through the corners in path order in pairs, so
    # keep a rotated copy to use as the 'previous' corner
    prev_corner_y = np.roll(corner_y, 1)
    prev_corner_x = np.roll(corner_x, 1)

    # The original algorithm iterates row-by-row. We want to vectorise this so
    # will create data structures to hold all the corners for each row
    prev_corner_x = np.tile(prev_corner_x, height)
    prev_corner_y = np.tile(prev_corner_y, height)
    corner_x = np.tile(corner_x, height)
    corner_y = np.tile(corner_y, height)
    pixel_y = np.repeat(np.arange(height), corners.shape[0])

    # The crux of the algorithm. Find nodes, i.e. crossing points
    has_node = ((corner_y < pixel_y) & (prev_corner_y >= pixel_y)) | ((prev_corner_y < pixel_y) & (corner_y >= pixel_y))
    # Calculate the x-values at each node
    polyx = corner_x[has_node]
    polyy = corner_y[has_node]
    prevy = prev_corner_y[has_node]
    prevx = prev_corner_x[has_node]
    new_y = pixel_y[has_node]
    new_x = (polyx + (new_y - polyy) / (prevy - polyy) * (prevx - polyx)).astype(int)

    # Save the number of crossings at each node
    for x, y in zip(new_x, new_y):
        crossings[y, x] += 1

    return crossings


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

    # get corners of the polygon, needed for the following algorithm.
    corners = roi_corners(roi)

    # Find the boundary crossings
    crossing_points = find_crossing_points(corners, roi.max(axis=0) + 1)

    # Interior locations are those with an odd number of boundary crossings
    mask_vec = np.zeros((maxrow, maxcol)).astype(int)
    mask_rows = crossing_points.cumsum(axis=1)
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
    interior_mask = flood(mask, (start_row, start_col), connectivity=1)
    mask[interior_mask] = 1
    return mask


def create_type_mask_flood_fill_negative(roi: np.array) -> np.array:
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
    # Initialise the mask with extra 1 buffer around the outside so can
    # guarantee that 0,0 is outside
    maxcol, maxrow = roi.max(axis=0) + 3
    mask = np.full((maxrow, maxcol), 1)
    mask[roi[:, 1] + 1, roi[:, 0] + 1] = 0

    # Fill values inside the ROI to 1 using floodfill
    # Using the median as a guaranteed point inside the ROI.
    # Not great, but given that the ROIs must all be 8x8 then should be
    # sufficient. Else can use Shapely's representative_point
    exterior_mask = flood(mask, (0, 0), connectivity=1)
    mask[exterior_mask] = -1

    # Remove the buffer
    return mask[1:-1, 1:-1]


def extract_subimage(image: np.array, roi: np.array, method: str = "flood_fill_negative") -> SubImage:
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
        "flood_fill_negative": create_type_mask_flood_fill_negative,
        "skimage": create_type_mask_skimage,
        "ray_cast_4": create_type_mask_ray_cast_4,
    }
    alg = algs[method]
    mask_subset = alg(roi)

    return SubImage(sub_image=image_subset, type_mask=mask_subset, centroid=centroid)

"""
    cellphe.processing.roi
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions related to processing ROIs.
"""

from __future__ import annotations

import numpy as np
from roifile import ImagejRoi, roiwrite

# pylint doesn't recognise line
# pylint: disable=no-name-in-module
from skimage.draw import line


def boundary_vertices(roi: np.array) -> np.array:
    """Returns the vertices that lie directly on the boundary of an ROI, i.e.
    removing any additional vertices that do not directly impact on the interior
    area.

    :param roi: 2D array of x,y coordinates.
    :return: A 2D array of x,y coordinates.
    """
    # Create a 2D array so can vectorise the search with a buffer in every
    # direction so can get the full 3x3 window around a point
    roi = roi + 1  # Padding from top left
    xrange, yrange = roi.max(axis=0) - roi.min(axis=0) + 3  # Padding at both ends
    grid = np.zeros((yrange, xrange))
    grid[roi[:, 1], roi[:, 0]] = 1

    # Iterate through coordinates finding all those where there are at least
    while True:
        to_remove = []
        for i, (col, row) in enumerate(roi):
            # subtract itself
            num_neighbours = grid[(row - 1) : (row + 2), (col - 1) : (col + 2)].sum() - 1
            if num_neighbours < 2:
                to_remove.append(i)
        if len(to_remove) == 0:
            break
        # Remove the extraneous points from both ROI list and grid
        roi_to_remove = roi[to_remove, :]
        grid[roi_to_remove[:, 1], roi_to_remove[:, 0]] = 0
        roi = np.delete(roi, to_remove, axis=0)

    # Remove the padding
    return roi - 1


def get_diagonals(mat: np.array) -> list[np.array]:
    """Retrieves the diagonals (both forward and backward) from a matrix.

    NB: There's a manual version here, which was found to be slower than Numpy
    but it's a useful resource providing the equations relating x,y to the
    diagonal positions. https://stackoverflow.com/a/43311126/1020006

    :parameter mat: A 2D Numpy array.
    :return: A 2D matrix containing the diagonals on each row, right padded with
        zeros.
    """
    maxy, maxx = mat.shape
    max_diag = min(maxy, maxx)
    bdiag = [mat.diagonal(i) for i in range(-maxy + 1, maxx)]
    fdiag = [np.flip(mat, axis=0).diagonal(i) for i in range(-maxy + 1, maxx)]
    # Pad so can get a 2D matrix so can vectorize cumsums
    bdiag = np.array([np.pad(x, [0, max_diag - x.size]) for x in bdiag])
    fdiag = np.array([np.pad(x, [0, max_diag - x.size]) for x in fdiag])
    return bdiag, fdiag


def get_corner_mask(mat: np.array) -> np.array:
    """Finds the corners of a masked boundary.

    The boundary must be split up into its constituent dimensions, so that the
    input mat is a 2D array where the first dimension (rows) each hold another
    dimension, be it rows, columns, or diagonals in the original matrix.
    The data should 0s and 1s, where a 1 indicates a boundary pixel and a 0
    otherwise.

    Corners (shown in parentheses) are found as points where the pixels either
    change from 0 -> (1) -> 1, or 1 -> (1) -> 0. These are identified by taking
    the second difference, which will equal to -1 in both cases. The rows are
    padded with starting and trailing 0s in case the boundary lies on the
    dimension edge.

    :parameter mat: A 2D array containing the dimensions of the original matrix. See
        the description and usage for more details.
    :return: A boolean array the same dimensions as mat, where a True indicates
        that that vertex is a corner.
    """
    return np.diff(np.pad(mat, ((0, 0), (1, 1))), 2, axis=1) == -1


def roi_corners(roi: np.array) -> np.array:
    # pylint: disable=too-many-locals
    """Gets the corners from an ROI, i.e. any vertex that connect 2 sides.

    :param roi: 2D array of x,y coordinates.
    :return: A 2D array of x,y coordinates.
    """
    # Form grid
    xrange, yrange = roi.max(axis=0) - roi.min(axis=0) + 1
    grid = np.zeros((yrange, xrange))
    grid[roi[:, 1], roi[:, 0]] = 1

    # Get boolean masks of corners in all 3 dimensions
    row_corners = get_corner_mask(grid)
    col_corners = get_corner_mask(grid.T)
    bdiags, fdiags = get_diagonals(grid)
    bdiag_corners = get_corner_mask(bdiags)
    fdiag_corners = get_corner_mask(fdiags)

    # Get these as indices of the 2D boundary layers
    row_rows, row_cols = np.nonzero(row_corners)
    col_cols, col_rows = np.nonzero(col_corners)
    bdiag_rows, bdiag_cols = np.nonzero(bdiag_corners)
    fdiag_rows, fdiag_cols = np.nonzero(fdiag_corners)

    # Convert back to coordinates of the original grid
    # Trivial for the row and column dimensions
    row_corners = np.vstack((row_rows, row_cols)).T
    col_corners = np.vstack((col_rows, col_cols)).T
    # Bit more involved for the diagonals!
    n_rows = grid.shape[0]
    fdiag_rows_2 = np.minimum(fdiag_rows - fdiag_cols, n_rows - 1 - fdiag_cols)
    fdiag_cols_2 = fdiag_rows - fdiag_rows_2
    fdiag_corners = np.vstack((fdiag_rows_2, fdiag_cols_2)).T

    bdiag_rows_2 = np.maximum(n_rows - bdiag_rows + bdiag_cols - 1, bdiag_cols)
    bdiag_cols_2 = np.maximum(bdiag_rows - n_rows + bdiag_cols + 1, bdiag_cols)
    bdiag_corners = np.vstack((bdiag_rows_2, bdiag_cols_2)).T

    # Combine and get unique set
    all_corners = np.vstack((row_corners, col_corners, fdiag_corners, bdiag_corners))
    all_corners = np.unique(all_corners, axis=0)

    # Flip back to x,y
    all_corners = np.flip(all_corners, axis=1)

    # Return in the order of the original ROI path
    roi_order = np.array([np.where((roi == x).all(axis=1))[0][0] for x in all_corners]).argsort()
    corner_order = np.arange(all_corners.shape[0])[roi_order]

    return all_corners[corner_order]


def interpolate_between_points(coords: np.array) -> np.array:
    """
    Interpolates between the ROI coordinates to ensure there aren't any breaks
    in the boundary.

    All the downstream CellPhe analysis assumes that there aren't any gaps in
    the ROIs. This function guarantees that.

    :param coords: A 2D Numpy array of coordinate pairs in the form (x,y).
    :return: An array in the same format as coords with either the same number
        of coordinates, or more.
    """
    new_coords_raw = []
    # Iterate through each consecutive coordinate pair interpolating a
    # line between them
    # NB: assumes that ROIs are stored in order, which they should be from
    # TrackMate
    for i in range(1, coords.shape[0]):
        # These are in format (y, x) (i.e. row/column)
        new_coords_raw.append(line(coords[i - 1, 1], coords[i - 1, 0], coords[i, 1], coords[i, 0]))
    # Ensure first and last point are connected
    new_coords_raw.append(line(coords[-1, 1], coords[-1, 0], coords[0, 1], coords[0, 0]))
    # Convert back to a single 2D numpy array in (y,x) format
    new_coords = np.concatenate([np.asarray(x).T for x in new_coords_raw])
    # Remove duplicate coordinates
    # Have to do a 2-step process as np.unique() sorts by default and cannot
    # be told not to
    _, inds = np.unique(new_coords, axis=0, return_index=True)
    new_coords = new_coords[np.sort(inds)]

    # Convert back to (x,y)
    new_coords = np.flip(new_coords, axis=1)
    return new_coords


def save_rois(rois: list[dict], filename: str = "rois.zip"):
    """
    Saves ROIs to disk.

    :param rois: List of dicts, each one representing an ROI with elements:
        - coords: 2D numpy array containing the ROI coordinates.
        - CellID: Cell ID
        - FrameID: Frame ID
        - filename: Filename to save the ROI to
    :param filename: Filename of output archive.
    :return: None, writes to disk as a side-effect.
    """
    roi_objs = []
    for roi in rois:
        new_coords = interpolate_between_points(roi["coords"].astype(int))
        roi_obj = ImagejRoi.frompoints(new_coords)
        roi_obj.position = roi["FrameID"]
        roi_obj.name = roi["Filename"]
        roi_objs.append(roi_obj)

    roiwrite(filename, roi_objs)

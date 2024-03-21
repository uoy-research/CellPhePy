"""
    cellphe.features.frame
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions for extracting frame-level features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pybind11_rdp import rdp
from scipy.spatial.distance import pdist, squareform


def extract_features(_df: pd.DataFrame, _roi_folder: str, _frame_folder: str, _framerate: float) -> pd.DataFrame:
    r"""
     Calculates cell features from timelapse videos

     Calculates 74 features related to size, shape, texture and movement for each cell on every non-missing frame,
     as well as the cell density around each cell on each frame.
     NB: while the ROI filenames are expected to be provided in ``df`` and found
     in ``roi_folder``,
     the frame filenames are just expected to follow the naming convention
     ``<some text>-<FrameID>.tiff``,
     where FrameID is a 4 digit leading zero-padded number, corresponding to the
     ``FrameID`` column in ``df``.

     :param df: DataFrame where every row corresponds to a combination of a cell
     tracked in a frame. It must have at least columns ``CellID``, ``FrameID`` and
     ``ROI_filename`` along with any additional features.
     :param roi_folder: A path to a directory containing multiple Report Object Instance
     (ROI) files named in the format ``cellid``-``frameid``.roi
     :param frame_folder: A path to a directory containing multiple frames in TIFF format.
     It is assumed these are named under the pattern ``<experiment
     name>-<frameid>.tif``, where
     ``<frameid>`` is a 4 digit zero-padded integer.
     :param framerate: The frame-rate, used to provide a meaningful measurement unit for velocity,
        otherwise a scaleless unit is implied with ``framerate=1``.
    :return: A dataframe with 77+N columns (where N is the number of imported features)
     and 1 row per cell per frame it's present in:
       * ``FrameID``: the numeric frameID
       * ``CellID``: the numeric cellID
       * ``ROI_filename``: the ROI filename
       * ``...``: 74 frame specific features
       * ``...``: Any other data columns that were present in ``df``
    """
    return None


def var_from_centre(boundaries: np.array) -> list[float]:
    """
    Determines the distance of boundary conditions from the centre.

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.

    :return: A tuple of the mean distance from the centre and the variance.
    """
    means = boundaries.mean(axis=0)
    distances = np.linalg.norm(boundaries - means, axis=1)
    return np.mean(distances), np.var(distances, ddof=1)


def curvature(boundaries: np.array, gap: int) -> float:
    """
    Identifies the curvature of a boundary condition.

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.
    :param gap: The gap.

    :return: The curvature as a float.
    """
    # Create index array
    n_points = boundaries.shape[0]
    indices = np.arange(n_points)
    # Offset the indices by the gaps in both directions
    indices_mgap = (indices + n_points - gap) % n_points
    indices_pgap = (indices + gap) % n_points

    # Get the coordinates in these orderings
    boundaries_mgap = boundaries[indices_mgap,]
    boundaries_pgap = boundaries[indices_pgap,]

    # Calculate the euclidean distance between them
    i1 = np.linalg.norm(boundaries - boundaries_mgap, axis=1)
    i2 = np.linalg.norm(boundaries - boundaries_pgap, axis=1)
    i3 = np.linalg.norm(boundaries_mgap - boundaries_pgap, axis=1)

    # Calculate curvature
    return np.mean(i1 + i2 - i3)


def minimum_box(boundaries: np.array) -> np.array:
    """
    Finds the minimum box around some boundary coordinates.

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.

    :return: A 1D numpy array of an [x,y] pair.
    """
    n_points = boundaries.shape[0]
    distances = squareform(pdist(boundaries))
    max_dist = distances.argmax()
    row = max_dist // n_points
    col = max_dist % n_points

    keepy1 = boundaries[row, 1]
    keepx1 = boundaries[row, 0]
    keepy2 = boundaries[col, 1]
    keepx2 = boundaries[col, 0]

    alpha = np.arctan((keepy1 - keepy2) / (keepx1 - keepx2))
    # rotating points by -alpha makes keepx1-keepx2 lie along x-axis
    roty = keepy1 - np.sin(alpha) * (boundaries[:, 0] - keepx1) + np.cos(alpha) * (boundaries[:, 1] - keepy1)
    rotx = keepx1 + np.cos(alpha) * (boundaries[:, 0] - keepx1) + np.sin(alpha) * (boundaries[:, 1] - keepy1)

    return np.array([rotx.max() - rotx.min(), roty.max() - roty.min()])


def polygon(boundaries: np.array) -> np.array:
    """
    Calculates the minimal polygon around a set of points using the
    Ramer-Douglas-Peucker method.
    Uses the pybind11_rdp implementation.
    https://github.com/cubao/pybind11-rdp

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.

    :return: A 2D array comprising the minimal set of points.

    """
    # The original implementation had epsilon hardcoded to 2.5
    # It also didn't return the last point in the way this implementation does.
    return rdp(boundaries, epsilon=2.5)


def polygon_features(boundaries: np.array) -> np.array:
    """
    Derives features from the minimal polygon surrounding the boundary
    coordinates.

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.

    :return: A 1D array with 4 values:
        -[0] The longest edge
        -[1] The smallest interior angle
        -[2] The variance of the interior angles
        -[3] The variance of the edges

    """
    # Fit reduced polygon
    points = polygon(boundaries)

    # Form tensor of these points rotated
    points1 = np.roll(points, -1, axis=0)
    points2 = np.roll(points, -2, axis=0)
    mat_all = np.stack([np.stack([points, points1]), np.stack([points, points2]), np.stack([points1, points2])])

    # For each point, calculate distance between:
    #   - Original point and rotated by 1
    #   - Original point and rotated by 2
    #   - Rotated by 1 and rotated by 2
    # These are the edges
    matt_differences = mat_all[:, 0, :, :] - mat_all[:, 1, :, :]
    lengths = np.linalg.norm(matt_differences, axis=2)
    # Convert into the same order / format as the original version
    lengths = lengths.transpose()[:, np.array([0, 2, 1])]

    # Determine the interior angles
    angles = polygon_angle(lengths)

    # Calculate features
    min_angle = np.min(angles)
    var_angle = np.var(angles, ddof=1)
    max_length = np.max(lengths[:, 0])
    var_length = np.var(lengths[:, 0], ddof=1)

    return np.array([max_length, min_angle, var_angle, var_length])


def polygon_angle(points: np.array) -> np.array:
    """
    Calculate interior angles from a polygon.

    :param points: An N x 3 matrix.

    :return: A 1D array of length N, each entry representing an angle.
    """
    points_squared = points**2
    calc = (points_squared[:, 0] + points_squared[:, 1] - points_squared[:, 2]) / (2.0 * points[:, 0] * points[:, 1])
    res_acos = np.arccos(calc)
    res_acos[np.abs(calc - 1) <= 0.001] = 2 * np.pi
    return res_acos

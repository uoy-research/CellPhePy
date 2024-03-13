"""
    cellphe.features.frame
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions for extracting frame-level features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_features(_df: pd.DataFrame, _roi_folder: str, _frame_folder: str, _framerate: float) -> pd.DataFrame:
    r"""
     Calculates cell features from timelapse videos

     Calculates 74 features related to size, shape, texture and movement for each cell on every non-missing frame,
     as well as the cell density around each cell on each frame.
     NB: while the ROI filenames are expected to be provided in \code{df} and found in \code{roi_folder},
     the frame filenames are just expected to follow the naming convention \code{<some text>-<FrameID>.tiff},
     where FrameID is a 4 digit leading zero-padded number, corresponding to the \code{FrameID} column in \code{df}.

     :param df: DataFrame where every row corresponds to a combination of a cell
     tracked in a frame. It must have at least columns \code{CellID}, \code{FrameID} and
     \code{ROI_filename} along with any additional features.
     :param roi_folder: A path to a directory containing multiple Report Object Instance
     (ROI) files named in the format \code{cellid}-\code{frameid}.roi
     :param frame_folder: A path to a directory containing multiple frames in TIFF format.
     It is assumed these are named under the pattern \code{<experiment name>-<frameid>.tif}, where
     \code{<frameid>} is a 4 digit zero-padded integer.
     :param framerate: The frame-rate, used to provide a meaningful measurement unit for velocity,
        otherwise a scaleless unit is implied with \code{framerate=1}.
    :return: A dataframe with 77+N columns (where N is the number of imported features)
     and 1 row per cell per frame it's present in:
     \itemize{
       \item{\code{FrameID}: the numeric frameID}
       \item{\code{CellID}: the numeric cellID}
       \item{\code{ROI_filename}: the ROI filename}
       \item{\code{...}: 74 frame specific features}
       \item{\code{...}: Any other data columns that were present in \code{df}}
     }
    """
    return None


def var_from_centre(boundaries: np.array) -> list[float]:
    """
    Determines the distance of boundary conditions from the centre.

    :param boundaries: A 2D array of [[x1, y1], [x2, y2], ..., [xn, yn]] pairs.

    :return: A tuple of the mean distance from the centre and the variance.
    """
    means = boundaries.mean(axis=0)
    dists_from_centre = np.power(boundaries - means, 2)
    distances = np.sqrt(dists_from_centre.sum(axis=1))
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

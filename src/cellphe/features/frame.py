"""
    cellphe.features.frame
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions for extracting frame-level features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from pybind11_rdp import rdp
from scipy.spatial.distance import pdist, squareform

from cellphe.processing import extract_subimage, normalise_image


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


def skewness(x: np.array) -> float:
    """
    Calculates the skewness of a sample.

    It uses the type 2method in the R e1071::skewness implementation, which is
    the version used in SAS and SPSS according to the documentation.

    :param x: Sample.
    :return: A float representing the skewness.
    """
    mu = x.mean()
    n = x.size
    deltas = x - mu
    m2 = np.sum(np.power(deltas, 2)) / n
    m3 = np.sum(np.power(deltas, 3)) / n
    g1 = m3 / np.power(m2, 3 / 2)
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    return G1


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
    points = rdp(boundaries, epsilon=2.5)
    return points


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


def cooccurrence_matrix(image1: np.array, image2: np.array, mask: np.array, levels: int) -> np.array:
    """
    Calculate cooccurrence matrix between 2 images downscaled to a certain
    level.

    :param image1: The first image as a 2D numpy array.
    :param image2: The second image as a 2D numpy array.
    :param mask: A boolean mask with the same dimensions as image1 and image2
    :param levels: Number of grayscale levels to downscale to.
    :return: Returns a levels x levels matrix of the cooccurrences of each level
    between the 2 images.
    """
    # Rescale both images to levels using the normalise_image function
    image1_rescaled = np.floor(normalise_image(image1, 0, levels)[mask])
    image2_rescaled = np.floor(normalise_image(image2, 0, levels)[mask])
    # Restrict to positive values
    to_keep = (image1_rescaled > 0) & (image2_rescaled > 0)

    # do the coccurrence calculation. This is using some indexing witchcraft
    # from Julie's code
    values, counts = np.unique(image1_rescaled[to_keep] + levels * (image2_rescaled[to_keep] - 1), return_counts=True)
    values = values.astype("int")

    # Format counts into pairwise matrix
    cooc = np.zeros((levels, levels))
    # NB: can't do 2D array indexing with 1D array as can in R
    # so need to convert into x&y
    x_vals = (values - 1) % levels
    y_vals = (values - 1) // levels
    cooc[x_vals, y_vals] = counts
    return cooc


def haralick(cooc: np.array) -> np.array:
    """
    Calculates Haralick features from the given cooccurrence matrix.

    :param cooc: Cooccurrence matrix.
    :return: A Numpy array of size 14 corresponding to each of the features.
    """
    o_hara = np.zeros(14)
    pglcm = cooc / cooc.sum()
    pglcm_raw = pglcm.copy()
    pglcm[pglcm == 0] = np.nan  # Silence numpy warnings
    nx = pglcm.shape[0]
    px = np.nansum(pglcm, axis=0)
    py = np.nansum(pglcm, axis=1)

    pxpy = np.repeat(px, nx).reshape(nx, nx) * np.repeat(py, nx).reshape(nx, nx, order="F")
    pxpy[pxpy == 0] = np.nan  # Removing numpy warnings
    px_y = np.zeros(2 * nx)
    pxmy = np.zeros(nx)
    vx = np.arange(1, nx + 1)
    mx = np.sum(px * vx)
    my = np.sum(py * vx)
    stdevx = np.sum(px * (vx - mx) ** 2)
    stdevy = np.sum(py * (vx - my) ** 2)
    hxy1_0 = pglcm * np.log10(pxpy)
    hxy2_0 = pxpy * np.log10(pxpy)
    hxy2 = -np.nansum(hxy2_0)
    op = np.arange(1, nx + 1).repeat(nx).reshape(nx, nx)
    oq = op.transpose()
    spq = op + oq
    dpq = np.abs(op - oq)
    o_hara[0] = np.nansum(pglcm**2)
    o_hara[1] = np.nansum(dpq**2 * pglcm)
    o_hara[2] = np.nansum(pglcm / (1 + dpq**2))
    o_hara[3] = -np.nansum(pglcm * np.log10(pglcm))
    stdev_mult = stdevx * stdevy
    if stdev_mult == 0:
        o_hara[4] = 0
    else:
        o_hara[4] = np.nansum((op - mx) * (oq - my) * pglcm / (np.sqrt(stdev_mult)))
    o_hara[5] = np.nansum((op - ((mx + my) / 2)) ** 2 * pglcm)
    o_hara[6] = np.nansum(spq * pglcm)
    sen = np.zeros(2 * nx)
    den_1 = np.zeros(nx)
    den_2 = np.zeros(nx)
    pglcm2 = pglcm[:, ::-1]
    sen[0] = pglcm2[0, nx - 1]
    den_1[0] = pglcm[0, nx - 1]
    for i in range(1, nx):
        rows = np.arange(i + 1)
        cols = rows + nx - i - 1
        sen[i] = np.nansum(pglcm2[rows, cols])
        den_1[i] = np.nansum(pglcm[rows, cols])
    for i in range(nx - 2):
        rows = np.arange(i + 1, nx)
        cols = np.arange(10 - i - 1)
        sen[i + nx] = np.nansum(pglcm2[rows, cols])
        den_2[nx - i - 2] = np.nansum(pglcm[rows, cols])
    sen[nx + nx - 2] = pglcm2[nx - 1, 0]
    sen[sen == 0] = np.nan
    den_2[0] = pglcm[nx - 1, 0]
    o_hara[7] = -np.nansum(sen * np.log10(sen))
    den = den_1 + den_2
    den[den == 0] = np.nan
    o_hara[8] = -np.nansum(den * np.log10(den))
    o_hara[9] = np.nansum(((dpq - o_hara[8]) ** 2) * pglcm)
    o_hara[10] = np.nansum(((spq - o_hara[7]) ** 2) * pglcm)
    o_hara[11] = np.sqrt(1 - np.exp(-2 * np.abs(hxy2 - o_hara[3])))
    spq_mx_my = spq - mx - my
    o_hara[12] = np.nansum(spq_mx_my**3 * pglcm)
    o_hara[13] = np.nansum(spq_mx_my**4 * pglcm)
    return o_hara


def intensity_quantiles(pixels: np.array) -> np.array:
    """
    Calculates the coefficient of variation in distance between pixels at
    different quantiles of intensity.

    :param pixels: A 2D array with 3 columns corresponding to x, y, and
    intensity.
    :return: A 1D array with length 9, corresponding to the coefficient of
    variation between pixel distances at different quantile thresholds
    (0.1-0.9).
    """
    quantiles = np.quantile(pixels[:, 2], np.arange(0.1, 1.0, 0.1))

    vals = np.zeros(9)
    for i, thresh in enumerate(quantiles):
        pixels_greater_thresh = pixels[:, 2] >= thresh
        dist_pixels_thresh = pdist(pixels[pixels_greater_thresh, 0:2])
        vals[i] = np.var(dist_pixels_thresh, ddof=1) / np.mean(dist_pixels_thresh)
    return vals


def haar_approximation(image: np.array) -> np.array:
    """
    Calculates the approximation coefficients of a 2D db1 (aka Haar) wavelet transform.

    :param image: 2D numpy array containing the image pixels.
    :return: A 2D numpy array containing the approximation coefficients.
    """
    cA, [cH, cV, cD] = pywt.dwt2(image, "db1")
    return cA / 2.0


def double_image(image: np.array) -> np.array:
    """
    Doubles the size of an image.

    :param image: 2D numpy array representing the downscaled image with
    dimensions m x n.
    :return: A 2D numpy array with dimensions 2m x 2n
    """
    return image.repeat(2, axis=0).repeat(2, axis=1)


def extract_static_features(image: np.array, roi: np.array) -> np.array:
    """
    Extracts the 68 frame-level static (i.e. no movement based) features for a given image and roi.

    :param image: The image as a 2D Numpy array.
    :param roi: The region of interest as an Mx2 Numpy array.
    :return: A 1D array of length 68 containing the features.
    """
    feats = np.zeros(69)

    # Extract sub image info
    sub_image = extract_subimage(image, roi)

    # Shape features
    # Averag radius & variance of boundary pixel distance to centre
    feats[0:2] = var_from_centre(roi)
    # boundary curvature
    feats[2] = curvature(roi, 4)
    # width and height of minimal box
    minbox = minimum_box(roi)
    feats[3:5] = minbox
    # cell area
    feats[5] = np.sum(sub_image.type_mask >= 0)
    # area to boundary ratio
    feats[6] = feats[5] / roi.shape[0] ** 2
    # minimal box to area ratio
    feats[7] = minbox.prod() / feats[5]
    # rectangularity
    feats[8] = minbox.max() / minbox.sum()
    # Polygon features
    feats[9:13] = polygon_features(roi)

    # Texture features
    # First order
    cell_mask = sub_image.type_mask >= 0
    cell_intensities = sub_image.sub_image[cell_mask]
    feats[13] = cell_intensities.mean()
    feats[14] = cell_intensities.std(ddof=1)
    feats[15] = skewness(cell_intensities)

    # Cooccurrence
    n_cooccurrences = 10
    level1 = haar_approximation(sub_image.sub_image)
    level2 = haar_approximation(level1)
    level1 = double_image(level1)
    level2 = double_image(double_image(level2))
    orig_dims = sub_image.sub_image.shape
    level1 = level1[: orig_dims[0], : orig_dims[1]]
    level2 = level2[: orig_dims[0], : orig_dims[1]]
    # TODO Debug cooccurrence matrix and see if it gives the same values as R
    # The indexing at the end might be wrong!
    cooc01 = cooccurrence_matrix(sub_image.sub_image, level1, cell_mask, n_cooccurrences)
    cooc02 = cooccurrence_matrix(sub_image.sub_image, level2, cell_mask, n_cooccurrences)
    cooc12 = cooccurrence_matrix(level1, level2, cell_mask, n_cooccurrences)
    feats[16:30] = haralick(cooc01)
    feats[30:44] = haralick(cooc12)
    feats[44:58] = haralick(cooc02)

    # Intensity quantiles
    interior_mask = sub_image.type_mask == 1
    vals = sub_image.sub_image[interior_mask]
    coords = np.asarray(np.where(interior_mask)).T
    interior_pixels = np.concatenate((coords, vals.reshape(vals.size, 1)), axis=1)
    feats[58:67] = intensity_quantiles(interior_pixels)
    feats[67:69] = sub_image.centroid

    return feats

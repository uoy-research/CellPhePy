"""
    cellphe.features.time_series
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Functions for extracting features from time-series.
"""

from __future__ import annotations

import janitor  # noqa: F401 # pylint: disable=unused-import
import numpy as np
import pandas as pd
import pywt

from cellphe.features.helpers import skewness


def skewness_positive(x: np.array) -> float:
    """
    Calculates the skewness of an array.

    If the array doesn't have at least one positive value then it returns 0.

    :param x: Input array.
    :return: Either the skewness or 0, depending if the array has no positive
    values.
    """
    if x.max() > 0:
        res = skewness(x)
    else:
        res = 0
    return res


def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolates a dataframe with missing frames.

    The resultant dataframe will have more rows than the input if at minimum one
    cell is missing from one frame. All feature columns will be linearly
    interpolated during these missing frames.

    :param df: The DataFrame with columns CellID, FrameID and any other feature
    columns.
    :return: A DataFrame with the same column structure as df, but with either
    the same number of rows or greater.
    """
    frame_range = {"FrameID": lambda x: range(x["FrameID"].min(), x["FrameID"].max())}
    all_cell_frames = df[["CellID", "FrameID"]].complete(frame_range, by="CellID")
    df = all_cell_frames.merge(df, on=["CellID", "FrameID"], how="left")
    df.interpolate(inplace=True)
    df.sort_values(by=["CellID", "FrameID"], inplace=True)
    return df


def ascent(x: np.array, diff: bool = True) -> float:
    """
    Calculates the ascent of a signal.

    This is defined as the sum of the point-to-point positive differences,
    divided by the total length of the signal.

    :param x: Input array.
    :param diff: Whether to take the difference first (required for elevation
    variables but not those from wavelets).
    :return: A float representing the ascent.
    """
    if diff:
        n = x.size
        x = np.diff(x)
    else:
        n = x.size
    return np.sum(x[x > 0]) / n


def descent(x: np.array, diff: bool = True) -> float:
    """
    Calculates the descent of a signal.

    This is defined as the sum of the point-to-point negative differences,
    divided by the total length of the signal.

    :param x: Input array.
    :param diff: Whether to take the difference first (required for elevation
    variables but not those from wavelets).
    :return: A float representing the descent.
    """
    if diff:
        n = x.size
        x = np.diff(x)
    else:
        n = x.size
    return np.sum(x[x < 0]) / n


def haar_approximation_1d(x: pd.Series) -> list(np.array):
    """
    Haar wavelet approximation for a 1D signal with 3 levels of decomposition.

    :param x: The input signal.
    :return: A list of length 3 for each level, with each entry containing the
    detail coefficients.
    """

    def remove_last_value_if_odd(signal, approx, detail):
        is_odd = signal.size % 2
        detail = detail[: detail.size - is_odd]
        approx = approx[: approx.size - is_odd]
        return approx, detail

    a1, d1 = pywt.dwt(x, "db1")
    a1, d1 = remove_last_value_if_odd(x, a1, d1)
    a2, d2 = pywt.dwt(a1 / np.sqrt(2), "db1")
    a2, d2 = remove_last_value_if_odd(a1, a2, d2)
    a3, d3 = pywt.dwt(a2 / np.sqrt(2), "db1")
    a3, d3 = remove_last_value_if_odd(a2, a3, d3)
    output = [d1, d2, d3]
    return output


def wavelet_features(x: pd.Series) -> pd.DataFrame:
    """
    Calculates the elevation metrics for the detail coefficients from 3 levels
    of a Haar wavelet approximation.

    :param x: The raw data as an array.
    :return: A 1-row DataFrame comprising 9 columns, one for each of the 3
    elevation metrics for each of the 3 Wavelet levels.
    """
    wave_coefs = haar_approximation_1d(x)

    # For each set of wavelet coefficients calculate the elevation metrics
    wave_coefs_dict = {f"l{i + 1}": x for i, x in enumerate(wave_coefs)}
    metrics = {"asc": lambda x: ascent(x, diff=False), "des": lambda x: descent(x, diff=False), "max": np.max}
    res_dict = {f"{kw}_{km}": vm(vw) for kw, vw in wave_coefs_dict.items() for km, vm in metrics.items()}

    # Convert into DataFrame for output
    return pd.DataFrame(res_dict, index=[0])


def calculate_trajectory_area(df) -> float:
    """
    Calculates the trajectory area of a cell.

    :param xs: An array of x-coordinates.
    :param ys: An array of y-coordinates.
    :return: The trajectory area as a float.
    """
    return ((df["x"].max() - df["x"].min()) * (df["y"].max() - df["y"].min())) / df["x"].size


def time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 15 time-series based features for each frame-level feature.

    :param df: A DataFrame as output from extract.features.
    :return: A DataFrame with CellID then 15*F+1 columns, where F is the number
    of feature columns in df.
    """
    # Remove columns that aren't used, as they aren't either unique identifiers
    # or feature columns
    df.drop(columns=["ROI_filename"], inplace=True)
    feature_cols = np.setdiff1d(df.columns.values, ["CellID", "FrameID", "x", "y"])

    # Calculate summary statistics
    summary_vars = df.groupby("CellID", as_index=False)[feature_cols].agg(["mean", "std", skewness_positive])
    summary_vars.rename(columns={"skewness_positive": "skew"}, inplace=True)

    # Interpolate any missing frames
    interpolated = interpolate(df)

    # Calculate elevation metrics
    grouped = interpolated.groupby(["CellID"], as_index=False)
    ele_vars = grouped[feature_cols].agg([ascent, descent, "max"])
    ele_vars.rename(columns={"ascent": "asc", "descent": "des"}, inplace=True)

    # Calculate variables from wavelet details
    # This is a nested loop: the groupby apply sends a DataFrame for each
    # cell to an anonymous function that then applies the 'wavelet_features'
    # function to every column, which returns 9 columns (3 wavelet levels x 3
    # elevation variables).
    # I can't see a one-liner way of doing this, as the aggregate functions
    # accept functions that only return a single scalar, not the 9 columns here.
    # The alternative is to have 9 separate functions in a single .agg call, but
    # that means running the Wavelet decomposition 9 times rather than once.
    wave_vars = interpolated.groupby(["CellID"], as_index=True)[feature_cols].apply(lambda x: x.agg([wavelet_features]))

    # Calculate trajectory area for each cell
    # The trajectory difference is due to the R code using the number of
    # frames before interpolation as the denominator, whereas this code uses the
    # full range of frames (i.e. max - min frame number) that cell was observed
    # in, which I think is more correct.
    traj_vars = grouped.apply(calculate_trajectory_area, include_groups=False)

    # Prepare for combination
    # Set CellID as index for easy joining, and merge hierarchical column names
    summary_vars.set_index("CellID", inplace=True)
    summary_vars.columns = summary_vars.columns.map("_".join).str.strip("|")
    ele_vars.set_index("CellID", inplace=True)
    ele_vars.columns = ele_vars.columns.map("_".join).str.strip("|")
    # Remove unused middle level from both columns and indices, which arose from
    # the double loop
    wave_vars.columns = wave_vars.columns.droplevel(1)
    wave_vars.index = wave_vars.index.droplevel(1)
    wave_vars.columns = wave_vars.columns.map("_".join).str.strip("|")
    traj_vars.set_index("CellID", inplace=True)
    traj_vars.columns.values[0] = "trajArea"

    # Combine!
    comb = summary_vars.join([ele_vars, wave_vars, traj_vars])
    comb.reset_index(inplace=True)

    return comb

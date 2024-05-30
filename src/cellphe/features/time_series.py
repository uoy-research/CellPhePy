"""
    cellphe.features.time_series
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Functions for extracting features from time-series.
"""

from __future__ import annotations

import janitor
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


def ascent(x: np.array) -> float:
    """
    Calculates the ascent of a signal.

    This is defined as the sum of the point-to-point positive differences,
    divided by the total length of the signal.

    :param x: Input array.
    :return: A float representing the ascent.
    """
    x_diff = np.diff(x)
    return np.sum(x_diff[x_diff > 0]) / x.size


def descent(x: np.array) -> float:
    """
    Calculates the descent of a signal.

    This is defined as the sum of the point-to-point negative differences,
    divided by the total length of the signal.

    :param x: Input array.
    :return: A float representing the descent.
    """
    x_diff = np.diff(x)
    return np.sum(x_diff[x_diff < 0]) / x.size


def haar_approximation_1d(x: pd.Series) -> list(np.array):
    """
    Haar wavelet approximation for a 1D signal with 3 levels of decomposition.

    :param x: The input signal.
    :return: A list of length 3 for each level, with each entry containing the
    detail coefficients.
    """

    def remove_last_value_if_odd(approx, detail):
        is_odd = approx.size % 2
        return detail[: detail.size - is_odd]

    a1, d1 = pywt.dwt(x, "db1")
    d1 = remove_last_value_if_odd(x, d1)
    a2, d2 = pywt.dwt(a1 / np.sqrt(2), "db1")
    d2 = remove_last_value_if_odd(a1, d2)
    _, d3 = pywt.dwt(a2 / np.sqrt(2), "db1")
    d3 = remove_last_value_if_odd(a2, d3)
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
    wave_coefs_dict = {f"l{i+1}": x for i, x in enumerate(wave_coefs)}
    metrics = {"ascent": ascent, "descent": descent, "max": np.max}
    res_dict = {f"{kw}_{km}": vm(vw) for kw, vw in wave_coefs_dict.items() for km, vm in metrics.items()}

    # Convert into DataFrame for output
    return pd.DataFrame(res_dict, index=[0])


def time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 15 time-series based features for each frame-level feature.

    :param df: A DataFrame as output from extract.features.
    :return: A DataFrame with CellID then 15*F+1 columns, where F is the number
    of feature columns in df.
    """
    # Remove columns that aren't used, as they aren't either unique identifiers
    # or feature columns
    df.drop(columns=["ROI_filename", "xpos", "ypos"], inplace=True)
    feature_cols = np.setdiff1d(df.columns.values, ["CellID", "FrameID"])

    # Calculate summary statistics
    summary_stats = df.groupby("CellID", as_index=False)[feature_cols].agg(["mean", "std", "max", skewness_positive])

    # Interpolate any missing frames
    interpolated = interpolate(df)

    # Calculate elevation metrics
    ele_vars = interpolated.groupby(["CellID"], as_index=False)[feature_cols].agg([ascent, descent, "max"])

    # Calculate variables from wavelet details
    # For each wavelet level, calculate the 3 elevation vars
    # This is a nested loop: the groupby apply here sends a DataFrame for each
    # cell to an anonymous function that then applies the 'wavelet_features'
    # function to every column. I can't see a one-liner way of doing this in
    # Pandas, as all the agg/aggregate functions expect a function that only
    # returns a single scalar, rather than the 9 returned here (3 wavelet levels
    # * 3 elevation features). We want to calculate all 9 in 1 function to save
    # repeatedly calculating the Wavelet decomposition
    wave_vars = interpolated.groupby(["CellID"], as_index=True)[feature_cols].apply(lambda x: x.agg([wavelet_features]))

    # Calculate trajectory area

    # Combine

    return summary_stats

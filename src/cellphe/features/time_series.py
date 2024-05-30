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


# TODO implement on wide dataframe?
def ascent(x: np.array) -> float:
    """
    Calculates the ascent of a signal.

    This is defined as the sum of the point-to-point positive differences,
    divided by the total length of the signal.

    :param x: Input array.
    :return: A float representing the ascent.
    """
    return np.sum(x[x > 0]) / x.size


def descent(x: np.array) -> float:
    """
    Calculates the descent of a signal.

    This is defined as the sum of the point-to-point negative differences,
    divided by the total length of the signal.

    :param x: Input array.
    :return: A float representing the descent.
    """
    return np.sum(x[x < 0]) / x.size


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
    # Identify time-series feature column names
    all_cols = df.columns.values
    frame_features = np.setdiff1d(all_cols, ["CellID", "FrameID"])

    # Calculate summary statistics
    summary_stats = df.groupby("CellID")[frame_features].agg(["mean", "std", "max", skewness_positive])

    # Interpolate any missing frames and convert to long
    interpolated = interpolate(df)
    interpolated_long = interpolated.melt(id_vars=["CellID", "FrameID"], value_vars=frame_features)
    interpolated_long.sort_values(by=["variable", "CellID", "FrameID"], inplace=True)

    # Calculate elevation metrics
    # Take difference between consecutive vals, ascent is sum of positive
    # differences / n vals and descent is sum of negative differences / n vals
    interpolated_long["diff"] = interpolated_long.groupby(["variable", "CellID"])["value"].diff()
    grouped = interpolated_long.groupby(["variable", "CellID"], as_index=False)
    ele_vars = grouped["diff"].agg([ascent, descent])
    max_vars = grouped["value"].agg(["max"])

    # Calculate variables from wavelet details
    # Looks straight forward ish. Calculate wavelet and get the DETAIL
    # coefficients and then do the same elevation methods

    # Calculate trajectory area

    # Combine

    return summary_stats

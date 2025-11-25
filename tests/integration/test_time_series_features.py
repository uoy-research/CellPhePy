from __future__ import annotations

import re

import pandas as pd
import pytest
from PIL import Image

from cellphe.features.time_series import time_series_features
from cellphe.input import read_roi

pytestmark = pytest.mark.integration


@pytest.fixture()
def frame_df():
    df = pd.read_csv("tests/resources/benchmark_features.csv")
    df.rename(columns={"xpos": "x", "ypos": "y"}, inplace=True)
    yield df


@pytest.fixture()
def ts_df():
    yield pd.read_csv("tests/resources/benchmark_time_series_features.csv")


def assert_frame_equal_extended_diff(df1, df2):
    cols = df1.columns.values
    incorrect_cols = []
    for col in cols:
        try:
            pd.testing.assert_frame_equal(df1.loc[:, [col]], df2.loc[:, [col]], check_dtype=False)
        except AssertionError as e:
            print(e, "\n")
            incorrect_cols.append(col)

    if len(incorrect_cols) > 0:
        raise AssertionError(
            f"Following columns had errors: {', '.join(incorrect_cols)}\n{len(incorrect_cols)}/{len(cols)} ({len(incorrect_cols)/len(cols)*100:.2f}%)"
        )


def test_time_series_features(frame_df, ts_df):
    # R package uses 'xpos' and 'ypos' - Python package uses 'x' and 'y'
    output = time_series_features(frame_df)
    output.rename(columns={"x": "xpos", "y": "ypos"}, inplace=True)
    assert_frame_equal_extended_diff(ts_df.reset_index(drop=True), output.reset_index(drop=True))


def test_error_3rd_level_wavelet(frame_df, ts_df):
    # R package uses 'xpos' and 'ypos' - Python package uses 'x' and 'y'
    df = frame_df.loc[frame_df["FrameID"] < 6]
    output = time_series_features(df)

    # All third level wavelet features should be NA
    wavelet_regex = re.compile("_l3_")
    wavelet_cols = list(filter(wavelet_regex.search, output.columns.values))
    assert output[wavelet_cols].isna().all().all()
    assert output.shape[1] == ts_df.shape[1]  # Correct # cols


def test_error_3rd_level_wavelet_some_cells(frame_df, ts_df):
    # Test with 1 short time series (that will error on wavelet) and 1 long
    df = frame_df.query("CellID == 30 | (CellID == 31 & FrameID <= 5)")
    output = time_series_features(df)

    # All third level wavelet features for CellID 31 should be NA, but not NA for 30
    wavelet_regex = re.compile("_l3_")
    wavelet_cols = list(filter(wavelet_regex.search, output.columns.values))
    assert output.loc[output["CellID"] == 31, wavelet_cols].isna().all().all()
    assert ~output.loc[output["CellID"] == 30, wavelet_cols].isna().any().any()
    assert output.shape[1] == ts_df.shape[1]  # Correct # cols


def test_error_wavelet(frame_df, ts_df):
    # If we have < 4 timepoints, the wavelet calculation itself fails and all
    # the wavelet features will be NAN
    df = frame_df.loc[frame_df["FrameID"] < 4]
    output = time_series_features(df)

    # All wavelet features should be NA
    wavelet_regex = re.compile("_l1_l2_l3_")
    wavelet_cols = list(filter(wavelet_regex.search, output.columns.values))
    assert output[wavelet_cols].isna().all().all()
    assert output.shape[1] == ts_df.shape[1]  # Correct # cols

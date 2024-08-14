from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from cellphe.features.time_series import time_series_features
from cellphe.input import read_roi

pytestmark = pytest.mark.integration


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


def test_time_series_features():
    # R package uses 'xpos' and 'ypos' - Python package uses 'x' and 'y'
    frame_features = pd.read_csv("tests/resources/benchmark_features.csv")
    frame_features = frame_features.rename(columns={"xpos": "x", "ypos": "y"})
    expected = pd.read_csv("tests/resources/benchmark_time_series_features.csv")
    output = time_series_features(frame_features)
    output = output.rename(columns={"x": "xpos", "y": "ypos"})

    assert_frame_equal_extended_diff(expected.reset_index(drop=True), output.reset_index(drop=True))

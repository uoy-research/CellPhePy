from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cellphe.features.time_series import *


def test_interpolate():
    df = pd.DataFrame(
        {
            "CellID": [1, 1, 1, 1, 1, 4, 4, 4, 4],
            "FrameID": [1, 3, 4, 8, 10, 5, 8, 9, 11],
            "Val1": [2, 5, 8, 10, 12, 6, 7, 9, 15],
            "Val2": [3, 1, 2, 18, 8, 4, 2, 7, 1],
        }
    )
    output = interpolate(df)

    expected = pd.DataFrame(
        {
            "CellID": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4],
            "FrameID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 11],
            "Val1": [2, 3.5, 5, 8, 8.5, 9, 9.5, 10, 11, 12, 6, 19 / 3, 20 / 3, 7, 9, 12, 15],
            "Val2": [3, 2, 1, 2, 6, 10, 14, 18, 13, 8, 4, 10 / 3, 8 / 3, 2, 7, 4, 1],
        }
    )
    assert_frame_equal(output, expected)


@pytest.fixture
def signal():
    # Differences are 1, 2, -3, 6, -7, 9
    # Sum of ascent are 1+2+6+9 = 18
    # Sum of ascent are -10
    # N = 7
    # ascent is 18/7
    # descent is -10/7
    return pd.Series([2, 3, 5, 2, 8, 1, 10]).diff()


def test_ascent(signal):
    expected = 18 / 7
    output = ascent(signal)
    assert output == pytest.approx(expected)


def test_descent(signal):
    expected = -10 / 7
    output = descent(signal)
    assert output == pytest.approx(expected)


def test_haar_approximation_1d():
    input = np.arange(1, 21)
    output = haar_approximation_1d(input)
    assert output[0] == pytest.approx(np.repeat(-1 / np.sqrt(2), 10))
    assert output[1] == pytest.approx(np.repeat(-np.sqrt(2), 5))
    assert output[2] == pytest.approx(np.repeat(-2 * np.sqrt(2), 2))

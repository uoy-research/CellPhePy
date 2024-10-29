from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.separation import calculate_separation_scores, optimal_separation_features


@pytest.fixture()
def df1():
    yield pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture()
def df2():
    yield pd.DataFrame({"x": [11, 28, 31], "y": [42, 55, 68]})


def mocked_return(separation):
    return ["foo", "bar"]


def test_calculate_separation_scores_errors_nonautostring(df1, df2):
    # If pass a string that isn't 'auto', the function should raise a ValueError
    with pytest.raises(ValueError):
        calculate_separation_scores([df1, df2], "foo")


def test_calculate_separation_scores_calls_optimal_separation_threshold(df1, df2, mocker):
    # If pass in 'auto' to the threshold argument, the
    # optimal_separation_features function should be called
    mocked_func = mocker.patch("cellphe.separation.optimal_separation_features")
    mocked_func.side_effect = mocked_return
    calculate_separation_scores([df1, df2], "auto")
    mocked_func.assert_called_once()


def test_calculate_separation_scores_doesnt_call_optimal_separation_threshold(df1, df2, mocker):
    # When pass a threshold in directly, there's no need to call
    # optimal_separation_features
    mocked_func = mocker.patch("cellphe.separation.optimal_separation_features")
    mocked_func.side_effect = mocked_return
    calculate_separation_scores([df1, df2], 0.3)
    mocked_func.assert_not_called()

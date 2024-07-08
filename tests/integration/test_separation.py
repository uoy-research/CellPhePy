from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.separation import calculate_separation_scores, optimal_separation_threshold

pytestmark = pytest.mark.integration


@pytest.fixture()
def treated():
    yield pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def untreated():
    yield pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def separation():
    yield pd.read_csv("tests/resources/benchmark_separation.csv")


def test_calculate_separation_scores(treated, untreated, separation):
    output = calculate_separation_scores(treated, untreated)
    expected = pd.read_csv("tests/resources/benchmark_separation.csv")

    # Prepare for testing
    output = output.sort_values("Feature", ascending=False).reset_index(drop=True)
    expected = separation.sort_values("Feature", ascending=False).reset_index(drop=True)

    pd.testing.assert_frame_equal(output, expected)


def test_optimal_separation_threshold(treated, untreated, separation):
    np.random.seed(17)
    output = optimal_separation_threshold(untreated, treated, separation)
    # Expected value from manual testing of the R package
    expected = 0.1

    assert output == expected

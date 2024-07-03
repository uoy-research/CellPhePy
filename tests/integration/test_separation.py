from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.separation import calculate_separation_scores

pytestmark = pytest.mark.integration


def test_calculate_separation_scores():
    untreated = pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")
    treated = pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")

    output = calculate_separation_scores(treated, untreated)
    expected = pd.read_csv("tests/resources/benchmark_separation.csv")

    # Prepare for testing
    output = output.sort_values("Feature", ascending=False).reset_index(drop=True)
    expected = expected.sort_values("Feature", ascending=False).reset_index(drop=True)

    pd.testing.assert_frame_equal(output, expected)

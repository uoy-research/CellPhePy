from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.classification import classify_cells

pytestmark = pytest.mark.integration


def test_calculate_separation_scores():
    train_untreated = pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")
    train_treated = pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")
    test_untreated = pd.read_csv("data/UntreatedTest.csv").drop(columns="Unnamed: 0")
    test_treated = pd.read_csv("data/TreatedTest.csv").drop(columns="Unnamed: 0")
    training = pd.concat((train_untreated, train_treated))
    test = pd.concat((test_untreated, test_treated))
    labels = np.concatenate(
        (np.repeat("Untreated", train_untreated.shape[0]), np.repeat("Treated", train_treated.shape[0]))
    )

    expected = {"Treated": 0.1368421, "Untreated": 0.8631579}
    actual = classify_cells(training, test, labels)
    assert expected["Treated"] == 0.5
    assert expected["Untreated"] == 0.5

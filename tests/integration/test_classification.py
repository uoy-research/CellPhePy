from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.classification import classify_cells

pytestmark = pytest.mark.integration


def test_classify_cells():
    train_untreated = pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")
    train_treated = pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")
    test_untreated = pd.read_csv("data/UntreatedTest.csv").drop(columns="Unnamed: 0")
    test_treated = pd.read_csv("data/TreatedTest.csv").drop(columns="Unnamed: 0")
    training = pd.concat((train_untreated, train_treated))
    test = pd.concat((test_untreated, test_treated))
    labels = np.concatenate(
        (np.repeat("Untreated", train_untreated.shape[0]), np.repeat("Treated", train_treated.shape[0]))
    )

    # Allow 5% margin
    expected = {"Treated": 0.4041096, "Untreated": 0.5958904}
    actual = classify_cells(training, labels, test)
    for label, target in expected.items():
        assert target - 0.05 < np.mean(actual == label) < target + 0.05

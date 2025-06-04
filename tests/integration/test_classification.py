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
    train_labels = np.concatenate(
        (np.repeat("Untreated", train_untreated.shape[0]), np.repeat("Treated", train_treated.shape[0]))
    )
    test_labels = np.concatenate(
        (np.repeat("Untreated", test_untreated.shape[0]), np.repeat("Treated", test_treated.shape[0]))
    )

    # Allow 5% margin on accuracy
    actual = classify_cells(training, train_labels, test)
    accuracy = np.mean(actual == test_labels)
    expected_accuracy = 0.91
    assert (expected_accuracy - 0.05) < accuracy < (expected_accuracy + 0.05)

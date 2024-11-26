from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.classification import classify_cells


@pytest.fixture()
def train_x():
    yield pd.DataFrame({"a": np.random.normal(5, 2, 10), "b": np.random.normal(100, 10, 10)})


@pytest.fixture()
def test_x():
    yield pd.DataFrame({"a": np.random.normal(6, 2, 10), "b": np.random.normal(90, 20, 10)})


@pytest.fixture()
def train_y():
    yield np.concatenate((np.repeat("a", 5), np.repeat("b", 5)))


def test_get_labels_default(train_x, test_x, train_y):
    output = classify_cells(train_x, train_y, test_x)
    assert output.shape == (10,)
    assert (np.sort(np.unique(output)) == np.array(["a", "b"])).all()


def test_get_probs(train_x, test_x, train_y):
    labels, probs = classify_cells(train_x, train_y, test_x, return_probs=True)
    assert labels.shape == (10,)
    assert (np.sort(np.unique(labels)) == np.array(["a", "b"])).all()
    assert probs.shape == (10, 2)
    assert (np.sum(probs, axis=1) == 1).all()

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.segmentation import predict_segmentation_errors

pytestmark = pytest.mark.integration


@pytest.fixture()
def treated():
    yield pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def untreated():
    yield pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def seg_clean():
    yield pd.read_csv("data/CorrectSegs_MDAMB231.csv").drop(columns="CellIDs")


@pytest.fixture()
def seg_error():
    yield pd.read_csv("data/SegErrors_MDAMB231.csv").drop(columns="CellIDs")


def test_predict_segmentation_errors(treated, untreated, seg_clean, seg_error):
    np.random.seed(123)
    training = pd.concat((untreated, treated))
    obs = predict_segmentation_errors(seg_error, seg_clean, training, 5, 0.7)
    actual = np.full(training.shape[0], False)
    benchmark_errors = (
        np.array(
            [
                1,
                10,
                16,
                17,
                21,
                27,
                49,
                59,
                89,
                98,
                101,
                105,
                116,
                130,
                147,
                156,
                160,
                173,
                178,
                199,
                203,
                204,
                206,
                207,
                212,
                213,
                214,
                215,
                216,
                224,
                229,
                232,
                233,
                234,
                239,
                240,
                246,
                248,
                257,
                259,
            ]
        )
        - 1
    )  # original description is R 1-index, here use 0
    actual[benchmark_errors] = True

    # Wouldn't expect the exact same classifications, but let's aim for 90%
    # agreement. NB this test is stochastic as the tree fitting algorithm isn't
    # deterministic
    same_classification = obs == actual
    assert same_classification.mean() > 0.90


def test_predict_segmentation_errors_ensemble(treated, untreated, seg_clean, seg_error):
    np.random.seed(123)
    training = pd.concat((untreated, treated))

    obs = predict_segmentation_errors(seg_error, seg_clean, training, 5, 0.7, num_repeats=3)
    actual = np.full(training.shape[0], False)
    benchmark_errors = (
        np.array(
            [
                1,
                10,
                17,
                21,
                98,
                101,
                105,
                116,
                147,
                156,
                158,
                160,
                164,
                178,
                199,
                204,
                206,
                213,
                214,
                215,
                216,
                227,
                233,
                234,
                239,
                240,
                243,
                246,
                248,
                257,
                259,
            ]
        )
        - 1
    )
    actual[benchmark_errors] = True

    # Wouldn't expect the exact same classifications, but let's aim for 90%
    # agreement. NB this test is stochastic as the tree fitting algorithm isn't
    # deterministic
    same_classification = obs == actual
    assert same_classification.mean() > 0.90

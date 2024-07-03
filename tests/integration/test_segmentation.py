from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.segmentation import predict_segmentation_errors

pytestmark = pytest.mark.integration


def test_predict_segmentation_errors():
    clean = pd.read_csv("data/CorrectSegs_MDAMB231.csv")
    error = pd.read_csv("data/SegErrors_MDAMB231.csv")

    untreated = pd.read_csv("data/UntreatedTraining.csv")
    treated = pd.read_csv("data/TreatedTraining.csv")
    training = pd.concat((untreated, treated))

    # Remove cellID column as not a numeric feature
    clean.drop(columns="CellIDs", inplace=True)
    error.drop(columns="CellIDs", inplace=True)
    training.drop(columns="Unnamed: 0", inplace=True)

    obs = predict_segmentation_errors(error, clean, training, 5, 0.7)
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


def test_predict_segmentation_errors_ensemble():
    clean = pd.read_csv("data/CorrectSegs_MDAMB231.csv")
    error = pd.read_csv("data/SegErrors_MDAMB231.csv")

    untreated = pd.read_csv("data/UntreatedTraining.csv")
    treated = pd.read_csv("data/TreatedTraining.csv")
    training = pd.concat((untreated, treated))

    # Remove cellID column as not a numeric feature
    clean.drop(columns="CellIDs", inplace=True)
    error.drop(columns="CellIDs", inplace=True)
    training.drop(columns="Unnamed: 0", inplace=True)

    obs = predict_segmentation_errors(error, clean, training, 5, 0.7, num_repeats=3)
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

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

    obs = predict_segmentation_errors(error, clean, 20, training, 0.7)
    actual = np.full(training.shape[0], False)
    benchmark_errors = (
        np.array(
            [
                10,
                17,
                21,
                49,
                98,
                101,
                105,
                116,
                147,
                156,
                160,
                164,
                173,
                178,
                195,
                199,
                202,
                204,
                206,
                213,
                214,
                215,
                216,
                222,
                233,
                234,
                240,
                246,
                248,
                249,
                257,
                259,
            ]
        )
        - 1
    )  # original description is R 1-index, here use 0
    actual[benchmark_errors] = True

    assert (obs == actual).all()

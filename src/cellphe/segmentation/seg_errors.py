"""
    cellphe.segmentation.seg_errors
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Functions relating to handling segmentation errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import tree


def remove_predicted_seg_errors(dataset: pd.DataFrame, cellid_label: str, error_cells: list[int]) -> pd.DataFrame:
    """
    Remove predicted segmentation errors from a data set.

    This function can be used to automate removal of predicted segmentation
    errors from the test set.

    :param dataset: Test set for segmentation error predictions to be made
    :param cellid_label: Label for the column of cell identifiers within the
    test set.
    :param error_cells: Output from either predictSegErrors() or predictSegErrors_Ensemble(),
    a list of cell identifiers for cells classified as segmentation error
    :return: A dataframe with the predicted errors removed.
    """
    return dataset.loc[~dataset[cellid_label].isin(error_cells)]


def predict_segmentation_errors(
    errors: pd.DataFrame, clean: pd.DataFrame, num: int, testset: pd.DataFrame, proportion: float
) -> np.array:
    """
    TODO

    mention datasets must contain only numeric
    """
    feats = pd.concat((errors, clean))
    labels = np.concatenate((np.repeat(1, errors.shape[0]), np.repeat(-1, clean.shape[0])))

    preds = np.array([tree.DecisionTreeClassifier().fit(feats, labels).predict(testset) for i in range(num)])
    pct_voted = preds.mean(axis=0)
    return pct_voted > proportion

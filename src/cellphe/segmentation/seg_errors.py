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
    errors: pd.DataFrame, clean: pd.DataFrame, testset: pd.DataFrame, num: int = 5, proportion: float = 0.7
) -> np.array:
    """
    Predicts whether or not cells have experienced segmentation errors through
    the use of decision trees fitted on labelled training data.

    num decision trees are trained on the labelled training data (errors are
    features from incorrectly segmented cells and clean are features from known
    correctly segemented cells). They then predict whether the cells provided in
    the testset are segmented correctly or not. If proportion of the num trees
    vote for a segmentation error, then that cell is predicted to contain an
    error.

    :param errors: DataFrame containing the 1111 frame-level features from a set
    of cells known to be incorrectly segmented (having removed the CellID column).
    :param clean: DataFrame containing the 1111 frame-level features from a set
    of cells known to be correctly segmented (having removed the CellID column).
    :param testset: DataFrame containing the 1111 frame-level features from the
    cells to be assesed (having removed the CellID column).
    :param num: Numbe of decision trees to fit.
    :param proportion: Proportion of decision trees needed for a segmentation
    error vote to be successful.
    :return: Returns a Numpy boolean array the same length as there are rows in
    testset, indicating whether the associated Cell contains a segmentation
    error or not.
    """
    feats = pd.concat((errors, clean))
    labels = np.concatenate((np.repeat(1, errors.shape[0]), np.repeat(0, clean.shape[0])))

    preds = np.array(
        [
            tree.DecisionTreeClassifier(criterion="log_loss", min_samples_leaf=10, min_samples_split=5)
            .fit(feats, labels)
            .predict(testset)
            for i in range(num)
        ]
    )
    pct_voted = preds.mean(axis=0)
    return pct_voted > proportion

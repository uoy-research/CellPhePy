"""
    cellphe.classification
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions for classifying cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def classify_cells(
    train_x: pd.DataFrame, train_y: np.array, test_x: pd.DataFrame, return_probs: bool = False
) -> np.array:
    """Predicts cell class.

    Trains an xgboost classifier to predict cell labels.

    :param train_x: DataFrame containing cell features (but no labels) for the
    :param train_y: Array of labels (can be strings or integers) for the
        training set.
    :param test_x: DataFrame containing cell features (but no labels) for the
        test set.
    :param return_probs: Whether to return the probabilities as well as the
        labels. In this case the function returns a tuple of 2 items.
    :return: If return_probs is False, an N length 1D array where N is the
        number of rows in test_x. Each entry is the associated predicted label.
        If return_probs is True, then a tuple with the first item being the
        labels array, and the second being a matrix of N x M, where M is the
        number of unique classes in train_y.
    """
    # Transform labels
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)

    # Fit xgboost
    mod_xgb = xgb.XGBClassifier(tree_method="hist")
    mod_xgb.fit(train_x, train_y)

    # Make predictions and convert back to the original label range
    preds_xgb_raw = mod_xgb.predict(test_x)
    preds_xgb = le.inverse_transform(preds_xgb_raw)

    if return_probs:
        pred_probs = mod_xgb.predict_proba(test_x)
        return preds_xgb, pred_probs
    else:
        return preds_xgb

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


def classify_cells(train_x: pd.DataFrame, train_y: np.array, test_x: pd.DataFrame) -> np.array:
    """Predicts cell class using an ensemble.

    Trains three classifiers (Linear Discriminant Analysis, Random Forest and
    Support Vector Machine) and uses these in an ensemble by majority vote to
    obtain final predictions for cell type classification of a test set.

    :param train_x: DataFrame containing cell features (but no labels) for the
    :param train_y: Array of labels (can be strings or integers) for the
        training set.
    :param test_x: DataFrame containing cell features (but no labels) for the
        test set.
    :return: A 2D array with as many rows as there are rows in test_x, with 4
        columns containing the predictions for each of the 3 classifiers and the
        final ensemble prediction.
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

    return preds_xgb

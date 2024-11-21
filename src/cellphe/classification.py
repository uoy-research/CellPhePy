"""
    cellphe.classification
    ~~~~~~~~~~~~~~~~~~~~~~

    Functions for classifying cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler


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

    # Scale variables
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # Define the models to be used in the ensemble
    mod_lda = LinearDiscriminantAnalysis()
    mod_rf = RandomForestClassifier(n_estimators=200, max_features=5)
    mod_svm = svm.SVC(kernel="rbf")

    # Create the ensemble
    models = [("lda", mod_lda), ("rf", mod_rf), ("svm", mod_svm)]
    model_names = [x[0] for x in models]
    ensemble = VotingClassifier(models, voting="hard").fit(train_x, train_y)

    # Generate the ensemble predictions. This doesn't return the individual
    # classifier predictions which we also want, so we can retrieve them
    # manually. But then it returns the predictions unlabelled so then need to
    # reapply label
    # Pylint doesn't recognise the predict method and le_ attributes
    # TODO: Could probably individual_preds = use ensemble.transform(test_x)
    # And get model_names from list(ensemble.named_estimators.keys())
    # pylint: disable=no-member
    ensemble_preds = ensemble.predict(test_x)
    individual_preds = np.array([ensemble.named_estimators_[x].predict(test_x) for x in model_names])
    labeller = ensemble.le_
    individual_preds = np.array([labeller.inverse_transform(x) for x in individual_preds]).T
    all_preds = np.column_stack((individual_preds, ensemble_preds))
    return all_preds

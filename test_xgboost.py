from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from cellphe import classify_cells

# Load data
train_untreated = pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")
train_treated = pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")
test_untreated = pd.read_csv("data/UntreatedTest.csv").drop(columns="Unnamed: 0")
test_treated = pd.read_csv("data/TreatedTest.csv").drop(columns="Unnamed: 0")
training = pd.concat((train_untreated, train_treated))
test = pd.concat((test_untreated, test_treated))

training["label"] = np.concatenate(
    (np.repeat("Untreated", train_untreated.shape[0]), np.repeat("Treated", train_treated.shape[0]))
)
test["label"] = np.concatenate(
    (np.repeat("Untreated", test_untreated.shape[0]), np.repeat("Treated", test_treated.shape[0]))
)

# Shuffle data
training = training.sample(frac=1)
test = test.sample(frac=1)

# Predict using ensemble
preds_ensemble = classify_cells(training.drop(columns=["label"]), training["label"], test.drop(columns=["label"]))

# Fit xgboost with default parameters
mod_xgb = xgb.XGBClassifier(tree_method="hist")
# Fit the model, test sets are used for early stopping.
le = LabelEncoder()
y_train_encoded = le.fit_transform(training["label"])
mod_xgb.fit(training.drop(columns=["label"]), y_train_encoded)
preds_xgb_raw = mod_xgb.predict(test.drop(columns=["label"]))
preds_xgb = le.inverse_transform(preds_xgb_raw)

# Calculate accuracies
preds = {
    "lda": preds_ensemble[:, 0],
    "rf": preds_ensemble[:, 1],
    "svm": preds_ensemble[:, 2],
    "ensemble": preds_ensemble[:, 3],
    "xgb": preds_xgb,
}
accuracies = pd.DataFrame.from_records(
    [{"model": k, "accuracy": np.mean(test["label"] == v)} for k, v in preds.items()]
)
accuracies.sort_values(by="accuracy", ascending=False, inplace=True)
print(accuracies)

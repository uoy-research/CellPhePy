from __future__ import annotations

import pandas as pd


def remove_predicted_seg_errors(dataset: pd.DataFrame, cellid_label: str, error_cells: list[int]) -> pd.DataFrame:
    """
    Remove predicted segmentation errors from a data set.

    This function can be used to automate removal of predicted segmentation
    errors from the test set.

    :param dataset: Test set for segmentation error predictions to be made
    :param cellid_label: Label for the column of cell identifiers within the
    test set.
    :param error_cells: Output from either predictSegErrors() or predictSegErrors_Ensemble(), a list of cell identifiers for cells classified as segmentation error
    :return: A dataframe with the predicted errors removed.
    """
    return dataset.loc[~dataset[cellid_label].isin(error_cells)]

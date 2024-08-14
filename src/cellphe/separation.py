"""
    cellphe.separation
    ~~~~~~~~~~~~~~~~~~

    Functions related to calculating separation scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_separation_scores(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float | str = 0) -> pd.DataFrame:
    """Calculates the separation score between 2 datasets across a number of
    feature.
    A threshold can be supplied to identify discriminatory variables.

    Note that df1 and df2 should include only columns of features (i.e remove
    cell identifier columns prior to function use), and these columns must be
    the same in number and data type.

    :param df1: DataFrame for the first class containing only feature columns.
    :param df2: DataFrame for the second class containing only feature columns.
    :param threshold: Separation threshold either as as a number or the string
        'auto'. If a number then features with a separation score below this value
        are discarded. If 'auto' then the threshold is automatically identified.
    :return: A DataFrame comprising 2 columns: Feature and Separation, where
        each row corresponds to a different feature's separation score. Any
        features with separation scores less than threshold are removed.
    """
    if threshold != "auto" and not (isinstance(threshold, (int, float, complex)) and not isinstance(threshold, bool)):
        raise ValueError("threshold must be 'auto' or numeric")

    df1["group"] = "g1"
    df2["group"] = "g2"
    df_both = pd.concat((df1, df2))
    # Melt into long
    df_long = pd.melt(df_both, id_vars="group", var_name="Feature", value_name="value")
    # Now get wide with 1 column per group
    df_agg = df_long.groupby(["group", "Feature"]).agg(["count", "mean", "var"]).reset_index()

    # Remove multi-index columns
    df_agg.columns = [x[0] if i < 2 else x[1] for i, x in enumerate(df_agg.columns.values)]

    # Turn wide, so each row represents a feature and columns are group/features
    df = df_agg.pivot(columns="group", index="Feature").reset_index()

    # Calculations: haven't modified from the R code
    df["sum"] = df["count"]["g1"] + df["count"]["g2"]
    df["Vw"] = ((df["count"]["g1"] - 1) * df["var"]["g1"] + (df["count"]["g2"] - 1) * df["var"]["g2"]) / (df["sum"] - 2)
    overmean = (df["count"]["g1"] * df["mean"]["g1"] + df["count"]["g2"] * df["mean"]["g2"]) / df["sum"]
    df["Vb"] = (
        df["count"]["g1"] * (df["mean"]["g1"] - overmean) ** 2 + df["count"]["g2"] * (df["mean"]["g2"] - overmean) ** 2
    ) / (df["sum"] - 2)
    df["Separation"] = df["Vb"] / df["Vw"]

    # Clean up
    sep_df = df[["Feature", "Separation"]]
    sep_df.columns = sep_df.columns.droplevel(1)

    # Threshold check
    if threshold == "auto":
        subset = optimal_separation_features(sep_df)
        sep_df = sep_df.loc[sep_df["Feature"].isin(subset),]
    else:
        sep_df = sep_df.loc[(sep_df["Separation"] >= threshold) & (pd.notna(sep_df["Separation"])),]

    return sep_df


def optimal_separation_features(separation: pd.DataFrame) -> pd.Series:
    """Determines the optimal feature subset by an elbow method on their separation
    scores.

    :param separation: DataFrame containing separation scores for a number of
        features. Has 2 columns: Feature and Separation.

    :return: A Series of the feature names to keep.
    """
    scores = separation["Separation"].values
    ordered = scores[np.argsort(-scores)]
    mins = np.min(ordered)
    maxs = np.max(ordered)
    indices = np.arange(ordered.size)
    dists = ((mins - maxs) * indices) / (ordered.size - 1) + maxs - ordered
    ind = np.argmax(dists)
    thresh = ordered[ind]
    res = separation.loc[scores > thresh]["Feature"]
    return res

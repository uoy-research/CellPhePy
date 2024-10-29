"""
    cellphe.separation
    ~~~~~~~~~~~~~~~~~~

    Functions related to calculating separation scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_separation_scores(dfs: list[pd.DataFrame], threshold: float | str = 0) -> pd.DataFrame:
    """
    Calculates the separation score between multiple datasets (n groups) across a number of features.
    A threshold can be supplied to identify discriminatory variables.

    Note that each DataFrame should include only columns of features (i.e., remove
    cell identifier columns prior to function use), and these columns must be
    the same in number and data type across all DataFrames.

    :param dfs: List of DataFrames, each representing a different group, containing only feature columns.
    :param threshold: Separation threshold either as a number or the string 'auto'.
        If a number, then features with a separation score below this value are discarded.
        If 'auto', then the threshold is automatically identified.
    :return: A DataFrame comprising 2 columns: Feature and Separation, where
        each row corresponds to a different feature's separation score. Any
        features with separation scores less than threshold are removed.
    """
    # Validate input
    if len(dfs) < 2:
        raise ValueError("At least 2 DataFrames are required.")
    if threshold != "auto" and not (isinstance(threshold, (int, float))):
        raise ValueError("threshold must be 'auto' or numeric")

    # Assign group labels to each DataFrame and concatenate them
    for i, df in enumerate(dfs):
        df["group"] = f"g{i + 1}"
    # Combine all DataFrames into one
    df_combined = pd.concat(dfs)
    # Melt into long format
    df_long = pd.melt(df_combined, id_vars="group", var_name="Feature", value_name="value")
    # Aggregate data
    df_agg = df_long.groupby(["group", "Feature"]).agg(["count", "mean", "var"]).reset_index()

    # Remove multi-index columns
    df_agg.columns = [x[0] if i < 2 else x[1] for i, x in enumerate(df_agg.columns.values)]

    # Pivot the DataFrame to have each group as columns for count, mean, and var
    df = df_agg.pivot(columns="group", index="Feature").reset_index()

    # Calculate the grand total (sum of counts across all groups)
    df["sum"] = df["count"].sum(axis=1)
    # Calculate the grand mean (weighted mean of means from all groups)
    overmean = sum(df["count"][f"g{i + 1}"] * df["mean"][f"g{i + 1}"] for i in range(len(dfs))) / df["sum"]
    # Calculate within-group variance (Vw)
    df["Vw"] = sum((df["count"][f"g{i + 1}"] - 1) * df["var"][f"g{i + 1}"] for i in range(len(dfs))) / (
        df["sum"] - len(dfs)
    )

    # Calculate between-group variance (Vb)
    df["Vb"] = sum(df["count"][f"g{i + 1}"] * (df["mean"][f"g{i + 1}"] - overmean) ** 2 for i in range(len(dfs))) / (
        df["sum"] - len(dfs)
    )

    # Calculate the Separation score
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

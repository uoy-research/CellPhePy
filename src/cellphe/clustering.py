"""
    cellphe.clustering
    ~~~~~~~~~~~~~~~~~~

    Functions related to clustering cells.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, ward
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale


def identify_clusters(df: pd.DataFrame, k: int, plot=False) -> np.array:
    """Performs hierarchical clustering on a given data set in order
    to identify k heterogeneous cell clusters.

    :param df: DataFrame containing only feature variables (i.e. no cell IDs or
        other labels).
    :param k: Number of clusters to identify.
    :param plot: Whether to plot the resulting dendrogram. Can help in
        identifying a suitable k.
    :return: A numpy array containing as many entries as there are rows in df.
        Each entry contains the corresponding cell label.
    """
    data_proc = scale(df.values, axis=0)
    linkage = ward(pdist(data_proc))
    labs = fcluster(linkage, t=k, criterion="maxclust")

    if plot:
        thresh = linkage[-(k - 1), 2]
        dendrogram(linkage, color_threshold=thresh)
        plt.show()
    return labs

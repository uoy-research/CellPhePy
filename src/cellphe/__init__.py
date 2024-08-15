"""
    cellphe
    ~~~~~~~

    Top level package.
"""

from __future__ import annotations

from cellphe.classification import classify_cells
from cellphe.clustering import identify_clusters
from cellphe.features import extract_features, time_series_features
from cellphe.input import copy_features
from cellphe.separation import calculate_separation_scores

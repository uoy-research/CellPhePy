"""
    cellphe
    ~~~~~~~

    Top level package.
"""

from __future__ import annotations

from cellphe.classification import classify_cells
from cellphe.clustering import identify_clusters
from cellphe.features import cell_features, time_series_features
from cellphe.input import import_data
from cellphe.separation import calculate_separation_scores

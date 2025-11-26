"""
cellphe
~~~~~~~

Top level package.
"""

from __future__ import annotations

from cellphe.classification import classify_cells as classify_cells
from cellphe.clustering import identify_clusters as identify_clusters
from cellphe.features import cell_features as cell_features
from cellphe.features import time_series_features as time_series_features
from cellphe.input import import_data as import_data
from cellphe.separation import calculate_separation_scores as calculate_separation_scores


def segment_images():
    from cellphe.segmentation import segment_images as dummy

    return dummy


def track_images():
    from cellphe.tracking import track_images as dummy

    return dummy

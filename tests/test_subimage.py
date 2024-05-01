from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from cellphe.processing import create_type_mask


def test_create_type_mask():
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [1, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 6],
            [7, 6],
            [8, 5],
            [7, 4],
            [7, 3],
            [8, 2],
            [7, 1],
            [6, 1],
            [5, 1],
            [4, 2],
            [3, 1],
            [2, 2],
        ]
    )
    output = create_type_mask(image, roi)
    expected = np.zeros((6, 8), dtype="int64")
    expected[[0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5], [0, 1, 3, 7, 0, 7, 7, 0, 0, 1, 7]] = -1
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_double_layer():
    # Some ROIs have 2 layers of thickness
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [1, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 6],
            [7, 6],
            [8, 5],
            [7, 4],
            [7, 3],
            [8, 2],
            [7, 1],
            [6, 1],
            [5, 1],
            [4, 2],
            [3, 1],
            [2, 2],
            [2, 1],
        ]
    )
    output = create_type_mask(image, roi)
    expected = np.zeros((6, 8), dtype="int64")
    expected[[0, 0, 0, 1, 2, 3, 4, 5, 5, 5], [0, 3, 7, 0, 7, 7, 0, 0, 1, 7]] = -1
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_diagonal_connections():
    # ROIs aren't always defined by adjacent cells, can have diagonal paths
    # The original code had a for loop to help identify cells within the ROI
    # under this specific condition. Ensure this version handles it
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 7],
            [3, 8],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 6],
            [7, 6],
            [8, 5],
            [7, 4],
            [7, 3],
            [8, 2],
            [7, 1],
            [6, 1],
            [5, 1],
            [4, 2],
            [3, 1],
            [2, 2],
            [2, 1],
        ]
    )
    output = create_type_mask(image, roi)
    expected = np.full((8, 8), -1, dtype="int64")
    # ROI are 0 and are given in x/y rather than row/col
    # Subtract off the 1 for reducing the boundary box
    expected[roi[:, 1] - 1, roi[:, 0] - 1] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1],
    ] = 1
    assert (output == expected).all()

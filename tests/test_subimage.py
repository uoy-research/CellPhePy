from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from cellphe.processing import create_type_mask, extract_subimage
from cellphe.processing.image import SubImage


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
    expected = np.full((10, 10), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
        [3, 5, 6, 7, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7],
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
    expected = np.full((10, 10), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
        [3, 5, 6, 7, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7],
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
    expected = np.full((10, 10), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
        [3, 5, 6, 7, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 2],
    ] = 1
    assert (output == expected).all()


def test_subimageinfo():
    roi = np.array([[2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [3, 2], [2, 2], [1, 1]])
    image = np.arange(1, 16).reshape(3, 5, order="F") ** 2
    expected = SubImage(
        sub_image=np.array([[16, 49, 100, 169], [25, 64, 121, 196], [36, 81, 144, 225]]),
        type_mask=np.array([[-1, 0, 0, 0], [0, 1, 1, 0], [-1, 0, 0, 0]]),
        centroid=np.array([2.875, 1]),
    )
    output = extract_subimage(image, roi)

    assert (output.sub_image == expected.sub_image).all()
    assert (output.type_mask == expected.type_mask).all()
    assert output.centroid == pytest.approx(expected.centroid)

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from cellphe.processing import *
from cellphe.processing.image import SubImage


def test_create_type_mask_matplotlib():
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
        ]
    )
    output = create_type_mask_matplotlib(roi)
    expected = np.full((6, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_double_layer_matplotlib():
    # Some ROIs have 2 layers of thickness
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
            [1, 0],
        ]
    )
    output = create_type_mask_matplotlib(roi)
    expected = np.full((6, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_diagonal_connections_matplotlib():
    # ROIs aren't always defined by adjacent cells, can have diagonal paths
    # The original code had a for loop to help identify cells within the ROI
    # under this specific condition. Ensure this version handles it
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 6],
            [2, 7],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
            [1, 0],
        ]
    )
    output = create_type_mask_matplotlib(roi)
    expected = np.full((8, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_flood_fill():
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
        ]
    )
    output = create_type_mask_flood_fill(roi)
    expected = np.full((6, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_double_layer_flood_fill():
    # Some ROIs have 2 layers of thickness
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
            [1, 0],
        ]
    )
    output = create_type_mask_flood_fill(roi)
    expected = np.full((6, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    ] = 1
    assert (output == expected).all()


def test_create_type_mask_diagonal_connections_flood_fill():
    # ROIs aren't always defined by adjacent cells, can have diagonal paths
    # The original code had a for loop to help identify cells within the ROI
    # under this specific condition. Ensure this version handles it
    roi = np.array(
        [
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 6],
            [2, 7],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 4],
            [6, 3],
            [6, 2],
            [7, 1],
            [6, 0],
            [5, 0],
            [4, 0],
            [3, 1],
            [2, 0],
            [1, 1],
            [1, 0],
        ]
    )
    output = create_type_mask_flood_fill(roi)
    expected = np.full((8, 8), -1, dtype="int64")
    expected[roi[:, 1], roi[:, 0]] = 0
    expected[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5],
        [2, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1],
    ] = 1
    assert (output == expected).all()


def test_subimageinfo():
    roi = np.array([[2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [3, 3], [2, 3], [1, 1], [1, 2]])
    image = np.arange(1, 16).reshape(3, 5, order="F") ** 2
    expected = SubImage(
        sub_image=np.array([[16, 49, 100, 169], [25, 64, 121, 196], [36, 81, 144, 225]]),
        type_mask=np.array([[-1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [-1, 0, 0, 0]]),
        centroid=np.array([2.8, 1.5]),
    )
    output = extract_subimage(image, roi)

    assert (output.sub_image == expected.sub_image).all()
    assert (output.type_mask == expected.type_mask).all()
    assert output.centroid == pytest.approx(expected.centroid)

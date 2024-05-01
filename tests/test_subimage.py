from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from cellphe.processing import create_type_mask


def test_create_type_mask():
    image = np.arange(100).reshape(10, 10)
    roi = np.array(
        [
            [3, 1],
            [4, 1],
            [5, 2],
            [6, 3],
            [6, 4],
            [6, 5],
            [6, 6],
            [6, 7],
            [5, 8],
            [4, 7],
            [3, 7],
            [2, 8],
            [1, 7],
            [1, 6],
            [1, 5],
            [2, 4],
            [1, 3],
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


# TODO create test for the behaviour where can have outside cells next to
# positive cells

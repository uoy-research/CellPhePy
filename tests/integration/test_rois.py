from __future__ import annotations

import numpy as np
import pytest

from cellphe.features.frame import polygon
from cellphe.input import read_roi
from cellphe.processing.roi import roi_corners


@pytest.fixture()
def roi():
    roi = read_roi("tests/resources/roi.roi")
    roi = roi - roi.min(axis=0)
    yield roi


def test_roi_corners_real_roi(roi):
    output = roi_corners(roi)
    expected = np.array(
        [
            [0, 7],
            [1, 6],
            [1, 5],
            [4, 2],
            [5, 2],
            [6, 1],
            [7, 1],
            [8, 0],
            [14, 0],
            [15, 1],
            [16, 1],
            [17, 2],
            [18, 2],
            [21, 5],
            [21, 7],
            [22, 8],
            [22, 13],
            [21, 14],
            [21, 15],
            [20, 16],
            [20, 17],
            [19, 18],
            [18, 18],
            [17, 19],
            [3, 19],
            [2, 18],
            [2, 17],
            [1, 16],
            [1, 14],
            [0, 13],
            [1, 19],  # This and the following vertex should have been removed!
            [1, 18],
        ]
    )
    expected = np.flip(expected, axis=1)

    assert (np.sort(expected, axis=0) == np.sort(output, axis=0)).all()


def test_polygon_real_roi(roi):
    output = polygon(roi)
    expected = np.array([[0, 8], [7, 0], [19, 1], [18, 19], [5, 21]])
    assert (output == expected).all()

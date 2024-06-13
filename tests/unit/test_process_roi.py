from __future__ import annotations

import numpy as np

from cellphe.processing.roi import boundary_vertices, roi_corners


def test_boundary_vertices():
    # The point [3, 7] doesn't contribute to the inner cell area and should be
    # removed
    input = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 7], [3, 5], [2, 4], [3, 3]])
    expected = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 5], [2, 4], [3, 3]])
    # Normalise ROIs
    expected = expected - input.min(axis=0)
    input = input - input.min(axis=0)
    output = boundary_vertices(input)
    assert (expected == output).all()


def test_boundary_vertices_multiple_redundant_vertices():
    # There is a short path that doesn't contribute to the inner cell area that
    # needs to be iteratively removed [3,7], [3, 8] as well as a second
    # offshoot at [7, 4]
    input = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 7], [3, 5], [2, 4], [3, 3], [3, 8], [7, 4]])
    expected = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 5], [2, 4], [3, 3]])
    # Normalise ROIs
    expected = expected - input.min(axis=0)
    input = input - input.min(axis=0)
    output = boundary_vertices(input)
    assert (expected == output).all()


def test_roi_corners_diamond():
    # This is a simple diamond with 4 corners
    input = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 5], [2, 4], [3, 3]])
    expected = np.array([[4, 2], [6, 4], [4, 6], [2, 4]])
    # Normalise ROIs
    expected = expected - input.min(axis=0)
    input = input - input.min(axis=0)
    output = roi_corners(input)
    assert (np.sort(expected, axis=0) == np.sort(output, axis=0)).all()


def test_roi_corners_diamond():
    # This is a simple diamond with 4 corners
    input = np.array([[4, 2], [5, 3], [6, 4], [5, 5], [4, 6], [3, 5], [2, 4], [3, 3]])
    expected = np.array([[4, 2], [6, 4], [4, 6], [2, 4]])
    # Normalise ROIs
    expected = expected - input.min(axis=0)
    input = input - input.min(axis=0)
    output = roi_corners(input)
    assert (np.sort(expected, axis=0) == np.sort(output, axis=0)).all()

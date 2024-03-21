from __future__ import annotations

import numpy as np
import pytest

from cellphe.features.frame import *


def test_var_from_centre():
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    output = var_from_centre(boundaries)
    assert output[0] == pytest.approx(3.535534)
    assert output[1] == pytest.approx(4.444444)


def test_curvature():
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    output = curvature(boundaries, 4)
    assert output == pytest.approx(9.050967)


def test_min_box():
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    output = minimum_box(boundaries)
    assert output == pytest.approx(np.array([1.272792e01, 3.552714e-15]))


def test_polygon_line():
    # For test bc
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    expected = np.array([[1, 11], [10, 20]])
    output = polygon(boundaries)
    assert (output == expected).all()


def test_polygon_real_example():
    # For test bc
    ys = [
        10,
        9,
        8,
        7,
        6,
        6,
        5,
        4,
        3,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        24,
        24,
        24,
        24,
        25,
        25,
        25,
        25,
        24,
        24,
        23,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
    ]
    xs = [
        1,
        2,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        22,
        23,
        23,
        23,
        23,
        23,
        23,
        23,
        23,
        22,
        21,
        20,
        19,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        3,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    boundaries = [[x, y] for x, y in zip(xs, ys)]
    expected = [[1, 10], [6, 3], [16, 1], [23, 9], [23, 16], [16, 24], [8, 25], [3, 22], [1, 11]]
    output = polygon(boundaries)
    assert (output == expected).all()


def test_poly_class():
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    expected = np.array([12.727922, 6.283185, 0.000000, 0.000000])
    output = polygon_features(boundaries)
    assert expected == pytest.approx(output)


def test_poly_angle():
    input = np.sqrt(np.arange(1, 16).reshape(3, 5).transpose())
    expected = np.array([2.526113, 1.983286, 1.776365, 1.654226, 1.570796])
    output = polygon_angle(input)
    assert expected == pytest.approx(output)

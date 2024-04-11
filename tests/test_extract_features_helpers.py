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


def test_cooccurence_matrix():
    image1 = np.arange(1, 21).reshape(4, 5).transpose()
    image2 = np.concatenate((np.arange(1, 11), np.arange(21, 31))).reshape(4, 5).transpose()
    mask = np.ones((5, 4)).astype("int") == 1
    output = cooccurrence_matrix(image1, image2, mask, levels=10)
    expected = np.zeros((10, 10))
    expected[(0, 3, 3, 4, 4, 7, 7, 8, 9), (0, 1, 2, 5, 6, 7, 8, 8, 9)] = 1
    expected[(1, 2, 5, 6), (0, 1, 6, 7)] = 2
    assert (output == expected).all()


def test_haralick():
    input = np.zeros((10, 10))
    input[(0, 3, 3, 4, 4, 7, 7, 8, 9), (0, 1, 2, 5, 6, 7, 8, 8, 9)] = 1
    input[(1, 2, 5, 6), (0, 1, 6, 7)] = 2
    expected = np.array(
        [
            0.0865051903114187,
            1.11764705882353,
            0.582352941176471,
            1.08878774694817,
            0.956323969489567,
            6.56141868512111,
            10.6470588235294,
            1.08878774694817,
            0.379530127650022,
            0.591931127823415,
            123.824213793479,
            0.881906412984029,
            -13.5827396702625,
            1700.89828905305,
        ]
    )
    output = haralick(input)
    assert expected == pytest.approx(output)


def test_intensity_quantiles():
    expected = np.array(
        [0.3963749, 0.4367558, 0.3332230, 0.2120448, 0.2204704, 0.2337691, 0.3209318, 0.4305304, 0.4305304]
    )
    input = np.array(
        [
            [1, 1, 210],
            [1, 2, 220],
            [1, 3, 180],
            [2, 1, 190],
            [2, 2, 150],
            [2, 3, 140],
            [3, 1, 80],
            [3, 2, 100],
            [3, 3, 210],
            [4, 3, 130],
            [4, 4, 120],
        ]
    )
    output = intensity_quantiles(input)
    assert output == pytest.approx(expected)


def test_haar_approximation():
    input = np.arange(1, 25).reshape(6, 4, order="F")
    expected = np.array([[4.5, 16.5], [6.5, 18.5], [8.5, 20.5]])
    output = haar_approximation(input)
    assert output == pytest.approx(expected)


def test_double_image():
    input = np.arange(1, 5).reshape(2, 2, order="F")
    expected = np.array([[1, 1, 3, 3], [1, 1, 3, 3], [2, 2, 4, 4], [2, 2, 4, 4]])
    output = double_image(input)
    assert output == pytest.approx(expected)

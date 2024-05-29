from __future__ import annotations

import numpy as np
from PIL import Image

from cellphe.features.frame import *
from cellphe.input import read_roi


def test_cooccurrence_matrix_real_image():
    # Read in raw image
    image = Image.open("tests/resources/frame.tif")
    image = np.array(image)
    roi = read_roi("tests/resources/roi.roi")

    # Extract subimage info
    subimage = extract_subimage(image, roi)

    # Calculate cooccurrence matrix of subimage and its wavelet approx
    image_approx = double_image(haar_approximation(subimage.sub_image))
    image_approx = image_approx[: subimage.sub_image.shape[0], : subimage.sub_image.shape[1]]
    mask = subimage.type_mask >= 0
    output = cooccurrence_matrix(subimage.sub_image, image_approx, mask, 10)

    # Calculated from original code
    expected = np.zeros(100).reshape(10, 10)
    expected[0, 0] = 12
    expected[0, 1] = 8
    expected[0, 2] = 1
    expected[1, 0] = 5
    expected[1, 1] = 6
    expected[1, 2] = 8
    expected[1, 3] = 2
    expected[2, 0] = 4
    expected[2, 1] = 6
    expected[2, 2] = 7
    expected[2, 3] = 6
    expected[2, 4] = 1
    expected[3, 1] = 1
    expected[3, 2] = 7
    expected[3, 3] = 4
    expected[3, 4] = 10
    expected[4, 1] = 1
    expected[4, 2] = 5
    expected[4, 3] = 7
    expected[4, 4] = 16
    expected[4, 5] = 12
    expected[5, 3] = 1
    expected[5, 4] = 9
    expected[5, 5] = 26
    expected[5, 6] = 11
    expected[6, 5] = 10
    expected[6, 6] = 27
    expected[6, 7] = 9
    expected[7, 6] = 6
    expected[7, 7] = 33
    expected[7, 8] = 3
    expected[8, 7] = 2
    expected[8, 8] = 28
    expected[8, 9] = 3
    expected[9, 8] = 1
    expected[9, 9] = 1

    assert (output == expected).all()

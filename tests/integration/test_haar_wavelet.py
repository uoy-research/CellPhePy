from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cellphe.features.frame import haar_approximation

pytestmark = pytest.mark.integration


def test_haar_real_image():
    # Test Haar wavelet approximation using Pywavelets as used in this
    # package against the hand-rolled version in the original CellPhe
    image = Image.open("tests/resources/frame.tif")
    image = np.array(image)
    output = haar_approximation(image)
    expected = np.genfromtxt("tests/resources/haar_output.csv", delimiter=",")
    assert output == pytest.approx(expected)

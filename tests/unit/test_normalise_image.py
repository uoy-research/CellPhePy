from __future__ import annotations

import numpy as np
import pytest

from cellphe.processing import normalise_image


def test_normalise_image_max():
    image = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.max(normalise_image(image, 7, 13)) == pytest.approx(13)


def test_normalise_image_min():
    image = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.min(normalise_image(image, 7, 13)) == pytest.approx(7)

from __future__ import annotations

import numpy as np


def normalise_image(image: np.array, lower: int, upper: int) -> np.array:
    """
    Normalises an image to a specified range.

    :param image: The image to normalise as a 2D numpy array.
    :param lower: The lower bound of the target normalisation range, as an integer.
    :param upper: The upper boundo of the target normalisation range, as an integer.

    :return: The normalised image as a 2D numpy array.
    """
    scale = (upper - lower) / (np.max(image) - np.min(image))
    offset = upper - (scale * np.max(image))

    return scale * image + offset

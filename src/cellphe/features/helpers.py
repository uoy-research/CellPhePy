"""
    cellphe.features.helpers
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Helper functions for use in both frame and time-series level feature
    calculations.
"""

from __future__ import annotations

import numpy as np


def skewness(x: np.array) -> float:
    """Calculates the skewness of a sample.

    Uses the type 2method in the R e1071::skewness implementation, which is
    the version used in SAS and SPSS according to the documentation.

    :param x: Sample.
    :return: A float representing the skewness.
    """
    mu = x.mean()
    n = x.size
    deltas = x - mu
    m2 = np.sum(np.power(deltas, 2)) / n
    m3 = np.sum(np.power(deltas, 3)) / n
    g1 = m3 / np.power(m2, 3 / 2)
    return g1 * np.sqrt(n * (n - 1)) / (n - 2)

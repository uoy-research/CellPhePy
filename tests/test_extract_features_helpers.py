from __future__ import annotations

import numpy as np
import pytest

from cellphe.features.frame import var_from_centre


def test_var_from_centre():
    boundaries = np.column_stack((np.arange(1, 11), np.arange(11, 21)))
    output = var_from_centre(boundaries)
    assert output[0] == pytest.approx(3.535534)
    assert output[1] == pytest.approx(4.444444)

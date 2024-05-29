from __future__ import annotations

import numpy as np

from cellphe.input import read_roi


def test_read_roi():
    input = "tests/resources/roi2.roi"
    output = read_roi(input)
    expected = np.array(
        [
            [545, 100],
            [546, 100],
            [547, 100],
            [548, 101],
            [548, 102],
            [549, 103],
            [550, 104],
            [551, 105],
            [551, 106],
            [551, 107],
            [551, 108],
            [550, 108],
            [549, 108],
            [548, 107],
            [547, 107],
            [546, 106],
            [545, 105],
            [545, 104],
            [545, 103],
            [545, 102],
            [545, 101],
        ]
    )
    assert (expected == output).all()

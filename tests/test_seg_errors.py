from __future__ import annotations

import pandas as pd

from cellphe.segmentation import remove_predicted_seg_errors


def test_remove_predicted_seg_errors():
    df = pd.DataFrame({"CellID": [1, 1, 3, 4, 5], "FrameID": [0, 1, 0, 0, 1], "Feat1": [8, 9, 10, 11, 12]})
    errors = [1, 4]
    expected = pd.DataFrame({"CellID": [3, 5], "FrameID": [0, 1], "Feat1": [10, 12]})
    output = remove_predicted_seg_errors(df, "CellID", errors)
    # Have to remove index from test as it will be different. The function
    # output index will have the row numbers from the raw larger dataframe,
    # while the target only has 1:2
    pd.testing.assert_frame_equal(output.reset_index(drop=True), expected.reset_index(drop=True))
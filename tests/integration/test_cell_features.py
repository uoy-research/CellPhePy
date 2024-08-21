from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cellphe.features.frame import *
from cellphe.input import import_data, read_roi, read_tiff
from cellphe.processing import normalise_image

pytestmark = pytest.mark.integration


def assert_frame_equal_extended_diff(df1, df2):
    cols = df1.columns.values
    incorrect_cols = []
    for col in cols:
        try:
            pd.testing.assert_frame_equal(df1.loc[:, [col]], df2.loc[:, [col]])
        except AssertionError as e:
            res = re.search("values are different \\(([0-9.]+) %\\)", str(e))
            # The only permitted discrepancy is a 1% difference within the 4
            # poly features
            if col[:4] == "poly" and res is not None and float(res.group(1)) < 1:
                pass
            else:
                print(e, "\n")
                incorrect_cols.append(col)

    if len(incorrect_cols) > 0:
        raise AssertionError(
            f"Following columns had errors: {', '.join(incorrect_cols)}\n{len(incorrect_cols)}/{len(cols)} ({len(incorrect_cols)/len(cols)*100:.2f}%)"
        )


def test_cell_features():
    # Read features from the full dataset and compare to the output from the R
    # package, saved as CSV
    expected = pd.read_csv("tests/resources/benchmark_features.csv")
    phase_features = import_data("data/05062019_B3_3_Phase-FullFeatureTable.csv", "Phase", 200)

    output = cell_features(phase_features, "data/05062019_B3_3_Phase", "data/05062019_B3_3_imagedata", 0.0028)
    # Rename x and y to match how it was in the R version
    output.rename(columns={"x": "xpos", "y": "ypos"}, inplace=True)

    assert_frame_equal_extended_diff(expected.reset_index(drop=True), output.reset_index(drop=True))


def test_extract_static_features():
    # Compare features for a single ROI from a single frame
    image = read_tiff("tests/resources/frame.tif")
    image = normalise_image(image, 0, 255)
    roi = read_roi("tests/resources/roi.roi")
    expected = [
        10.652648965011,
        0.841711747441643,
        0.244603424461705,
        24.4131112314674,
        21.7096458937546,
        387,
        0.0915976331360947,
        1.36950904392765,
        0.529307282415631,
        18.0277563773199,
        1.59843905343762,
        0.0589385388484289,
        7.78794416690062,
        94.4909560723514,
        61.2086112322435,
        -0.129330105967309,
        0.054126911332088,
        0.642140468227425,
        0.751170568561873,
        1.38332796413143,
        0.945110668404989,
        5.71660272256462,
        11.2709030100334,
        1.2115949701977,
        0.37748090465276,
        0.390739183706359,
        123.868798160663,
        0.800082015671963,
        -36.1805036934972,
        1069.79921633389,
        0.0437230249577782,
        1.64383561643836,
        0.572602739726027,
        1.4349105149426,
        0.865764051626487,
        5.73390880090073,
        11.3972602739726,
        1.19903347432548,
        0.505077515806511,
        0.902621621957693,
        126.654233309846,
        0.764886620345796,
        -24.8515309099602,
        1031.7299877834,
        0.0383867832847425,
        1.81292517006803,
        0.574209683873549,
        1.52950951520548,
        0.852532454323163,
        5.70764785506039,
        11.3231292517007,
        1.22160929269432,
        0.521722678161525,
        1.03102696505788,
        124.565544669579,
        0.697025192740825,
        -25.6275454325374,
        1028.40664781797,
        1.90885707287804,
        1.78003228533968,
        1.6539754955893,
        1.52930191099884,
        1.41436338913974,
        1.25237392533331,
        1.06610368560602,
        0.879498449755716,
        0.623873879947033,
        325.584615384615,
        458.523076923077,
    ]
    output = extract_static_features(image, roi)
    assert output == pytest.approx(expected)

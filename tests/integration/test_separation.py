from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellphe.separation import calculate_separation_scores, optimal_separation_features

pytestmark = pytest.mark.integration


@pytest.fixture()
def treated():
    yield pd.read_csv("data/TreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def untreated():
    yield pd.read_csv("data/UntreatedTraining.csv").drop(columns="Unnamed: 0")


@pytest.fixture()
def separation():
    yield pd.read_csv("tests/resources/benchmark_separation.csv")


def test_calculate_separation_scores(treated, untreated, separation):
    output = calculate_separation_scores(treated, untreated)
    expected = pd.read_csv("tests/resources/benchmark_separation.csv")

    # Prepare for testing
    output = output.sort_values("Feature", ascending=False).reset_index(drop=True)
    expected = separation.sort_values("Feature", ascending=False).reset_index(drop=True)

    pd.testing.assert_frame_equal(output, expected)


def test_optimal_separation_features(separation):
    np.random.seed(17)
    output = optimal_separation_features(separation)
    # Expected value from manual testing of the R package
    expected = np.array(
        [
            "Sph_mean",
            "Dis_l2_des",
            "Dis_l3_asc",
            "Dis_l3_des",
            "Vel_asc",
            "Vel_des",
            "Rad_std",
            "Rad_asc",
            "Rad_des",
            "Rad_l1_asc",
            "Rad_l1_des",
            "Rad_l1_max",
            "Rad_l2_asc",
            "Rad_l2_des",
            "Rad_l2_max",
            "Rad_l3_des",
            "VfC_mean",
            "VfC_std",
            "VfC_asc",
            "VfC_des",
            "VfC_max",
            "VfC_l1_asc",
            "VfC_l1_des",
            "VfC_l2_asc",
            "VfC_l2_des",
            "VfC_l3_des",
            "Len_std",
            "Len_asc",
            "Len_des",
            "Len_max",
            "Len_l1_asc",
            "Len_l1_des",
            "Len_l2_asc",
            "Len_l2_des",
            "Len_l2_max",
            "Len_l3_des",
            "Wid_asc",
            "Wid_des",
            "A2B_mean",
            "Rect_mean",
            "Rect_std",
            "Rect_asc",
            "Rect_des",
            "Rect_max",
            "Rect_l1_des",
            "Rect_l2_asc",
            "Rect_l2_des",
            "Rect_l2_max",
            "Rect_l3_des",
            "poly1_mean",
            "poly1_std",
            "poly1_asc",
            "poly1_des",
            "poly1_max",
            "poly1_l1_asc",
            "poly1_l2_des",
            "poly1_l3_des",
            "poly4_mean",
            "poly4_std",
            "poly4_asc",
            "poly4_des",
            "poly4_max",
            "poly4_l1_asc",
            "poly4_l1_des",
            "poly4_l2_asc",
            "poly4_l2_des",
            "poly4_l2_max",
            "poly4_l3_des",
            "FOmean_mean",
            "FOsd_l2_des",
            "FOskew_mean",
            "Cooc01ASM_asc",
            "Cooc01Ent_asc",
            "Cooc01Ent_des",
            "Cooc01Sav_mean",
            "Cooc01Sha_mean",
            "Cooc12ASM_asc",
            "Cooc12ASM_des",
            "Cooc12Sha_mean",
            "Cooc12Pro_asc",
            "Cooc12Pro_des",
            "Cooc02ASM_asc",
            "Cooc02ASM_des",
            "Cooc02Sha_mean",
            "Cooc02Pro_asc",
            "Cooc02Pro_des",
            "IQ1_std",
            "IQ1_asc",
            "IQ1_des",
            "IQ1_max",
            "IQ1_l1_asc",
            "IQ1_l1_des",
            "IQ1_l1_max",
            "IQ1_l2_asc",
            "IQ1_l2_des",
            "IQ1_l2_max",
            "IQ1_l3_asc",
            "IQ1_l3_des",
            "IQ1_l3_max",
            "IQ2_std",
            "IQ2_asc",
            "IQ2_des",
            "IQ2_max",
            "IQ2_l1_asc",
            "IQ2_l1_des",
            "IQ2_l1_max",
            "IQ2_l2_asc",
            "IQ2_l2_des",
            "IQ2_l2_max",
            "IQ2_l3_asc",
            "IQ2_l3_des",
            "IQ2_l3_max",
            "IQ3_std",
            "IQ3_asc",
            "IQ3_des",
            "IQ3_max",
            "IQ3_l1_asc",
            "IQ3_l1_des",
            "IQ3_l1_max",
            "IQ3_l2_asc",
            "IQ3_l2_des",
            "IQ3_l2_max",
            "IQ3_l3_asc",
            "IQ3_l3_des",
            "IQ4_std",
            "IQ4_asc",
            "IQ4_des",
            "IQ4_max",
            "IQ4_l1_asc",
            "IQ4_l1_des",
            "IQ4_l1_max",
            "IQ4_l2_asc",
            "IQ4_l2_des",
            "IQ4_l2_max",
            "IQ4_l3_asc",
            "IQ4_l3_des",
            "IQ5_asc",
            "IQ5_des",
            "IQ5_l1_asc",
            "IQ5_l1_des",
            "IQ5_l2_asc",
            "IQ5_l2_des",
            "IQ5_l3_des",
            "IQ6_asc",
            "IQ6_des",
            "IQ6_l1_des",
            "IQ6_l2_asc",
            "IQ6_l2_des",
            "IQ6_l3_des",
        ]
    )

    assert (np.sort(output) == np.sort(expected)).all()

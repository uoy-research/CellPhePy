from __future__ import annotations

import numpy as np
import pytest
import scyjava as sj

from cellphe.imagej import setup_imagej
from cellphe.trackmate import load_tracker


@pytest.fixture()
def ij():
    yield setup_imagej()


def test_load_tracker_unknown_tracker_raises_error(ij):
    with pytest.raises(KeyError):
        load_tracker(None, "Unknown", None)


def test_load_tracker_settings_imported(ij):
    settings = sj.jimport("fiji.plugin.trackmate.Settings")()
    load_tracker(settings, "SimpleLAP", {"LINKING_MAX_DISTANCE": 25})
    new_distance = settings.trackerSettings["LINKING_MAX_DISTANCE"]
    assert new_distance == 25

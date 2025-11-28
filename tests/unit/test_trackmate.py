from __future__ import annotations

import pytest

# Only run tests if have optional dependencies available
pytest.importorskip("imagej")

import numpy as np
import scyjava as sj

from cellphe.tracking.imagej import setup_imagej
from cellphe.tracking.trackmate import load_tracker

pytestmark = pytest.mark.full


@pytest.fixture()
def ij():
    yield setup_imagej()


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def shutdown_jvm():
        sj.shutdown_jvm()

    request.addfinalizer(shutdown_jvm)


def test_load_tracker_unknown_tracker_raises_error(ij):
    with pytest.raises(KeyError):
        load_tracker(None, "Unknown", None)


def test_load_tracker_settings_imported(ij):
    settings = sj.jimport("fiji.plugin.trackmate.Settings")()
    load_tracker(settings, "SimpleSparseLAP", {"LINKING_MAX_DISTANCE": 25})
    new_distance = settings.trackerSettings["LINKING_MAX_DISTANCE"]
    assert new_distance == 25

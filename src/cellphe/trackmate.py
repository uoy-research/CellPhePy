"""
    cellphe.trackmate
    ~~~~~~~~~~~~~~~~~

    Functions related to importing TrackMate functionality.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scyjava as sj


def get_trackmate_xml(model, settings) -> str:
    """
    Retrieves the XML output from a TrackMate run.

    :param model: An instance of the Java class fiji.plugin.Trackmate.Model
    :param settings: An instance of the Java class
        fiji.plugin.Trackmate.Settings
    :return: An XML formatted string.
    """
    file_cls = sj.jimport("java.io.File")
    writer_cls = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")
    # There isn't a TmXml constructor without File, even if you don't write to it
    writer = writer_cls(file_cls(""))
    writer.appendSettings(settings)
    writer.appendModel(model)
    return str(writer.toString())


def parse_trackmate_xml(xml: str) -> list[pd.DataFrame, list]:
    """
    Parses the TrackMate XML output into a list of the tracked cells and their
    ROIs.

    :param xml: The XML contents.
    :return: A list with 2 items:
        - A DataFrame with the same contents as the exported Spots table in the
        TrackMate TableViewer in the GUI
        - A list of dictionaries representing Cells. Each dictionary has the
        following keys:
            - ID: CellID
            - frame: Frame number
            - coords: 2D Numpy array of the ROI coordinates as (x,y) pairs
    """
    tree = ET.fromstring(xml)
    spot_records = []
    rois = []
    # Get all Spots firstly
    for frame in tree.findall("./Model/AllSpots/SpotsInFrame"):
        # Get all spots, reading in their attributes and ROIs
        for spot in frame.findall("Spot"):
            spot_records.append(spot.attrib)
            # Read ROIs
            coords = np.array([spot.text.split(" ")]).astype(float)
            coords = coords.reshape(int(coords.size / 2), 2)
            coords[:, 0] = coords[:, 0] + float(spot.attrib["POSITION_X"])
            coords[:, 1] = coords[:, 1] + float(spot.attrib["POSITION_Y"])
            rois.append({"ID": spot.attrib["name"], "frame": spot.attrib["FRAME"], "coords": coords})
    spot_df = pd.DataFrame.from_records(spot_records)
    spot_df = spot_df.rename(columns={"name": "LABEL"})

    # Then get all Tracks so can add TRACK_ID
    track_records = []
    for track in tree.findall("./Model/AllTracks/Track"):
        for i, edge in enumerate(track.findall("Edge")):
            track_records.append({"TRACK_ID": track.attrib["TRACK_ID"], "ID": edge.attrib["SPOT_TARGET_ID"]})
            # We are parsng the edges list getting the target cellID from each
            # edge. To complete the set we also need the first source cellid.
            if i == 0:
                track_records.append({"TRACK_ID": track.attrib["TRACK_ID"], "ID": edge.attrib["SPOT_SOURCE_ID"]})
    track_df = pd.DataFrame.from_records(track_records)

    # Combine Spots and Tracks
    comb_df = pd.merge(spot_df, track_df, on="ID")
    # Reorder columns to be the same as exported from the GUI
    col_order = [
        "LABEL",
        "ID",
        "TRACK_ID",
        "QUALITY",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "POSITION_T",
        "FRAME",
        "RADIUS",
        "VISIBILITY",
        "MEAN_INTENSITY_CH1",
        "MEDIAN_INTENSITY_CH1",
        "MIN_INTENSITY_CH1",
        "MAX_INTENSITY_CH1",
        "TOTAL_INTENSITY_CH1",
        "STD_INTENSITY_CH1",
        "CONTRAST_CH1",
        "SNR_CH1",
        "ELLIPSE_X0",
        "ELLIPSE_Y0",
        "ELLIPSE_MAJOR",
        "ELLIPSE_MINOR",
        "ELLIPSE_THETA",
        "ELLIPSE_ASPECTRATIO",
        "AREA",
        "PERIMETER",
        "CIRCULARITY",
        "SOLIDITY",
        "SHAPE_INDEX",
    ]
    comb_df = comb_df[col_order]
    return comb_df, rois


def load_detector(settings) -> None:
    """
    Loads a TrackMate detector.
    Currently hardcoded to be the LabelImageDetector, as this works with the
    labelled masks output from Cellpose.

    :param settings: An instance of the Java class
        fiji.plugin.Trackmate.Settings
    :return: None, updates settings as a side-effect.
    """
    settings.detectorFactory = sj.jimport("fiji.plugin.trackmate.detection.LabelImageDetectorFactory")()
    settings.detectorSettings = settings.detectorFactory.getDefaultSettings()


def load_tracker(settings, tracker: str, tracker_settings: dict) -> None:
    """
    Loads a TrackMate tracker.

    :param settings: An instance of the Java class
        fiji.plugin.Trackmate.Settings
    :param tracker: String specifying which tracking algorithm to use.
    :param tracker_settings: Dictionary containing parameters for the specified
        tracker.
    :return: None, updates settings as a side-effect.
    """
    options = {"SimpleLAP": "fiji.plugin.trackmate.tracking.jaqaman.SimpleSparseLAPTrackerFactory"}
    try:
        selected = options[tracker]
    except KeyError as ex:
        raise KeyError(f"tracker must be one of {','.join(options.keys())}") from ex

    settings.trackerFactory = sj.jimport(selected)()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    if tracker_settings is not None:
        for k, v in tracker_settings.items():
            settings.trackerSettings[k] = v


def configure_trackmate(model, settings):
    """
    Instantiates a TrackMate object with the specified model and settings.

    :param model: An instance of the Java class fiji.plugin.Trackmate.Model
    :param settings: An instance of the Java class
        fiji.plugin.Trackmate.Settings
    :return: An instance of the Java class fiji.plugin.TrackMate.
    """
    settings.addAllAnalyzers()
    settings.initialSpotFilterValue = 1.0

    tm_cls = sj.jimport("fiji.plugin.trackmate.TrackMate")
    trackmate = tm_cls(model, settings)
    trackmate.computeSpotFeatures(True)
    trackmate.computeTrackFeatures(True)

    return trackmate

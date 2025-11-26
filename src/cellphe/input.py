"""
cellphe.input
~~~~~~~~~~~~~

Functions related to importing feature tables into CellPhe from various
microscopy platforms.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from roifile import ImagejRoi
from skimage import io

from cellphe.processing.roi import save_rois


def import_data(file: str, source: str, minframes: int = 0) -> pd.DataFrame:
    """Copy metadata and cell-frame features from an existing TrackMate or
    PhaseFocus export.

    Loads the frame and cell IDs along with the filename used to refer to each ROI.
    None of the TrackMate generated features are retained, while for PhaseFocus
    sources volume and sphericity features are also extracted.
    Only cells that are tracked for a minimum of `minframes` are included.

    :param file: The filepath to a CSV file containing features output by PhaseFocus or Trackmate software.
    :type file: str
    :param source: The name of the software that produced the metadata file.
        - Trackmate_auto refers to an exported CSV produced within CellPhe by
        track_images().
        - Trackmate_imagej refers to an exported CSV from the ImageJ GUI.
        - Phase refers to a CSV exported by PhaseFocus software.
    :type source: str
    :param minframes: The minimum number of frames a cell must be tracked for to
        be included in the output features.
    :type minframes: int
    :return: A dataframe with 1 row corresponding to 1 cell tracked in 1 frame
        with the following columns:
          * ``FrameID``: the numeric FrameID
          * ``CellID``: the numeric CellID
          * ``ROI_filename``: the label used to refer to this ROI
          * ``Volume``: a real-valued number
          * ``Sphericity``: a real-valued number
    """
    sources = ["Phase", "Trackmate_imagej", "Trackmate_auto"]
    if source not in sources:
        raise ValueError(f"Invalid source value '{source}'. Must be one of {', '.join(sources)}")

    if source == "Phase":
        df = pd.read_csv(file, skiprows=1, encoding="utf-8-sig")
        df["ROI_filename"] = df["Frame"].astype(str) + "-" + df["Tracking ID"].astype(str)
        out = df[["Frame", "Tracking ID", "ROI_filename", "Volume (µm³)", "Sphericity ()"]]
        out = out.rename(
            columns={
                "Frame": "FrameID",
                "Tracking ID": "CellID",
                "Volume (µm³)": "Volume",
                "Sphericity ()": "Sphericity",
            }
        )
    elif source == "Trackmate_imagej":
        df = pd.read_csv(file)
        # Lines 2-4 in the raw file contain additional header information and can be safely discarded
        out = df.loc[3 : df.shape[0], ["FRAME", "TRACK_ID", "LABEL"]]
        out = out.rename(columns={"FRAME": "FrameID", "TRACK_ID": "CellID", "LABEL": "ROI_filename"})
        out["FrameID"] = out["FrameID"].astype(int) + 1  # Convert from 0-indexed to 1-indexed
    elif source == "Trackmate_auto":
        # Basically the same as Trackmate_imagej but with 3 differences:
        #   - No redundant header lines
        #   - There is already a column called ROI_FILENAME
        #   - FrameIDs and CellIDs are already 1-indexed
        df = pd.read_csv(file)
        out = df[["FRAME", "TRACK_ID", "ROI_FILENAME"]]
        out = out.rename(columns={"FRAME": "FrameID", "TRACK_ID": "CellID", "ROI_FILENAME": "ROI_filename"})

    # Want IDs as integers
    out["CellID"] = out["CellID"].astype(int)
    out["FrameID"] = out["FrameID"].astype(int)

    # Restrict to cells which are in minimum number of frames
    out = out.groupby("CellID").filter(lambda x: x["FrameID"].count() >= minframes)

    # Order by CellID and FrameID to help manual inspection of the data
    out = out.sort_values(["CellID", "FrameID"])

    return out


def read_rois(archive: str) -> dict[str, np.array]:
    """Reads multiple ROI files saved in a Zip archive.

    :param archive: Filepath to an archive containing ROI files.
    :return: A dict where each entry is a 2D numpy array containing the
        coordinates, and the keys are the ROI filenames ("<frameid>-<roiid.roi").
    """
    rois = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            with zf.open(name, "r") as roi_f:
                raw = roi_f.read()
                roi = ImagejRoi.frombytes(raw)
                rois[name] = roi.integer_coordinates + [roi.left, roi.top]
    # The coordinates() method returns the subpixel coordinates for TrackMate
    # ROIs as these are available. These are floats however and result in
    # problems downstream. Want to explicitly use the integer coordinates.
    return rois


def read_roi(filename: str) -> np.array:
    """Returns the coordinates from an ImageJ produced ROI file.

    :param filename: Filepath to the ROI file (extension .roi).
    :return: A 2D numpy array containing the coordinates.
    """
    roi = ImagejRoi.fromfile(filename)
    # The coordinates() method returns the subpixel coordinates for TrackMate
    # ROIs as these are available. These are floats however and result in
    # problems downstream. Want to explicitly use the integer coordinates.
    coords = roi.integer_coordinates + [roi.left, roi.top]
    return coords


def read_tiff(filename: str) -> np.array:
    """Reads a TIF image into a Numpy array.

    :param filename: TIF filename.
    :return: A 2D Numpy array.
    """
    image = Image.open(filename)
    if image.mode == "RGB":
        image = image.convert("L")
    image = np.array(image)
    return image

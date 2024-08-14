"""
    cellphe.input
    ~~~~~~~~~~~~~

    Functions related to importing feature tables into CellPhe from various
    microscopy platforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from read_roi import read_roi_file


def copy_features(file: str, minframes: int, source: str = "Phase") -> pd.DataFrame:
    """Copy metadata and cell-frame features from an existing PhaseFocus or Trackmate table

    Loads the frame and cell IDs along with the filename used to refer to each ROI.
    For PhaseFocus generated data, volume and sphericity features are also extracted.
    Only cells that are tracked for a minimum of `minframes` are included.

    :param file: The filepath to a CSV file containing features output by PhaseFocus or Trackmate software.
    :type file: str
    :param minframes: The minimum number of frames a cell must be tracked for to
        be included in the output features.
    :type minframes: int
    :param source: The name of the software that produced the metadata file,
        either 'Phase' or 'Trackmate' are currently supported.
    :type source: str
    :return: A dataframe with 1 row corresponding to 1 cell tracked in 1 frame
        with the following columns:
          * ``FrameID``: the numeric FrameID
          * ``CellID``: the numeric CellID
          * ``ROI_filename``: the label used to refer to this ROI
          * ``Volume``: a real-valued number
          * ``Sphericity``: a real-valued number
    """
    sources = ["Phase", "Trackmate"]
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
    elif source == "Trackmate":
        df = pd.read_csv(file)
        # Lines 2-4 in the raw file contain additional header information and can be safely discarded
        out = df.loc[3 : df.shape[0], ["FRAME", "TRACK_ID", "LABEL"]]
        out = out.rename(
            columns={
                "Frame": "FrameID",
                "Tracking ID": "CellID",
                "Volume (µm³)": "Volume",
                "Sphericity ()": "Sphericity",
            }
        )
        out["FrameID"] = out["FrameID"].astype(int) + 1  # Convert from 0-indexed to 1-indexed

    # Want IDs as integers
    out["CellID"] = out["CellID"].astype(int)
    out["FrameID"] = out["FrameID"].astype(int)

    # Restrict to cells which are in minimum number of frames
    out = out.groupby("CellID").filter(lambda x: x["FrameID"].count() >= minframes)

    # Order by CellID and FrameID to help manual inspection of the data
    out = out.sort_values(["CellID", "FrameID"])

    return out


def read_roi(filename: str) -> np.array:
    """Returns the coordinates from an ImageJ produced ROI file.

    :param filename: Filepath to the ROI file (extension .roi).
    :return: A 2D numpy array containing the coordinates.
    """
    fn_stripped = Path(filename).stem
    roi = read_roi_file(filename)
    x = roi[fn_stripped]["x"]
    y = roi[fn_stripped]["y"]
    return np.column_stack((x, y))


def read_tiff(filename: str) -> np.array:
    """Reads a TIF image into a Numpy array.

    :param filename: TIF filename.
    :return: A 2D Numpy array.
    """
    image = Image.open(filename)
    image = np.array(image)
    return image

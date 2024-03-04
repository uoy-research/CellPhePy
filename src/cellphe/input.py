from __future__ import annotations

import pandas as pd


def copy_features(file: str, minframes: int, source: str = "Phase") -> pd.DataFrame:
    """
    Copy metadata and cell-frame features from an existing PhaseFocus or Trackmate table

    Loads the frame and cell IDs along with the filename used to refer to each ROI.
    For PhaseFocus generated data, volume and sphericity features are also extracted.
    Only cells that are tracked for a minimum of `minframes` are included.

    :param file: The filepath to a CSV file containing features output by PhaseFocus or Trackmate software.
    :type file: str
    :param minframes: The minimum number of frames a cell must be tracked for to
    be included in the output features.
    :type minframes: int
    :param source: The name of the software that produced the metadata file, either 'Phase' or 'Trackmate' are currently supported.
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
        df = pd.read_csv(file, skiprows=1)
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

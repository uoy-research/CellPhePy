"""
    cellphe.input
    ~~~~~~~~~~~~~

    Functions related to importing feature tables into CellPhe from various
    microscopy platforms.
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imagej
import numpy as np
import pandas as pd
import scyjava as sj
from cellpose import models
from PIL import Image
from roifile import ImagejRoi
from skimage import io


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
        out = out.rename(columns={"FRAME": "FrameID", "TRACK_ID": "CellID", "LABEL": "ROI_filename"})
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
    roi = ImagejRoi.fromfile(filename)
    return roi.coordinates()


def read_tiff(filename: str) -> np.array:
    """Reads a TIF image into a Numpy array.

    :param filename: TIF filename.
    :return: A 2D Numpy array.
    """
    image = Image.open(filename)
    image = np.array(image)
    return image


def segment_images(input_dir: str, output_dir: str) -> None:
    """
    Segments a batch of images using cellpose.

    Currently only TIFs are supported. The output segmentations are saved to
    disk as TIFs with index labelling. I.e. on each frame, all pixels belonging
    to the first identified cell are given value 1, the second cell are
    assigned 3 etc... Background pixels are 0.

    :param input_dir: The path to the directory containing the TIFs.
    :param output_dir: The path to the directory where the masks will be saved
    to.
    :return: None, saves masks to disk as a side-effect.
    """
    model = models.Cellpose(gpu=False, model_type="cyto")
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    os.makedirs(output_dir, exist_ok=True)
    for tif_file in tif_files:
        print(f"Processing image: {tif_file}")
        try:
            image = read_tiff(tif_file)
            masks, _, _, _ = model.eval(image)

            # Save masks
            filename = os.path.splitext(os.path.basename(tif_file))[0] + "_mask.tif"
            save_path = os.path.join(output_dir, filename)
            io.imsave(save_path, masks.astype("uint16"))  # Assuming masks are uint16

            print(f"Masks saved for {tif_file} at {save_path}")
        except Exception as e:
            print(f"Error processing file {tif_file}: {e}")


def track_images(mask_dir: str, csv_filename: str, roi_folder: str) -> None:
    """
    Tracks cells across a set of frames using TrackMate, storing the frame
    features in a CSV, and the ROIs in a specified folder.

    The CSV will have the same columns as that extracted by the TrackMate GUI
    (from the Spots tab of the Tracks Table).
    The ROIs will be saved in the binary ROI format and can be read into
    TrackMate.
    NB: The IDs will not necessarily be identical to those from outputs from the
    GUI TrackMate.

    As this function doesn't require a local install of ImageJ, it downloads a
    remote version from the SciJava Maven repository. As such the first run can
    a while (~10 mins), although this will be cached for subsequent uses.

    :param mask_dir: Path to the directory containing the image masks created by
    CellPose as in segment_images.
    :param csv_filename: Filename for the resultant CSV.
    :param roi_folder: Folder where ROIs will be saved to. Will be created if it
        doesn't exist.
    :return: None, writes the CSV file and ROI files to disk as a side-effect.
    """
    # TODO refactor into smaller functions
    # TODO interpolate ROIs
    # TODO use roifile for input too
    os.makedirs(roi_folder, exist_ok=True)
    ij = imagej.init(["net.imagej:imagej", "sc.fiji:TrackMate:7.13.2"], add_legacy=False)

    FolderOpener = sj.jimport("ij.plugin.FolderOpener")
    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    TrackMate = sj.jimport("fiji.plugin.trackmate.TrackMate")
    LabelImageDetectorFactory = sj.jimport("fiji.plugin.trackmate.detection.LabelImageDetectorFactory")
    SimpleSparseLAPTrackerFactory = sj.jimport("fiji.plugin.trackmate.tracking.jaqaman.SimpleSparseLAPTrackerFactory")
    TmXmlWriter = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")
    File = sj.jimport("java.io.File")

    # Load all images as framestack
    imp = FolderOpener.open(mask_dir, "")

    # Swap T and Z in the same way that is done in the manual approach when
    # opening TrackMate
    # TODO Understand if this is needed in all situations or if it should be
    # checked for automatically
    dims = imp.getDimensions()
    imp.setDimensions(dims[2], dims[4], dims[3])

    # Trackmate datastructures
    model = Model()
    settings = Settings(imp)

    # Configure detection - using the CellPose labels
    settings.detectorFactory = LabelImageDetectorFactory()
    settings.detectorSettings = {"TARGET_CHANNEL": ij.py.to_java(1), "SIMPLIFY_CONTOURS": True}

    # Configure tracking - just using Simple LAP for now with default settings
    # TODO make the choice of tracker & parameters exposed to the user
    settings.trackerFactory = SimpleSparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = 15.0
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 15.0
    settings.trackerSettings["MAX_FRAME_GAP"] = ij.py.to_java(2)

    # Configure Trackmate itself
    settings.addAllAnalyzers()
    settings.initialSpotFilterValue = 1.0
    trackmate = TrackMate(model, settings)
    trackmate.computeSpotFeatures(True)
    trackmate.computeTrackFeatures(True)

    # Check settings are ok
    ok = trackmate.checkInput()
    if not ok:
        print("Settings error")
        sys.exit(str(trackmate.getErrorMessage()))

    # Run the full detection + tracking process
    ok = trackmate.process()
    if not ok:
        print("process error")
        sys.exit(str(trackmate.getErrorMessage()))

    # Export to XML so we can retrieve the Spots and Tracks info
    # There is an export to CSV method but it's only in Trackmate 12.2, which
    # isn't on the scijava public repository
    # TODO Make this use stdout rather than a file
    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as fp:
        out_file = File(fp.name)
        writer = TmXmlWriter(out_file)
        writer.appendSettings(settings)
        writer.appendModel(model)
        writer.writeToFile()
        tree = ET.parse(fp.name)

    # Parse XML
    spot_records = []
    rois = {}
    # Get all Spots firstly
    for frame in tree.findall("./Model/AllSpots/SpotsInFrame"):
        # Get all spots, reading in their attributes (saving to dict)
        spots = frame.findall("Spot")
        for spot in spots:
            spot_records.append(spot.attrib)
            coords_raw = np.array([spot.text.split(" ")]).astype(float)
            coords = coords_raw.reshape(int(coords_raw.size / 2), 2)
            coords[:, 0] = coords[:, 0] + float(spot.attrib["POSITION_X"])
            coords[:, 1] = coords[:, 1] + float(spot.attrib["POSITION_Y"])
            rois[spot.attrib["name"]] = coords
    spot_df = pd.DataFrame.from_records(spot_records)
    spot_df = spot_df.rename(columns={"name": "LABEL"})

    for cellid, roi in rois.items():
        fn = os.path.join(roi_folder, f"{cellid}.roi")
        roi_obj = ImagejRoi.frompoints(roi)
        roi_obj.tofile(fn)

    # Then get all Tracks so can add TRACK_ID
    track_records = []
    for track in tree.findall("./Model/AllTracks/Track"):
        track_id = track.attrib["TRACK_ID"]
        for i, edge in enumerate(track.findall("Edge")):
            track_records.append({"TRACK_ID": track_id, "ID": edge.attrib["SPOT_TARGET_ID"]})
            # We are parsng the edges list getting the target cellID from each
            # edge. To complete the set we also need the first source cellid.
            if i == 0:
                track_records.append({"TRACK_ID": track_id, "ID": edge.attrib["SPOT_SOURCE_ID"]})
    track_df = pd.DataFrame.from_records(track_records)

    # TODO extract ROIs

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
    comb_df.to_csv(csv_filename, index=False)

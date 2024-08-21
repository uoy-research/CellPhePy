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

import numpy as np
import pandas as pd
import scyjava as sj
from cellpose import models
from PIL import Image
from roifile import ImagejRoi
from skimage import io

from cellphe.imagej import read_image_stack, setup_imagej
from cellphe.processing.roi import save_rois
from cellphe.trackmate import configure_trackmate, get_trackmate_xml, load_detector, load_tracker, parse_trackmate_xml


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


def track_images(
    mask_dir: str,
    csv_filename: str,
    roi_folder: str,
    create_roi_zip: bool = False,
    tracker: str = "SimpleLAP",
    tracker_settings: dict = None,
) -> None:
    # pylint: disable=too-many-arguments
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
    :param create_roi_zip: Whether to create a Zip archive of the ROI files. If
        selected, a zip will be created with the name '<roi_folder>.zip', at the
        level above the roi_folder.
    :return: None, writes the CSV file and ROI files to disk as a side-effect.
    """
    os.makedirs(roi_folder, exist_ok=True)
    setup_imagej()

    imp = read_image_stack(mask_dir)
    settings = sj.jimport("fiji.plugin.trackmate.Settings")(imp)
    load_detector(settings)
    load_tracker(settings, tracker, tracker_settings)

    # Configure TrackMate instance
    model = sj.jimport("fiji.plugin.trackmate.Model")()
    trackmate = configure_trackmate(model, settings)
    if not trackmate.checkInput():
        print("Settings error")
        sys.exit(str(trackmate.getErrorMessage()))

    # Run the full detection + tracking process
    if not trackmate.process():
        print("process error")
        sys.exit(str(trackmate.getErrorMessage()))

    # Export to and extract the Spots, Tracks, and ROIs
    raw_xml = get_trackmate_xml(model, settings)
    comb_df, rois = parse_trackmate_xml(raw_xml)

    # Write CSV and ROIs to disk
    comb_df.to_csv(csv_filename, index=False)
    save_rois(rois, roi_folder, create_roi_zip)

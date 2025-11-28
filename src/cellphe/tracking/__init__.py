"""
Functions related to tracking segmented cells.
"""

from __future__ import annotations

import sys

import scyjava as sj

from cellphe.processing.roi import save_rois
from cellphe.tracking.imagej import read_image_stack, setup_imagej
from cellphe.tracking.trackmate import (
    configure_trackmate,
    get_trackmate_xml,
    load_detector,
    load_tracker,
    parse_trackmate_xml,
)


def track_images(
    mask_dir: str,
    csv_filename: str,
    roi_filename: str = "rois.zip",
    tracker: str = "SimpleSparseLAP",
    tracker_settings: dict = None,
    max_heap: int | None = None,
) -> None:
    # pylint: disable=too-many-positional-arguments
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
    :param roi_filename: Filename of output archive.
    :param max_heap: Size in GB of the maximum heap size allocated to the JVM.
        Use if you are encountering memory problems with large datasets. Be careful
        when using this parameter, the rule of thumb is not to assign more than 80%
        of your computer's available memory.
    :return: None, writes the CSV file and ROI files to disk as a side-effect.
    """
    setup_imagej(max_heap)

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
    save_rois(rois, roi_filename)

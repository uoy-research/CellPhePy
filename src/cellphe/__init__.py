"""
cellphe
~~~~~~~

Top level package.
"""

from __future__ import annotations

from cellphe.classification import classify_cells as classify_cells
from cellphe.clustering import identify_clusters as identify_clusters
from cellphe.features import cell_features as cell_features
from cellphe.features import time_series_features as time_series_features
from cellphe.input import import_data as import_data
from cellphe.separation import calculate_separation_scores as calculate_separation_scores


def segment_images(*args, **kwargs):
    """
    Segments a batch of images using cellpose.

    Currently only TIFs are supported. The output segmentations are saved to
    disk as TIFs with index labelling. I.e. on each frame, all pixels belonging
    to the first identified cell are given value 1, the second cell are
    assigned 3 etc... Background pixels are 0.

    :param input_dir: The path to the directory containing the TIFs.
    :param output_dir: The path to the directory where the masks will be saved
    to.
    :param model_params: Parameters to pass into the Cellpose instantiation,
    including the model type. See
    https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.Cellpose
    for a full list of options.
    :param eval_params: Parameters to pass into the Cellpose eval function
    governing the segmentation.
    https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.CellposeModel.eval
    for a full list of options.
    :return: None, saves masks to disk as a side-effect.
    """
    # Hacky wrapper around segment_images that imports at runtime rather than
    # when the top level 'cellphe' module is imported. Ideally this re-export
    # of `segment_images` wouldn't be done and users would instead use the full
    # path (i.e. `from cellphe.segmentation import segment_images`), but this is
    # retained for backwards compatibility.
    try:
        from cellphe.segmentation import segment_images as dummy

        return dummy(*args, **kwargs)
    except ModuleNotFoundError:
        print(
            "Unable to load cellpose. Ensure that the full version of "
            "cellphe was installed with `pip install cellphe[full]`."
        )


def track_images(*args, **kwargs):
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
    # Hacky wrapper around track_images that imports at runtime rather than
    # when the top level 'cellphe' module is imported. Ideally this re-export
    # of `track_images` wouldn't be done and users would instead use the full
    # path (i.e. `from cellphe.tracking import track_images`), but it is
    # retained for backwards compatibility.
    try:
        from cellphe.tracking import track_images as dummy

        return dummy(*args, **kwargs)
    except ModuleNotFoundError:
        print(
            "Unable to load trackmate. Ensure that the full version of "
            "cellphe was installed with `pip install cellphe[full]`."
        )

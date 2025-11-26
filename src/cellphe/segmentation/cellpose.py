"""
cellphe.segmentation.cellpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions related to interfacing with cellpose.
"""

from __future__ import annotations

import glob
import os

from cellpose import models


def segment_images(
    input_dir: str, output_dir: str, model_params: dict = {"model_type": "cyto3"}, eval_params: dict = {}  # noqa: B006
) -> None:  # noqa: B006
    # Can ignore {} warning in both pylint and flake8 as the dicts aren't being
    # modified within the function.
    # pylint: disable=dangerous-default-value
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
    model = models.Cellpose(**model_params)
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError:
        pass  # exist_ok doesn't work if dir exists but with different mode
    for tif_file in tif_files:
        print(f"Processing image: {tif_file}")
        try:
            image = read_tiff(tif_file)
            masks, _, _, _ = model.eval(image, **eval_params)

            # Save masks
            filename = os.path.splitext(os.path.basename(tif_file))[0] + "_mask.tif"
            save_path = os.path.join(output_dir, filename)
            io.imsave(save_path, masks.astype("uint16"))  # Assuming masks are uint16

            print(f"Masks saved for {tif_file} at {save_path}")
        except Exception as e:
            print(f"Error processing file {tif_file}: {e}")

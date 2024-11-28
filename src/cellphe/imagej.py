"""
    cellphe.imagej
    ~~~~~~~~~~~~~~

    Functions related to using ImageJ functionality.
"""

from __future__ import annotations

import imagej
import scyjava as sj


def setup_imagej(max_heap: int | None = None) -> None:
    """
    Sets up a JVM with ImageJ and the TrackMate plugin loaded.

    :param max_heap: Size in GB of the maximum heap size allocated to the JVM.
    Use if you are encountering memory problems with large datasets. Be careful
    when using this parameter, the rule of thumb is not to assign more than 80%
    of your computer's available memory.
    :return: None, although could be updated to return reference to a Java class
    net.imagej.ImageJ.
    """
    if max_heap is not None and isinstance(max_heap, int) and max_heap > 0:
        sj.config.add_option(f"-Xmx{max_heap}g")
    imagej.init(["net.imagej:imagej", "sc.fiji:TrackMate:7.13.2"], add_legacy=False)


def read_image_stack(image_dir: str):
    """
    Reads a directory containing TIFs into ImageJ as a stack.

    :param dir: Directory where the TIFs are located.
    :return: An ImagePlus instance containing an ImageStack.
    """
    # pylint doesn't like the camel case naming. I think it helps readability as
    # it denotes a Java class
    # pylint: disable=invalid-name
    FolderOpener = sj.jimport("ij.plugin.FolderOpener")
    # Load all images as framestack
    imp = FolderOpener.open(image_dir, "")

    # When reading in the imagestack, the the number of frames is often
    # (always?) interpreted as the number of channels. This corrects that in the
    # same way as the info box that pops up when using TrackMate in the GUI
    # Dims are ordered X Y Z C T
    dims = imp.getDimensions()
    if dims[4] == 1:
        # If time dimension is actually in Z, swap Z & T
        if dims[2] > 1:
            imp.setDimensions(dims[4], dims[3], dims[2])
        # If time dimension is actually in channels (usual case), swap C & T
        elif dims[3] > 1:
            imp.setDimensions(dims[2], dims[4], dims[3])
        # If none of Z, C, T contain more than 1 (i.e. time), then we have an
        # error
        else:
            raise ValueError(
                f"""Time-dimension could not be identified as none of the Z, C, or T
                channels contain more than 1 value: {dims[2:]}"""
            )
    return imp

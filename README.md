# CellPhe

<!-- badges: start -->
<a href="https://zenodo.org/badge/latestdoi/449769672"><img src="https://zenodo.org/badge/449769672.svg" alt="DOI"></a>
<!-- badges: end -->

CellPhe provides functions to phenotype cells from time-lapse videos and accompanies the paper:\
Wiggins, L., Lord, A., Murphy, K.L. et al.\
The CellPhe toolkit for cell phenotyping using time-lapse imaging and pattern recognition.\
Nat Commun 14, 1854 (2023).\
https://doi.org/10.1038/s41467-023-37447-3

The Python package is a port of the [original R implementation](https://github.com/uoy-research/CellPhe).

# Installation

## Basic install (phenotyping only)

You can install the latest version of CellPhe from
[PyPi](https://pypi.org/project/cellphe/) with:

```bash
pip install cellphe
```

The default installation provides access to the core phenotyping functionality, but if you would also like to segment and track your images, the full installation will need to be installed as below. Segmentation and tracking have large dependencies and so are not included by default.

## Full install (segmentation + tracking)

The tracking functionality depends on the ImageJ plugin TrackMate, which in turn depends on having a Java runtime available.
We recommend the **Eclipse Temurin** variant, which is built on OpenJDK and is free, stable, and widely supported.

1.  **Download:** Go to the [Adoptium Temurin website](https://adoptium.net/).
2.  **Select & Install:**
    * **Windows:** Click the "Latest LTS Release" button. Run the downloaded `.msi` file and follow the prompts. (Ensure the "Set `JAVA_HOME` variable" option is checked during installation).
    * **macOS:** Download the `.pkg` file for your chip type (Apple Silicon or Intel). Open it and follow the installation prompts.
    * **Linux (Debian/Ubuntu):** Open a terminal and run the following (NB: you can also install Canonical's build from the `openjdk-21-jdk` package)
        ```bash
        sudo apt update
        sudo apt install temurin-21-jdk
        ```

After installing, open a **new** terminal/command prompt and run:
```bash
java -version
```
If you see the version details, your Java installation is successful.

The full version of CellPhe can now be installed:
```bash
pip install cellphe[full]
```

## Windows issues

Windows users might encounter some additional OS-specific issues.
This section aims to document some of these in order to remove any barriers to use.

### Installing Python

Unlike MacOS or Linux, Python doesn't come installed with Windows by default.
The best way to install Python is from the [official website](https://www.python.org/downloads/windows/), with the following considerations:

  - CellPhe isn't compatible with Python 3.14 - we'd recommend either 3.12 or 3.13
  - Check the box for adding Python to `PATH` during the install process
  - It's a good idea to also install the separate Install Manager from the same website to allow for [quicker switching between Python versions](https://docs.python.org/3/using/windows.html#installing-runtimes) in future

### Visual C++ Build Tools

If you come across an error message like the below when installing CellPhe (or other Python packages), the solution is to install C++ Build Tools from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (tick 'Desktop development with C++').

```python
  WARNING: Failed to activate VS environment: Could not find C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe

      ..\meson.build:1:0: ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
```

### Pip not found

If you get an error when running `pip install cellphe` about `pip` not being found ("'pip' is not recognized as an internal or external command" or similar), try running `python -m pip install cellphe` instead.
This is because `pip` isn't always added to the `PATH` in Windows.

# Example Usage

An example dataset to demonstrate CellPhe’s capabilities is hosted on [Dryad](https://doi.org/10.5061/dryad.4xgxd25f0) in the archive `example_data.zip` and comprises 3 parts:

-   The time-lapse stills as TIFF images (`05062019_B3_3_imagedata`)
-   Existing pre-extracted features from PhaseFocus Livecyte.
    (`05062019_B3_3_Phase-FullFeatureTable.csv`)
-   Region-of-interest (ROI) boundaries already demarked in ImageJ
    format (`05062019_B3_3_Phase`)

These should be extracted into a suitable location (this guide assumes they have been extracted into `data`) before proceeding
with the rest of the tutorial.

The first step in the CellPhe workflow is to prepare a dataframe containing
metadata identifying the tracked cells across all the frames, along with any
pre-existing attributes. The segmenting and tracking can be performed within
CellPhe, or pre-segmented and tracked data from two widely used software
(PhaseFocus Livecyte & Trackmate) can be directly imported.

## Segmenting and tracking

**NB: Please ensure that you have installed the full version of CellPhe as shown above before segmenting or tracking.**
This feature is still experimental, please report any bugs at the [issue tracker](https://github.com/uoy-research/CellPhePy/issues).

CellPhe provides 2 functions to segment and track an image sequence:

  - `segment_images`: Segments images using Cellpose
  - `track_images`: Uses the ImageJ plugin TrackMate to track cells between frames without requiring ImageJ to be installed

```python
from cellphe import segment_images, track_images
```

`segment_images` takes 4 arguments:

  - the path to the directory where the images are stored (where the folder `05062019_B3_3_imagedata` was extracted to)
  - a path to an output folder where the resultant Cellpose masks will be saved
  - parameters for the CellPose model instantiation, including the model type (defaults to `cyto3`).
  - parameters for the CellPose `eval` function which governs the segmentation

For the latter 2, refer to the [CellPose docs](https://cellpose.readthedocs.io/en/latest/models.html) for a full list of options.

Segmentation can take several minutes depending on the number of images and their resolution.

```python
segment_images("data/05062019_B3_3_imagedata", "data/masks")
```

Confirm that the `masks` directory has been created and populated with TIFs containing cell masks.
If it has, then you are ready to track the cells.
`track_images` takes at minimum 3 arguments:

  - the location of the masks created by `segment_images`
  - the filename to save the output metadata to
  - a filename for the output ROI zip

Optionally you can also change the tracking options - by default the Simple LAP method is employed - with the `tracker` and `tracker_settings` arguments.
NB: the first time tracking is run will be slow as an ImageJ instance will be setup. Subsequent runs will skip this step.

```python
track_images("data/masks", "data/tracked.csv", "data/rois.zip")
```

Confirm that the `tracked.csv` file was created and the `rois` folder has been populated with ROI files.
These outputs can now be loaded into CellPhe.

## Importing pre-segmented and tracked data

Once a metadata file (CSV format) and a zip of ROIs are available, either directly output from external software (PhaseFocus Livecyte or TrackMate in ImageJ), or from within CellPhe as in the previous section, they can be read into CellPhe.
The `import_data` function accepts metadata files from one of these sources and converts it into a standard format.
It takes 3 arguments: the metadata file path, the source, and the minimum number of frames that a cell must be tracked for to be retained in the dataset (optional).

For example, the dataset that was segmented and tracked in the previous section can be imported as:

```python
from cellphe import import_data
feature_table = import_data("data/tracked.csv", "Trackmate_auto", 50)
```

Alternatively, the example below creates the metadata dataframe from the supplied PhaseFocus dataset, only including cells that were tracked for at least 50 frames.

``` python
input_feature_table = "data/05062019_B3_3_Phase-FullFeatureTable.csv"
feature_table_phase = import_data(input_feature_table, "Phase", 50)
```

If a segmented and tracked dataset is available from a different source then it can still be used in CellPhe provided that it can be loaded into a `pandas.DataFrame` containing:

  - Each row corresponding to a cell tracked in a specific frame
  - A column `FrameID` (integer) denoted the frame number in chronological order
  - A column `CellID` (integer) identifying the cell
  - A column `ROI_filename` (string) denoting the filename (without extension) of the corresponding ROI file, not including the full path

Additional columns providing cell features can be included and will be retained and incorporated into the CellPhe analysis.
The PhaseFocus dataset keeps the volume and sphericity features, for example.

## Generating cell features

In addition to any pre-calculated features, the `cell_features()`
function generates 74 descriptive features for each cell on every frame
using the frame images and pre-generated cell boundaries, based on size,
shape, texture, and the local cell density. The output is a dataframe
comprising the `FrameID`, `CellID`, and `ROI_filename` columns
from the feature table input, the 74 features as columns,
and any additional features that may be present (such as from `import_data()`)
in further columns.

`cell_features()` takes as arguments the feature table, the archive where ROIs are saved, the folder where the images are, and the framerate.
It expects images to be saved with a filename ending with the frame id just before the file extension. The file extension can be `.tif`, `.tiff`, or the `ome.tif` and `.ome.tiff` equivalents.
The frame id can be zero-padded or not.
`myexperiment-1.tif`, `myexperiment_1.tiff`, `2.ome.tif` are all valid names.

ROI files are named according to the `ROI_filename` column but with a `.roi` extension.

The example below uses the PhaseFocus ROIs, but the ones generated using TrackMate just before can be used with the corresponding feature table.
There are 74 features generated during this step, which added to the 3 identifiers (`FrameID`, `CellID`, `ROI_filename`) and 2 PhaseFocus features (`Volume`, `Sphericity`) results in 79 columns.

```python
from cellphe import cell_features
roi_archive = "data/05062019_B3_3_Phase.zip"
image_folder = "data/05062019_B3_3_imagedata"
new_features = cell_features(feature_table_phase, roi_archive, image_folder, framerate=0.0028)
```

## Generating time-series features

The next step is to calculate features that incorporate the time-dimension.
This is done with the `time_series_features` function, which accepts a dataframe with the cell-level features as output earlier from `cell_features`.

Variables are calculated from the time series providing both
summary statistics and indicators of time-series behaviour at different
levels of detail obtained via wavelet analysis. 15 summary scores are
calculated for each feature, in addition to the cell trajectory, thereby
resulting in a default output of 1081 features (15x72 + 1).
With the 2 PhaseFocus features as well, this increases to 1111.
The output is a dataframe with the first column being the
`CellID` used previously.

``` python
from cellphe import time_series_features
ts_variables = time_series_features(new_features)
```

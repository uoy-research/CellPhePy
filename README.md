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

## Installation

You can install the latest version of CellPhe from
[PyPi](https://pypi.org/project/cellphe/) with:

```
pip install cellphe
```

## Example

An example dataset to demonstrate CellPhe’s capabilities is hosted on [Dryad](https://doi.org/10.5061/dryad.4xgxd25f0) in the archive `example_data.zip` and comprises 3 parts:

-   The time-lapse stills as TIFF images (`05062019_B3_3_imagedata`)
-   Existing pre-extracted features from PhaseFocus Livecyte.
    (`05062019_B3_3_Phase-FullFeatureTable.csv`)
-   Region-of-interest (ROI) boundaries already demarked in ImageJ
    format (`05062019_B3_3_Phase`)

These should be extracted into a suitable location before proceeding
with the rest of the tutorial.

The first step in the CellPhe workflow is to prepare a dataframe containing
metadata identifying the tracked cells across all the frames, along with any
pre-existing attributes. The segmenting and tracking can be performed within
CellPhe, or pre-segmented and tracked data from two widely used software
(PhaseFocus Livecyte & Trackmate) can be directly imported.

### Segmenting and tracking

CellPhe provides 2 functions to segment and track an image sequence:

  - `segment_images`: Segments images using Cellpose
  - `track_images`: Uses the ImageJ plugin TrackMate to track cells between frames without requiring ImageJ to be installed

```python
from cellphe import segment_images, track_images
```

`segment_images` takes 2 arguments: the path to the directory where the images are stored (where the folder `05062019_B3_3_imagedata` was extracted to), and a path to an output folder where the resultant Cellpose masks will be saved.

This can take several minutes depending on the number of images and their resolution.

```python
segment_images("05062019_B3_3_imagedata", "masks")
```

Confirm that the `masks` directory has been created and populated with TIFs containing cell masks.
If it has, then you are ready to track the cells.
`track_images` takes at minimum 3 arguments: the location of the masks created by `segment_images`, the filename to save the output metadata to, and a folder name to save the ROIs in.
Optionally you can also save the ROIs as a zip so they can be easily opened in ImageJ, and change the tracking options - by default the Simple LAP method is employed.

```python
track_images("masks", "tracked.csv", "rois")
```

Confirm that the `tracked.csv` file was created and the `rois` folder has been populated with ROI files.
These outputs can now be loaded into CellPhe.

### Importing pre-segmented and tracked data

Once a metadata file (CSV format) and a folder of ROIs are available, either directly output from external software (PhaseFocus Livecyte or TrackMate in ImageJ), or from within CellPhe as in the previous section, they can be read into CellPhe.
The `import_data` function accepts metadata files from one of these sources and converts it into a standard format.
It takes 3 arguments: the metadata file path, the source, and the minimum number of frames that a cell must be tracked for to be retained in the dataset (optional).

```python
from cellphe import import_data
```

For example, the dataset that was segmented and tracked in the previous section can be imported as:

```python
feature_table = import_data("tracked.csv", "Trackmate_auto")
```

Alternatively, the example below creates the metadata dataframe from the supplied PhaseFocus dataset, only including cells that were tracked for at least 50 frames.

``` python
input_feature_table = "05062019_B3_3_Phase-FullFeatureTable.csv"
feature_table = import_data(input_feature_table, "Phase", 50)
```

If a segmented and tracked dataset is available from a different source then it can still be used in CellPhe provided that it can be loaded into a `pandas.DataFrame` containing:

  - Each row corresponding to a cell tracked in a specific frame
  - A column `FrameID` (integer) denoted the frame number in chronological order
  - A column `CellID` (integer) identifying the cell
  - A column `ROI_filename` (string) denoting the filename (without extension) of the corresponding ROI file, not including the full path

Additional columns providing cell features can be included and will be retained and incorporated into the CellPhe analysis.
The PhaseFocus dataset keeps the volume and sphericity features, for example.

### Generating cell features

In addition to any pre-calculated features, the `cell_features()`
function generates 74 descriptive features for each cell on every frame
using the frame images and pre-generated cell boundaries, based on size,
shape, texture, and the local cell density. The output is a dataframe
comprising the `FrameID`, `CellID`, and `ROI_filename` columns
from the feature table input, the 74 features as columns,
and any additional features that may be present (such as from `import_data()`)
in further columns.

```python
from cellphe import cell_features
```

`cell_features()` takes as arguments the feature table, the folder where ROIs are saved, the folder where the images are, and the framerate.
It expects frames to be named according to the scheme
`<experiment name>-<frameid>.tif`, where `<frameid>` is a 4 digit
zero-padded integer corresponding to the `FrameID` column,
while ROI files are named according to
the `ROI_filename` column.

```python
roi_folder = "05062019_B3_3_Phase"
image_folder = "05062019_B3_3_imagedata"
new_features = cell_features(feature_table, roi_folder, image_folder, framerate=0.0028)
```

### Generating time-series features

The next step is to calculate features that incorporate the time-dimension.
This is done with the `time_series_features` function, which accepts a dataframe with the cell-level features as output earlier from `cell_features`.

```python
from cellphe import time_series_features
```

Variables are calculated from the time series providing both
summary statistics and indicators of time-series behaviour at different
levels of detail obtained via wavelet analysis. 15 summary scores are
calculated for each feature, in addition to the cell trajectory, thereby
resulting in a default output of 1081 features (15x72 + 1). These are
output in the form of a dataframe with the first column being the
`CellID` used previously.

``` python
tsvariables = time_series_features(new_features)
```

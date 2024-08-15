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
-   Existing pre-extracted features
    (`05062019_B3_3_Phase-FullFeatureTable.csv`)
-   Region-of-interest (ROI) boundaries already demarked in ImageJ
    format (`05062019_B3_3_Phase`)

These should be extracted into a suitable location before proceeding
with the rest of the tutorial.

```python
from cellphe import copy_features, extract_features, time_series_features
```

The first step is to prepare a dataframe containing metadata and any
pre-existing attributes. If PhaseFocus Livecyte or Trackmate software
has been used to generate the region-of-interest (ROI) files, then a
helper function is available to create the required metadata format:
`copy_features`. The dataframe format comprises each row corresponding to
a cell tracked in a given frame, indexed by columns `FrameID` and
`CellID` which contain numerical identifiers (NB: `FrameID` must be in
ascending chronological order). The only other required field is
`ROI_filename`, which specifies the filename of the ROI file
corresponding to the frame-cell combination. Any features can be
provided in additional columns, `copyFeatures` returns volume and
sphericity from PhaseFocus software.

The example below creates the metadata dataframe from a PhaseFocus
experimental setup, only including cells that were tracked for at least
50 frames.

``` python
min_frames = 50
input_feature_table = "05062019_B3_3_Phase-FullFeatureTable.csv"
feature_table = copy_features(input_feature_table, min_frames, source="Phase")
```

In addition to any pre-calculated features, the `extract_features()`
function generates 74 descriptive features for each cell on every frame
using the frame images and pre-generated cell boundaries, based on size,
shape, texture, and the local cell density. The output is a dataframe
comprising the `FrameID`, `CellID`, and `ROI_filename` identifying
columns, the 74 features as columns, and any additional features that
may be present (such as from `copy_features()`) in further columns. The
program expects frames to be named according to the scheme
`<experiment name>-<frameid>.tif`, where `<frameid>` is a 4 digit
zero-padded integer corresponding to the `FrameID` column, and located
in the `frame_folder` directory, while ROI files are named according to
the `ROI_filename` column and located in the `roi_folder` directory.

``` python
roi_folder = "05062019_B3_3_Phase"
image_folder = "05062019_B3_3_imagedata"
new_features = extract_features(feature_table, roi_folder, image_folder, framerate=0.0028)
```

Variables are calculated from the time series for any pre-existing
features as well as the output of `extract_features()`, providing both
summary statistics and indicators of time-series behaviour at different
levels of detail obtained via wavelet analysis. 15 summary scores are
calculated for each feature, in addition to the cell trajectory, thereby
resulting in a default output of 1081 features (15x72 + 1). These are
output in the form of a dataframe with the first column being the
`CellID` used previously.

``` python
tsvariables = time_series_features(new_features)
```

==============
Python Package
==============

.. image:: https://zenodo.org/badge/latestdoi/449769672.svg
   :target: https://zenodo.org/badge/latestdoi/449769672
   :alt: DOI

CellPhe provides functions to phenotype cells from time-lapse videos and accompanies the paper:

Wiggins, L., Lord, A., Murphy, K.L. et al.
The CellPhe toolkit for cell phenotyping using time-lapse imaging and pattern recognition.
Nat Commun 14, 1854 (2023).
https://doi.org/10.1038/s41467-023-37447-3

The Python package is a port of the `original R implementation <https://github.com/uoy-research/CellPhe>`_.

Installation
------------

Basic install (phenotyping only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the latest version of CellPhe from `PyPi <https://pypi.org/project/cellphe/>`_ with:

.. code-block:: bash

   pip install cellphe

The default installation provides access to the core phenotyping functionality, but if you would also like to segment and track your images, the full installation will need to be installed as below.

Full install (segmentation + tracking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tracking functionality depends on the ImageJ plugin TrackMate, which in turn depends on having a Java runtime available. We recommend the **Eclipse Temurin** variant.

1. **Download:** Go to the `Adoptium Temurin website <https://adoptium.net/>`_.
2. **Select & Install:**

   * **Windows:** Click the "Latest LTS Release" button. Run the downloaded ``.msi`` file and follow the prompts. (Ensure the "Set ``JAVA_HOME`` variable" option is checked during installation).
   * **macOS:** Download the ``.pkg`` file for your chip type (Apple Silicon or Intel). Open it and follow the installation prompts.
   * **Linux (Debian/Ubuntu):** Open a terminal and run the following:

     .. code-block:: bash

        sudo apt update
        sudo apt install temurin-21-jdk

After installing, verify by running ``java -version`` in a new terminal.

The full version of CellPhe can now be installed:

.. code-block:: bash

   pip install cellphe[full]

Windows issues
--------------

Windows users might encounter some additional OS-specific issues. This section aims to document some of these in order to remove any barriers to use.

Installing Python
~~~~~~~~~~~~~~~~~

Unlike MacOS or Linux, Python doesn't come installed with Windows by default. The best way to install Python is from the `official website <https://www.python.org/downloads/windows/>`_:

* CellPhe isn't compatible with Python 3.14 - we'd recommend either 3.12 or 3.13.
* Check the box for adding Python to ``PATH`` during the install process.
* It's a good idea to also install the separate Install Manager to allow for `quicker switching between Python versions <https://docs.python.org/3/using/windows.html#installing-runtimes>`_ in future.

Visual C++ Build Tools
~~~~~~~~~~~~~~~~~~~~~~

If you come across an error message regarding a missing ``vswhere.exe`` or unknown compilers during installation, the solution is to install C++ Build Tools from the `Microsoft website <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ (tick 'Desktop development with C++').

Pip not found
~~~~~~~~~~~~~

If you get an error that ``pip`` is not recognized, try running:

.. code-block:: bash

   python -m pip install cellphe

This occurs because ``pip`` is not always added to the ``PATH`` in Windows.

Example Usage
-------------

An example dataset is hosted on `Dryad <https://doi.org/10.5061/dryad.4xgxd25f0>`_ in the archive ``example_data.zip``. Extract these into a ``data`` directory before proceeding.

Segmenting and tracking
~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Please ensure that you have installed the full version of CellPhe as shown above before segmenting or tracking.

.. code-block:: python

   from cellphe import segment_images, track_images

   # Segment images using Cellpose
   segment_images("data/05062019_B3_3_imagedata", "data/masks")

   # Track cells using TrackMate
   track_images("data/masks", "data/tracked.csv", "data/rois.zip")

Importing pre-segmented and tracked data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``import_data`` function accepts metadata files from CellPhe, PhaseFocus Livecyte, or TrackMate.

.. code-block:: python

   from cellphe import import_data
   feature_table = import_data("data/tracked.csv", "Trackmate_auto", 50)

Generating cell features
~~~~~~~~~~~~~~~~~~~~~~~~

The ``cell_features()`` function generates 74 descriptive features (size, shape, texture, and density) for each cell on every frame.

.. code-block:: python

   from cellphe import cell_features
   roi_archive = "data/05062019_B3_3_Phase.zip"
   image_folder = "data/05062019_B3_3_imagedata"
   new_features = cell_features(feature_table_phase, roi_archive, image_folder, framerate=0.0028)

Generating time-series features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``time_series_features`` function calculates variables incorporating the time-dimension, including summary statistics and wavelet analysis. This results in a default output of 1081 features.

.. code-block:: python

   from cellphe import time_series_features
   ts_variables = time_series_features(new_features)

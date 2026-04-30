=================
Nextflow Pipeline
=================

This Nextflow pipeline runs a cell timelapse through the full CellPhe pipeline, including:

* Image processing
* Segmentation
* Tracking
* Frame feature extraction
* Time-series feature extraction
* QC report generation

.. mermaid::

    flowchart TB
        subgraph " "
        v0["ome.companion file"]
        v22["Segmentation Config"]
        v31["Tracking Config"]
        end
        v1([ome_get_filename])
        v6([ome_get_frame_t])
        v10([ome_get_global_t])
        v15([split_ome_frames])
        v17([remove_spaces])
        v20([rename_frames])
        v23([save_segmentation_config])
        subgraph " "
        v24[" "]
        v30[" "]
        v33[" "]
        v39[" "]
        v45[" "]
        v48[" "]
        end
        v25([segment_image])
        v29([segmentation_qc])
        v32([save_tracking_config])
        v34([track_images])
        v35([parse_trackmate_xml])
        v36([filter_size_and_observations])
        v38([tracking_qc])
        v40([cellphe_frame_features_image])
        v42([combine_frame_features])
        v43([create_frame_summary_features])
        v44([cellphe_time_series_features])
        v47([create_tiff_stack])
        v2(( ))
        v16(( ))
        v21(( ))
        v26(( ))
        v41(( ))
        v0 --> v1
        v1 --> v2
        v0 --> v6
        v6 --> v2
        v0 --> v10
        v10 --> v2
        v2 --> v15
        v15 --> v16
        v16 --> v17
        v17 --> v16
        v16 --> v20
        v20 --> v21
        v22 --> v23
        v23 --> v24
        v21 --> v25
        v25 --> v26
        v21 --> v29
        v26 --> v29
        v29 --> v30
        v31 --> v32
        v32 --> v33
        v26 --> v34
        v34 --> v35
        v35 --> v40
        v35 --> v36
        v35 --> v38
        v36 --> v38
        v36 --> v40
        v36 --> v43
        v38 --> v39
        v21 --> v40
        v40 --> v41
        v41 --> v42
        v42 --> v43
        v43 --> v44
        v44 --> v45
        v21 --> v47
        v47 --> v48

Nextflow provides several advantages over doing all this in Python through the `CellPhe <https://pypi.org/project/cellphe/>`_ package:

* **Explicit Structure:** Makes the pipeline structure explicit.
* **Modular Design:** Allows for easy extension and modification.
* **Containerization:** Each step is run in a container, facilitating full reproducibility and dependency management.
* **Resumability:** Failed pipelines can be resumed from previously cached steps.
* **HPC Integration:** Integrates seamlessly with High Performance Computing clusters (HPC).

Prerequisites
-------------

Because the actual pipeline steps are run in containers, there is a minimal set of dependencies: **Nextflow** and **Apptainer**.

* **Nextflow:** Install following the `official instructions <https://www.nextflow.io/docs/latest/install.html>`_. Windows users should refer to the `WSL setup guide <https://seqera.io/blog/setup-nextflow-on-windows/>`_.
* **Apptainer:** Used instead of Docker as it does not require elevated access on HPC. Follow the `Apptainer installation guide <https://apptainer.org/docs/admin/main/installation.html>`_.

Pipeline Arguments
------------------

Three things are needed to run the full CellPhe pipeline:

1. A folder containing a timelapse.
2. A parameters file.
3. A location where the outputs can be saved.

Images
~~~~~~

The folder should only contain image files related to the timelapse (TIFF, JPG, or OME.TIFF). Files must be named to provide a natural ordering (e.g., ``image_1.tiff``, ``image_2.tiff``).
**Supported extensions:** .tif, .tiff, .TIF, .TIFF, .jpg, .jpeg, .JPG, .JPEG, .ome.companion.

Parameters File
~~~~~~~~~~~~~~~

The parameters file is a JSON file storing options for every pipeline step.

.. important::
   The only parameter that **must** be changed is the ``folder_names -> timelapse_id`` field.

Key sections include:

* **folder_names:** Controls output directory naming.
* **run:** Boolean flags to enable/disable specific stages (e.g., ``"cellphe": false``).
* **segmentation:** Configures `Cellpose <https://cellpose.readthedocs.io/en/latest/>`_.
* **tracking:** Configures `Trackmate <https://imagej.net/plugins/trackmate/>`_. Supported algorithms: SimpleSparseLAP, SparseLAP, Kalman, AdvancedKalman, NearestNeighbor, Overlap.
* **QC:** Filter cells by ``minimum_cell_size`` or ``minimum_observations``.

Running the Pipeline
--------------------

Once a parameter file is prepared, execute the pipeline:

.. code-block:: bash

   nextflow run uoy-research/cellphe-data-pipeline \
     --raw_dir /path/to/raw/dir \
     --output_dir /path/to/output \
     -params-file /path/to/params.json

Configuration
-------------

Infrastructure properties (resource limits, HPC profiles) are handled via ``.config`` files.

Example: Increasing Trackmate memory in ``custom.config``:

.. code-block:: groovy

   process {
       withName: track_images {
           memory = 16.GB
       }
   }

Run with the custom config:

.. code-block:: bash

   nextflow run uoy-research/cellphe-data-pipeline [...] -c custom.config

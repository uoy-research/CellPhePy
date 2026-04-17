=========
Dashboard
=========

A dashboard for the `CellPhe <https://pypi.org/project/cellphe/>`_ cell phenotyping library.

Usage
-----

The dashboard is run locally so that you are in full control of your images and do not need to upload them anywhere. It is cross-platform, runs in a web browser, and doesn't require any coding experience beyond that required to run the Python package.

Prerequisites
~~~~~~~~~~~~~

The only prerequisite is to install **Docker** (or `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_). Optionally, you can run the dashboard from the source code, although this is more involved.

Running with Docker
~~~~~~~~~~~~~~~~~~~

Simply download the Docker image from the repository:

.. code-block:: bash

   docker pull ghcr.io/uoy-research/cellphe-dashboard:main

Then run it:

.. code-block:: bash

   docker run --rm -p 8501:8501 ghcr.io/uoy-research/cellphe-dashboard:main

Running from source
~~~~~~~~~~~~~~~~~~~

The app can be run without Docker, although it will require setting up a suitable Python environment.

.. note::
   **Currently Python 3.12 is the supported version**; other versions may require tweaking of the dependencies.

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/uoy-research/CellPhe-dashboard.git

2. **Install the Python prerequisites**, ideally in a `new virtual environment <https://docs.python.org/3/library/venv.html>`_:

   .. code-block:: bash

      pip install -r requirements.txt

3. **Run the app**:

   .. code-block:: bash

      streamlit run CellPheDashboard.py --server.port=8501 --server.address=0.0.0.0

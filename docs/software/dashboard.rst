=========
Dashboard
=========

A dashboard for the `CellPhe <https://pypi.org/project/cellphe/>`_ cell phenotyping library.

Usage
-----

The dashboard is run locally so that you are in full control of your images and do not need to upload them anywhere. It is cross-platform, runs in a web browser, and does not require any coding experience beyond that required to run the CellPhe python package.

Prerequisites
~~~~~~~~~~~~~

The only prerequisite is to install **Docker** (or `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ for Windows users). Optionally, you can run the dashboard from the source code, although this is more involved and slower as it requires manually installing the dependencies.

.. note::
   If you are unable to install Docker on your computer due to the elevated permissions it requires, running from source may be necessary.

Running with Docker
~~~~~~~~~~~~~~~~~~~

Simply download the Docker image from this repository by running the following command in a terminal or command prompt:

.. code-block:: bash

   docker pull ghcr.io/uoy-research/cellphe-dashboard:main

Then run it:

.. code-block:: bash

   docker run --rm -p 8501:8501 ghcr.io/uoy-research/cellphe-dashboard:main

You can then access the app by navigating to ``http://127.0.0.1:8501/`` in a web-browser.

Running from source
~~~~~~~~~~~~~~~~~~~

The app can be run without Docker, although it will require installing the dependencies by hand which takes some time.

.. important::
   Currently Python 3.12 is the supported version; other versions may require tweaking the Python dependencies.

The tracking functionality depends on the ImageJ plugin TrackMate, which in turn depends on having a Java runtime available. We recommend the **Eclipse Temurin** variant, which is built on OpenJDK and is free, stable, and widely supported.

1. **Download:** Go to the `Adoptium Temurin website <https://adoptium.net/>`_.
2. **Select & Install:**

   * **Windows:** Click the "Latest LTS Release" button. Run the downloaded ``.msi`` file and follow the prompts. (Ensure the "Set ``JAVA_HOME`` variable" option is checked during installation).
   * **macOS:** Download the ``.pkg`` file for your chip type (Apple Silicon or Intel). Open it and follow the installation prompts.
   * **Linux (Debian/Ubuntu):** Open a terminal and run the following (NB: you can also install Canonical's build from the ``openjdk-21-jdk`` package):

     .. code-block:: bash

        sudo apt update
        sudo apt install temurin-21-jdk

After installing, open a **new** terminal/command prompt and run:

.. code-block:: bash

   java -version

If you see the version details, your Java installation is successful and you are ready to setup the Python environment. Firstly, clone this repository and change into it:

.. code-block:: bash

   git clone https://github.com/uoy-research/CellPhe-dashboard.git
   cd CellPhe-dashboard

Then install the Python prequisites, ideally in a `new virtual environment <https://docs.python.org/3/library/venv.html>`_:

.. code-block:: bash

   pip install -r requirements.txt

Then you can run the app (NB: the first time tracking is run it will take some time to setup an ImageJ instance):

.. code-block:: bash

   streamlit run CellPheDashboard.py --server.port=8501 --server.address=0.0.0.0

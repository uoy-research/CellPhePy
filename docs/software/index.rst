Software
========

The CellPhe ecosystem comprises four distinct software interfaces, each tailored to specific research needs and computational environments. The following section provides a comparative overview of their capabilities; detailed deployment guides and tutorials for each tool are available in their respective sections of this documentation.

.. toctree::
   :maxdepth: 1

   cellphepy
   cellpher
   dashboard
   pipeline

Choosing a product
------------------

The CellPhe R package serves as the original implementation and is ideal for researchers already comfortable within the R ecosystem who only require the core phenotyping functions, such as feature extraction and time-series analysis. However, its primary disadvantages are that it lacks the newer features found in the Python package and does not support integrated segmentation or tracking. For users who require a more comprehensive, "all-in-one" solution with a low-level interface for fine-grained control, the Python package is the more suitable choice. While it offers full functionality, including experimental segmentation and tracking capabilities via cellpose and Trackmate respectively, it requires users to manually manage complex dependencies and navigate a code-based environment, which may be a barrier for those without programming expertise.

For users seeking a balance between power and accessibility, the CellPhe Dashboard provides a user-friendly GUI that runs in a web-browser and offers full functionality without requiring coding experience. An additional advantage is its ease of deployment if Docker is available, making it much faster to set up than the manual Python installation. Conversely, the Nextflow pipeline is designed specifically for power users and High-Performance Computing (HPC) environments. Its strengths lie in its modularity and scalability, allowing for the simultaneous processing of multiple long timelapses and seamless integration with a variety of hardware environments including traditional HPC and cloud-based (it comes with a Slurm configuration). It is the most flexible option, offering more features than are available in either the Dashboard or the Python package, e.g. being able to choose cellpose version, switch between CPU and GPU segmentation seamlessly, and generating automated QC reports. It only has 2 dependencies in Nextflow and Apptainer, but extending will require tweaking the Nextflow definition. It is the best choice for large-scale, reproducible research projects that extend beyond the limitations of a single Python environment.

The software options are summarized in the table below.

.. list-table:: CellPhe Software Comparison
   :widths: 10 10 5 5 5 35
   :header-rows: 1
   :class: tight-table centered-cols

   * - Product
     - Interface
     - Phenotyping
     - Seg + Track
     - Parallel
     - Dependencies
   * - Python package
     - Python
     - ✓
     - ✓
     - ✗
     - Python + scientific stack (+ Torch, Java, Maven for seg/track)
   * - R package
     - R
     - ✓
     - ✗
     - ✗
     - R
   * - Dashboard
     - Web-app
     - ✓
     - ✓
     - ✗
     - Either Docker or Python + Torch + Java + Maven
   * - Pipeline
     - Nextflow / JSON
     - ✓
     - ✓
     - ✓
     - Nextflow + Apptainer

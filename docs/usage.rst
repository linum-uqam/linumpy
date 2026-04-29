Usage
=====

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/linum-uqam/linumpy.git
   cd linumpy

   # Install with uv (recommended)
   uv sync

   # Or with pip in an existing environment
   pip install -e .

For GPU acceleration (CuPy, plus optional PyTorch for BaSiCPy), install the matching CUDA extra:

.. code-block:: bash

   uv sync --extra gpu          # CUDA 12
   uv sync --extra gpu-cuda13   # CUDA 13

See :doc:`GPU_ACCELERATION` for details on the GPU code paths.

Running scripts
---------------

All command-line tools are installed as ``linum_*`` entry points. List
them with:

.. code-block:: bash

   linum_aip --help

For an inventory of every script, see :doc:`scripts` (auto-generated from
the argparse definitions) or :doc:`SCRIPTS_REFERENCE` (curated overview).

Pipelines
---------

For end-to-end processing, use the Nextflow workflows shipped with the
repository — see :doc:`PIPELINE_OVERVIEW` and :doc:`NEXTFLOW_WORKFLOWS`.

Command-line scripts
====================

linumpy ships a large collection of ``linum_*`` command-line tools. The
authoritative inventory lives in :doc:`SCRIPTS_REFERENCE`, which groups
scripts by purpose (preprocessing, mosaic creation, stitching, stacking,
diagnostics, conversion, GPU benchmarks, …).

Each script uses ``argparse`` and prints its full options list with
``--help``. A handful of high-traffic entry points are documented here
inline using :mod:`sphinx-argparse`; the rest can be browsed in the
reference page above or queried directly on the command line.

Mosaic acquisition & preprocessing
----------------------------------

.. argparse::
   :module: scripts.linum_create_mosaic_grid_3d
   :func: _build_arg_parser
   :prog: linum_create_mosaic_grid_3d

.. argparse::
   :module: scripts.linum_compensate_psf_from_model
   :func: _build_arg_parser
   :prog: linum_compensate_psf_from_model

.. argparse::
   :module: scripts.linum_compensate_attenuation
   :func: _build_arg_parser
   :prog: linum_compensate_attenuation

Stitching, stacking & alignment
-------------------------------

.. argparse::
   :module: scripts.linum_register_pairwise
   :func: _build_arg_parser
   :prog: linum_register_pairwise

.. argparse::
   :module: scripts.linum_stack_slices_3d
   :func: _build_arg_parser
   :prog: linum_stack_slices_3d

.. argparse::
   :module: scripts.linum_align_to_ras
   :func: _build_arg_parser
   :prog: linum_align_to_ras

Slice quality & interpolation
-----------------------------

.. argparse::
   :module: scripts.linum_assess_slice_quality
   :func: _build_arg_parser
   :prog: linum_assess_slice_quality

.. argparse::
   :module: scripts.linum_interpolate_missing_slice
   :func: _build_arg_parser
   :prog: linum_interpolate_missing_slice

GPU & diagnostics
-----------------

.. argparse::
   :module: scripts.linum_gpu_info
   :func: _build_arg_parser
   :prog: linum_gpu_info

.. argparse::
   :module: scripts.linum_correct_bias_field
   :func: _build_arg_parser
   :prog: linum_correct_bias_field

.. note::

   To document additional scripts here, add an ``.. argparse::`` block with
   the script's module path and the name of the function that returns its
   ``argparse.ArgumentParser`` (most linumpy scripts expose
   ``_build_arg_parser`` — verify with ``grep -n _build_arg_parser
   scripts/<script>.py``).

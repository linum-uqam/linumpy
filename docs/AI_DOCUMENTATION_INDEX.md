# Documentation Index

> **Note**: This documentation was generated with AI assistance (Claude, Anthropic)

---

## Overview

Complete documentation for the linumpy microscopy processing library. The library provides tools for processing Serial Optical Coherence Tomography (S-OCT) data into reconstructed 3D volumes.

---

## Pipeline Documentation

### Workflow Guides

1. **[PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)** - Complete overview of preprocessing and 3D reconstruction pipelines
2. **[NEXTFLOW_WORKFLOWS.md](NEXTFLOW_WORKFLOWS.md)** - Nextflow workflow configuration and execution guide

### Data Formats

3. **[MOSAIC_GRID_FORMAT.md](MOSAIC_GRID_FORMAT.md)** - OME-Zarr mosaic grid format specification
4. **[SHIFTS_FILE_FORMAT.md](SHIFTS_FILE_FORMAT.md)** - XY shifts CSV file format and usage

---

## Feature Documentation

5. **[SLICE_CONFIG_FEATURE.md](SLICE_CONFIG_FEATURE.md)** - Slice selection and filtering system
6. **[SLICE_INTERPOLATION_FEATURE.md](SLICE_INTERPOLATION_FEATURE.md)** - Missing slice reconstruction using registration-based morphing
7. **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - GPU acceleration using NVIDIA CUDA/CuPy

---

## Reference

8. **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - Command-line scripts reference guide
9. **[LIBRARY_MODULES.md](LIBRARY_MODULES.md)** - Python library module documentation

---

## Contributing

10. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## Source Code Structure

```
linumpy/
├── io/                    # Input/output modules
│   ├── allen.py          # Allen Brain Atlas integration
│   ├── npz.py            # NPZ file handling
│   ├── thorlabs.py       # Thorlabs microscope support
│   └── zarr.py           # OME-Zarr I/O
├── gpu/                   # GPU acceleration
│   ├── __init__.py       # GPU detection & utilities
│   ├── array_ops.py      # Per-pixel operations
│   ├── corrections.py    # Galvo detection
│   ├── fft_ops.py        # FFT & phase correlation
│   ├── interpolation.py  # Resampling & transforms
│   ├── morphology.py     # Binary operations & filtering
│   └── registration.py   # Hybrid GPU/CPU registration
├── microscope/           # Microscope-specific modules
│   └── oct.py            # OCT tile reading
├── preproc/              # Preprocessing modules
│   ├── icorr.py          # Illumination correction
│   └── xyzcorr.py        # XYZ correction
├── psf/                  # Point spread function
│   └── psf_estimator.py  # PSF estimation
├── stitching/            # Image stitching
│   ├── registration.py   # Image registration
│   ├── stitch_utils.py   # Stitching utilities
│   └── topology.py       # Mosaic topology
├── utils/                # Utility modules
│   ├── data_io.py        # Data I/O helpers
│   ├── io.py             # CLI argument helpers
│   ├── metrics.py        # Quality metrics collection
│   └── mosaic_grid.py    # Mosaic grid utilities
├── reconstruction.py     # Core reconstruction
├── segmentation.py       # Segmentation tools
└── utils_images.py       # Image utilities
```

---

## Workflow Files

```
workflows/
├── preproc/
│   ├── nextflow.config       # Preprocessing config
│   └── preproc_rawtiles.nf   # Raw tiles → mosaic grids
└── reconst_3d/
    ├── nextflow.config       # Reconstruction config
    └── soct_3d_reconst.nf    # Mosaic grids → 3D volume
```

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
3. **[RECONST_2_5D_WORKFLOW.md](RECONST_2_5D_WORKFLOW.md)** - Legacy 2.5D reconstruction workflow (TIFF mosaic grids)

### Data Formats

4. **[MOSAIC_GRID_FORMAT.md](MOSAIC_GRID_FORMAT.md)** - OME-Zarr mosaic grid format specification
5. **[SHIFTS_FILE_FORMAT.md](SHIFTS_FILE_FORMAT.md)** - XY shifts CSV file format and usage

---

## Feature Documentation

6. **[SLICE_CONFIG_FEATURE.md](SLICE_CONFIG_FEATURE.md)** - Slice selection and filtering system
7. **[SLICE_INTERPOLATION_FEATURE.md](SLICE_INTERPOLATION_FEATURE.md)** - Missing slice reconstruction using registration-based morphing
8. **[GPU_ACCELERATION.md](GPU_ACCELERATION.md)** - GPU acceleration using NVIDIA CUDA/CuPy

---

## Reference

9. **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - Command-line scripts reference guide
10. **[LIBRARY_MODULES.md](LIBRARY_MODULES.md)** - Python library module documentation
11. **[RECONSTRUCTION_DIAGNOSTICS.md](RECONSTRUCTION_DIAGNOSTICS.md)** - Diagnostic tools for troubleshooting reconstruction artifacts
12. **[PIPELINE_PERFORMANCE_ANALYSIS.md](PIPELINE_PERFORMANCE_ANALYSIS.md)** - Pipeline performance benchmarks and optimization guide

---

## Contributing

13. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## Source Code Structure

```
linumpy/
├── _thread_config.py      # CPU thread management (NumPy/SciPy/Dask/JAX/ITK)
├── io/                    # Input/output modules
│   ├── allen.py           # Allen Brain Atlas integration
│   ├── data_io.py         # Legacy slicer / NIfTI readers
│   ├── npz.py             # NPZ file handling
│   ├── slice_config.py    # slice_config.csv read/write/stamp helpers
│   ├── test_data.py       # Test dataset access
│   ├── thorlabs.py        # Thorlabs microscope support
│   └── zarr.py            # OME-Zarr I/O
├── gpu/                   # GPU acceleration
│   ├── __init__.py        # GPU detection & utilities
│   ├── array_ops.py       # Per-pixel operations
│   ├── corrections.py     # Galvo detection
│   ├── cuda_env.py        # CUDA environment utilities
│   ├── fft_ops.py         # FFT & phase correlation
│   ├── image_quality.py   # GPU image quality assessment
│   ├── interpolation.py   # Resampling & transforms
│   ├── morphology.py      # Binary operations & filtering
│   └── registration.py    # Hybrid GPU/CPU registration
├── microscope/            # Microscope-specific modules
│   └── oct.py             # OCT tile reading
├── preproc/               # Preprocessing modules
│   ├── icorr.py           # Illumination correction
│   ├── normalization.py   # Intensity normalization
│   ├── resampling.py      # Mosaic grid resampling utilities
│   └── xyzcorr.py         # XYZ correction & galvo shift detection
├── psf/                   # Point spread function
│   └── psf_estimator.py   # PSF estimation
├── shifts/                # XY shift utilities (cumulative shifts, unit detection)
│   └── utils.py
├── stitching/             # Image stitching
│   ├── FileUtils.py       # File handling utilities
│   ├── interpolation.py   # Missing-slice interpolation (zmorph)
│   ├── manual_registration.py  # GUI-based manual registration
│   ├── mosaic_grid.py     # MosaicGrid class + diffusion blending
│   ├── motor.py           # Motor-position-based tile placement
│   ├── registration.py    # Image registration
│   ├── stacking.py        # 3D slice stacking utilities
│   ├── stitch_utils.py    # Stitching utilities
│   └── topology.py        # Mosaic topology
├── utils/                 # Utility modules
│   ├── image_quality.py   # Image quality assessment (SSIM, edge, variance)
│   ├── io.py              # CLI argument helpers
│   ├── metrics.py         # Quality metrics collection
│   ├── orientation.py     # Volume orientation codes & RAS transforms
│   └── visualization.py   # Orthogonal view screenshots
├── reconstruction.py      # Core reconstruction
├── segmentation.py        # Segmentation tools
└── utils_images.py        # Image utilities
```

---

## Workflow Files

```
workflows/
├── preproc/
│   ├── nextflow.config       # Preprocessing config
│   └── preproc_rawtiles.nf   # Raw tiles → mosaic grids
├── reconst_3d/
│   ├── diagnostics.nf        # Optional diagnostic processes
│   ├── nextflow.config       # 3D reconstruction config
│   └── soct_3d_reconst.nf    # Mosaic grids → 3D volume
└── reconst_2.5d/
    ├── soct_2.5d_reconst.nf              # 2.5D reconstruction workflow
    ├── soct_2.5d_reconst_beluga.config   # Beluga HPC cluster config
    └── soct_2.5d_reconst_docker.config   # Docker container config
```

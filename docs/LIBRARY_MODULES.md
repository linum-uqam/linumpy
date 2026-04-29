# Library Modules

A concise map of the `linumpy` Python package. Use this page to find which
subpackage owns what, then follow the auto-generated [API
reference](api/linumpy/index) for full signatures and docstrings.

## Package layout

```
linumpy/
├── cli/             # argparse helpers shared by linum_* scripts
├── config/          # thread/process limits (BLAS, OpenMP, threadpoolctl)
├── geometry/        # crop, galvo correction, interface detection, resampling
├── gpu/             # CuPy-backed FFT, morphology, B-spline, N4, registration
├── imaging/         # orientation, overlays, simple transforms, visualization
├── intensity/       # attenuation, bias-field, normalization, PSF model,
│                    #   vignette, intensity conversion
├── io/              # OME-Zarr I/O, slice-config, ThorLabs raw, test data
├── metrics/         # PipelineMetrics + per-step metric collectors
├── microscope/      # OCT-specific helpers
├── mosaic/          # mosaic-grid layout, motor positions, stacking,
│                    #   interpolation, blending, discovery
├── psf/             # PSF extraction and synthetic generation
├── reference/       # Allen brain atlas helpers
├── registration/    # phase correlation, SimpleITK, manual, refinement
├── segmentation/    # brain tissue segmentation
└── stack_alignment/ # shifts CSV I/O, unit detection, filtering
```

## Subpackage cheat sheet

### `linumpy.io`

OME-Zarr is the canonical on-disk format. Two writers are provided:

* `OmeZarrWriter` — generic chunked writer with power-of-2 pyramid levels.
* `AnalysisOmeZarrWriter` — writes analysis-friendly resolution levels
  (10/25/50/100 µm) for the final 3D volume.

Common entry points:

```python
from linumpy.io.zarr import read_omezarr, save_omezarr, OmeZarrWriter, AnalysisOmeZarrWriter
from linumpy.io.test_data import get_data
from linumpy.io import slice_config        # slice_config.json reader/writer
from linumpy.io.thorlabs import ThorImageOCT
```

`read_omezarr(path, level=0)` returns a `(zarr.Array, voxel_size)` tuple. The
voxel size is ordered to match the array axes (Z, Y, X for 3D volumes;
Y, X for 2D mosaics).

See [Mosaic Grid Format](MOSAIC_GRID_FORMAT.md) and
[Slice Config Feature](SLICE_CONFIG_FEATURE.md) for format details.

### `linumpy.cli` and `linumpy.config`

`linumpy.cli.args` provides `add_overwrite_arg`, `add_processes_arg`,
`parse_processes_arg`, `assert_output_exists`, and `get_available_cpus`
(honours `LINUMPY_MAX_CPUS` / `LINUMPY_RESERVED_CPUS` env vars).

`linumpy.config.threads` configures BLAS/OpenMP/threadpoolctl thread caps and
is imported as a side-effect at the top of every `linum_*` script (before
NumPy/SciPy).

### `linumpy.metrics`

Structured metrics collected at every pipeline stage and aggregated into
`PipelineMetrics`. Per-step collectors live alongside the dataclass.

```python
from linumpy.metrics import (
    PipelineMetrics,
    collect_normalization_metrics,
    collect_pairwise_registration_metrics,
    aggregate_metrics,
    compute_summary_statistics,
)
```

See [Reconstruction Diagnostics](RECONSTRUCTION_DIAGNOSTICS.md).

### `linumpy.geometry`

Pixel/world geometry, plus OCT-specific corrections:

* `geometry.galvo` — `detect_galvo_shift`, `fix_galvo_shift`,
  `detect_galvo_for_slice` (B-scan galvo artifact correction).
* `geometry.interface` — tissue-surface detection.
* `geometry.crop` — crop volumes around the tissue interface.
* `geometry.resampling` — resample mosaic grids and 3D volumes to a target
  isotropic resolution.

### `linumpy.intensity`

Intensity-domain corrections used during preprocessing.

* `intensity.attenuation` — depth-dependent attenuation modelling.
* `intensity.bias_field` — N4 bias-field correction (CPU SimpleITK and the
  GPU port; `n4_correct`, `n4_correct_per_section`, `compute_tissue_mask`).
* `intensity.normalization` — per-slice histogram matching, Z-profile
  smoothing, agarose flattening.
* `intensity.psf_model` — `estimate_psf` and PSF-based deconvolution.
* `intensity.vignette` — vignette/illumination flat-field correction.

### `linumpy.mosaic`

Everything that turns a folder of tiles into a stacked 3D volume.

* `mosaic.grid` — `MosaicGrid` (pixel-space tile layout).
* `mosaic.motor` — motor-position handling (`compute_motor_positions`).
* `mosaic.stacking` — `find_z_overlap`, slab assembly utilities.
* `mosaic.interpolation` — `interpolate_z_morph`, `interpolate_weighted`.
* `mosaic.discovery` — discover tiles on disk.
* `mosaic.overlap` — overlap-region utilities.
* `mosaic.quick_stitch` — fast diagnostic stitching.

### `linumpy.registration`

* `registration.phase_correlation` — masked phase correlation primitives.
* `registration.sitk` — `register_2d_images_sitk`, `apply_transform`.
* `registration.refinement` — `find_best_z`, `register_refinement`,
  `gradient_magnitude_alignment`, `centre_of_mass_offset`.
* `registration.transforms` — transform composition / decomposition / I/O.
* `registration.manual` — manual landmark registration GUI.

### `linumpy.gpu`

CuPy-backed versions of hot paths. Each public entry point either takes a
`backend="cpu"|"gpu"|"auto"` flag or has an explicit `_gpu` suffix.

* `gpu.fft_ops`, `gpu.array_ops`, `gpu.morphology`, `gpu.interpolation`
  — FFT, element-wise ops, morphology, interpolation primitives.
* `gpu.bias_field`, `gpu.n4`, `gpu.bspline` — GPU N4 and B-spline grid.
* `gpu.image_quality` — GPU image-quality metrics.
* `gpu.registration`, `gpu.corrections` — GPU registration and correction
  passes used by `linum_estimate_transform.py`,
  `linum_normalize_intensities_per_slice.py`, etc.

See [GPU Acceleration](GPU_ACCELERATION.md) and [N4 GPU](N4_GPU.md).

### `linumpy.stack_alignment`

Shifts CSV utilities used to align serial sections.

```python
from linumpy.stack_alignment.io import load_shifts_csv, write_shifts_csv
from linumpy.stack_alignment.units import detect_shift_units
from linumpy.stack_alignment.filter import filter_outliers
```

See [Shifts File Format](SHIFTS_FILE_FORMAT.md).

### `linumpy.imaging`

Convenience helpers for figures and quick previews.

* `imaging.orientation` — anatomical-axis labelling.
* `imaging.overlay` — overlay generation for diagnostics.
* `imaging.transform` — 2D affine helpers.
* `imaging.visualization` — matplotlib panels used by diagnostic scripts.

### `linumpy.psf`

* `psf.extract` — extract PSF estimates from acquisitions.
* `psf.synthetic` — synthetic PSF generators for tests.

(Higher-level PSF model lives in `linumpy.intensity.psf_model`.)

### `linumpy.reference`

* `reference.allen` — download the Allen mouse atlas template, register a
  3D volume to it, and produce RAS-aligned templates.

### `linumpy.segmentation`

* `segmentation.brain` — tissue / background segmentation in 3D OCT.

### `linumpy.microscope`

* `microscope.oct.OCT` — OCT acquisition metadata wrapper.

## How the pieces fit together

A typical pipeline run wires these subpackages as follows:

1. **Discovery** (`mosaic.discovery`) finds raw tiles on disk and reads
   ThorLabs metadata via `io.thorlabs`.
2. **Tile preprocessing** uses `geometry.galvo`, `intensity.vignette`,
   `intensity.attenuation`, and `intensity.normalization`.
3. **Mosaic assembly** builds 2D AIPs (`mosaic.grid`) and 3D mosaic grids
   per slice, then writes OME-Zarr via `io.zarr`.
4. **Stitching** uses `registration.phase_correlation` (CPU/GPU) plus
   `stack_alignment` to compute and store shifts.
5. **Stacking** matches consecutive slices with `registration.refinement`
   and assembles the volume with `mosaic.stacking`.
6. **Bias-field & global polish** run `intensity.bias_field` (with the
   `gpu.n4` backend when CUDA is available) and `imaging.orientation` / RAS
   alignment via `reference.allen`.
7. **Diagnostics** use `metrics` collectors throughout; see
   [Reconstruction Diagnostics](RECONSTRUCTION_DIAGNOSTICS.md).

## See also

* Auto-generated API reference: [api/linumpy](api/linumpy/index.rst)
* [Pipeline Overview](PIPELINE_OVERVIEW.md)
* [Scripts Reference](SCRIPTS_REFERENCE.md)
* [Nextflow Workflows](NEXTFLOW_WORKFLOWS.md)

# Library Modules Documentation


---

## Overview

The linumpy library provides Python modules for microscopy data processing. This document describes the main modules and their functionality.

---

## Module Structure

```
linumpy/
├── __init__.py
├── io/                    # Input/Output modules
├── microscope/            # Microscope-specific modules
├── preproc/               # Preprocessing modules
├── psf/                   # Point spread function
├── stitching/             # Image stitching
├── utils/                 # Utility modules
├── reconstruction.py      # Core reconstruction
├── segmentation.py        # Segmentation tools
└── utils_images.py        # Image utilities
```

---

## I/O Modules (`linumpy.io`)

### zarr.py - OME-Zarr I/O

Read and write OME-Zarr format files.

```python
from linumpy.io.zarr import read_omezarr, save_omezarr, OmeZarrWriter, AnalysisOmeZarrWriter

# Read OME-Zarr
image, resolution = read_omezarr("input.ome.zarr")
# image: dask.array.Array
# resolution: tuple (z, x, y) in mm/pixel

# Save OME-Zarr
save_omezarr(image, "output.ome.zarr", resolution, chunks=(10, 128, 128))

# Write large volumes incrementally (traditional power-of-2 pyramid)
writer = OmeZarrWriter("output.ome.zarr", shape, chunks, dtype)
writer[0:10] = data_slice
writer.finalize(resolution, n_levels=5)  # 2x downsampling per level

# Write with analysis-optimized resolutions (10, 25, 50, 100 µm)
writer = AnalysisOmeZarrWriter("output.ome.zarr", shape, chunks, dtype)
writer[0:10] = data_slice
writer.finalize(resolution, [10, 25, 50, 100])  # or use default
writer.finalize(resolution)  # defaults to [10, 25, 50, 100] µm
```

### allen.py - Allen Brain Atlas

Download and access Allen Brain Atlas data.

```python
from linumpy.io.allen import download_allen_atlas

# Download atlas
download_allen_atlas(output_dir)
```

### npz.py - NPZ Files

Handle NumPy compressed archives.

```python
from linumpy.io.npz import load_npz, save_npz

data = load_npz("data.npz")
save_npz("output.npz", array)
```

### thorlabs.py - Thorlabs Files

Read Thorlabs microscope data formats.

```python
from linumpy.io.thorlabs import read_thorlabs

data = read_thorlabs("thorlabs_file")
```

### test_data.py - Test Data

Access test datasets for development.

```python
from linumpy.io.test_data import get_data

# Get path to test data
raw_tiles_path = get_data('raw_tiles')
```

---

## Microscope Module (`linumpy.microscope`)

### oct.py - OCT Tile Reading

Read raw OCT tiles with metadata and optional corrections.

```python
from linumpy.microscope.oct import OCT

# Open tile
tile = OCT("/path/to/tile_x00_y00_z00")

# Access properties
shape = tile.shape           # (z, x, y)
resolution = tile.resolution # (res_z, res_x, res_y)
dimension = tile.dimension   # Physical dimensions
position = tile.position     # Stage position (if available)

# Read data with corrections
data = tile.load_image(
    crop=True,              # Crop galvo return region
    fix_galvo_shift=True,   # Auto-detect and fix galvo artifacts
    fix_camera_shift=False  # Fix camera timing shift (old data)
)
```

**Properties:**
- `shape`: Data shape (z, x, y)
- `resolution`: Pixel size in mm
- `dimension`: Physical dimensions in mm
- `position`: Stage position (x, y, z) in mm
- `position_available`: Whether position metadata exists

**load_image() Parameters:**
- `crop`: Remove extra pixels from galvo return
- `fix_galvo_shift`: 
  - `True`: Auto-detect artifact and fix if confident (≥0.3 confidence)
  - `int`: Apply specific shift value
  - `False`: No correction
- `fix_camera_shift`: Correct camera timing offset (legacy data)

---

## Preprocessing Modules (`linumpy.preproc`)

### normalization.py - Intensity Normalization

Normalize OCT volume intensities based on agarose background.

```python
from linumpy.preproc.normalization import normalize_volume

# Normalize volume intensities
# agarose_mask: 2D binary mask indicating agarose regions
normalized, background_thresholds = normalize_volume(
    vol,                    # Input volume (Z, X, Y)
    agarose_mask,           # 2D mask for agarose detection
    percentile_max=99.9     # Clip values above this percentile
)
```

### icorr.py - Illumination Correction

Correct illumination inhomogeneity.

```python
from linumpy.preproc.icorr import estimate_illumination, apply_illumination_correction

# Estimate illumination profile
profile = estimate_illumination(image)

# Apply correction
corrected = apply_illumination_correction(image, profile)
```

### xyzcorr.py - XYZ Corrections

Apply spatial corrections including galvo shift detection and correction.

```python
from linumpy.preproc.xyzcorr import (
    detect_galvo_shift,
    detect_galvo_for_slice,
    detect_galvo_artifact_presence,
    fix_galvo_shift,
    findTissueInterface,
    cropVolume
)

# Detect galvo shift with confidence score
aip = volume.mean(axis=0)  # Average intensity projection
shift, confidence = detect_galvo_shift(
    aip, 
    n_pixel_return=40,      # Number of pixels in galvo return region
    return_confidence=True  # Return confidence score (0-1)
)

# Only apply fix if confident (artifact is present)
if confidence >= 0.3:
    corrected = fix_galvo_shift(volume, shift=shift, axis=1)

# Or apply with known shift value
corrected = fix_galvo_shift(volume, shift=15, axis=1)

# For slice-level detection (samples multiple tiles, skips background)
shift, confidence = detect_galvo_for_slice(
    tiles,                  # zarr array of tiles
    n_extra=40,             # Number of extra pixels from galvo return
    threshold=0.6,          # Minimum confidence threshold
    n_samples=5,            # Number of tiles to sample
    min_intensity=20.0      # Skip tiles with mean intensity below this
)

# For batch processing: check artifact presence separately
presence_score = detect_galvo_artifact_presence(
    aip,
    n_pixel_return=40,
    detected_shift=shift
)
```

**Galvo Shift Detection:**

The galvo mirror in OCT systems can cause horizontal banding artifacts when the galvo return region is not at the edge of the raw tile data. The detection system has three parts:

1. **`detect_galvo_shift()`** - Finds *where* the galvo return boundary is located
   - Analyzes average A-line intensity profile
   - Searches for the shift that maximizes boundary discontinuities
   - Returns shift value (in pixels) and optional confidence score

2. **`detect_galvo_for_slice(tiles, n_extra, ...)`** - Slice-level detection
   - Samples tiles from the center of the mosaic (more likely to contain tissue)
   - Skips background tiles with low mean intensity
   - Returns the detection with highest confidence
   - Returns `(0, confidence)` if no artifact detected above threshold

**Confidence Score Interpretation (0-1):**
| Score | Meaning | Action |
|-------|---------|--------|
| < 0.5 | No clear artifact detected | Skip correction |
| ≥ 0.5 | Galvo artifact likely present | Apply correction |
| > 0.7 | Clear galvo artifact | High confidence |

**Key Algorithm Details:**
- Uses **gradient-based detection** to find intensity discontinuities
- Finds pairs of high gradients separated by exactly `n_pixel_return` pixels
- Checks B-scan subregions for subtle artifacts only visible in parts of the tile
- Validates using peak dominance, boundary gradient ranking, and intensity contrast
- Default threshold: 0.6 (configurable via `galvo_threshold` parameter)

---

## PSF Module (`linumpy.psf`)

### psf_estimator.py - PSF Estimation

Estimate and model point spread functions.

```python
from linumpy.psf.psf_estimator import estimate_psf, apply_deconvolution

# Estimate PSF from data
psf = estimate_psf(image)

# Apply deconvolution
deconvolved = apply_deconvolution(image, psf)
```

---

## Stitching Module (`linumpy.stitching`)

### registration.py - Image Registration

Register images using various methods.

```python
from linumpy.stitching.registration import (
    register_2d_images_sitk,
    apply_transform,
    pairWisePhaseCorrelation
)

# Get initial translation estimate using phase correlation
deltas = pairWisePhaseCorrelation(fixed_image, moving_image)
initial_translation = (deltas[1], deltas[0])  # (x, y) order for SimpleITK

# Register 2D images with phase correlation initialization
transform, moving_registered, error = register_2d_images_sitk(
    fixed_image, 
    moving_image,
    metric='MSE',              # MSE, CC, AntsCC, MI
    method='affine',           # affine, euler, translation
    max_iterations=2500,
    grad_mag_tol=1e-6,
    moving_mask=None,
    fixed_mask=None,
    return_3d_transform=True,
    initial_translation=initial_translation,  # Optional: phase correlation result
    initial_step=None          # Optional: optimizer step size (auto-reduced with initial_translation)
)

# Apply transform to volume
transformed = apply_transform(volume, transform)
```

**Note:** When `initial_translation` is provided, the optimizer uses a smaller step size (1.0 vs 4.0 pixels) to prevent drifting away from the correct solution.

### stitch_utils.py - Stitching Utilities

Helper functions for stitching operations.

```python
from linumpy.stitching.stitch_utils import (
    compute_overlap,
    blend_images
)
```

### topology.py - Mosaic Topology

Manage tile arrangements and topology.

```python
from linumpy.stitching.topology import (
    build_topology_graph,
    find_optimal_path
)
```

### manual_registration.py - Manual Registration

GUI-based manual registration tools.

```python
from linumpy.stitching.manual_registration import ManualRegistrationGUI
```

### FileUtils.py - File Utilities

File handling utilities for stitching.

```python
from linumpy.stitching.FileUtils import (
    list_tiles,
    parse_tile_name
)
```

---

## Utilities Module (`linumpy.utils`)

### mosaic_grid.py - Mosaic Grid Utilities

Functions for working with mosaic grids.

```python
from linumpy.utils.mosaic_grid import (
    getDiffusionBlendingWeights,
    compute_grid_shape
)

# Compute blending weights for slice fusion
weights = getDiffusionBlendingWeights(
    mask_fixed,
    mask_moving,
    factor=2
)
```

### io.py - I/O Utilities

Command-line argument helpers.

```python
from linumpy.utils.io import (
    add_overwrite_arg,
    add_processes_arg,
    assert_output_exists,
    get_available_cpus,
    parse_processes_arg
)

# Add standard arguments to parser
parser = argparse.ArgumentParser()
add_overwrite_arg(parser)
add_processes_arg(parser)

# Parse and validate
args = parser.parse_args()
n_processes = parse_processes_arg(args.n_processes)
assert_output_exists(args.output, parser, args)

# Get available CPUs (respects LINUMPY_MAX_CPUS and LINUMPY_RESERVED_CPUS env vars)
available = get_available_cpus()
```

#### CPU Core Management

The `get_available_cpus()` function respects environment variables for limiting CPU usage:

```python
import os
from linumpy.utils.io import get_available_cpus

# Default: uses all CPUs minus 1
cpus = get_available_cpus()  # e.g., 15 on a 16-core system

# With LINUMPY_RESERVED_CPUS=4
os.environ['LINUMPY_RESERVED_CPUS'] = '4'
cpus = get_available_cpus()  # e.g., 12 on a 16-core system

# With LINUMPY_MAX_CPUS=8 (takes precedence)
os.environ['LINUMPY_MAX_CPUS'] = '8'
cpus = get_available_cpus()  # 8 (or total if less than 8)
```

### data_io.py - Data I/O Helpers

Data reading/writing utilities.

```python
from linumpy.utils.data_io import (
    load_image,
    save_image
)
```

### metrics.py - Pipeline Quality Metrics

Collect, save, and aggregate quality metrics from pipeline steps.

```python
from linumpy.utils.metrics import (
    PipelineMetrics,
    collect_mask_metrics,
    collect_normalization_metrics,
    collect_xy_transform_metrics,
    collect_pairwise_registration_metrics,
    collect_interface_crop_metrics,
    collect_psf_compensation_metrics,
    collect_stack_metrics,
    collect_stitch_3d_metrics,
    aggregate_metrics,
    compute_summary_statistics
)

# Manual metrics collection
metrics = PipelineMetrics('my_step', output_dir)
metrics.add_info('input', input_path, 'Input file path')
metrics.add_metric('error', 0.05, unit='pixels', threshold_name='registration_error')
metrics.save()

# Use step-specific collectors (recommended - simpler)
collect_mask_metrics(mask, input_vol, output_path, input_path, params={'sigma': 5.0})
collect_normalization_metrics(vol, agarose_mask, otsu_thresh, bg_thresh, output_path)
collect_pairwise_registration_metrics(error, tx, ty, rot, best_z, expected_z, output_path)

# Aggregate metrics from pipeline run
aggregated = aggregate_metrics('/path/to/pipeline/output')
# Returns: {'step_name': [list of metrics dicts], ...}

# Compute summary statistics
summary = compute_summary_statistics(aggregated['pairwise_registration'])
# Returns: {'count': N, 'status_counts': {...}, 'metric_name': {'mean': ..., 'std': ...}}
```

**Available Threshold Names:**
| Name | Warning | Error | Higher is Better |
|------|---------|-------|------------------|
| `registration_error` | 0.05 | 0.15 | No |
| `translation_magnitude` | 30.0 | 50.0 | No |
| `rotation_degrees` | 1.0 | 2.0 | No |
| `correlation` | 0.7 | 0.5 | Yes |
| `mask_coverage` | 0.05 | 0.01 | Yes |
| `agarose_coverage` | 0.05 | 0.01 | Yes |
| `rms_residual` | 5.0 | 15.0 | No |
| `z_offset_std` | 10.0 | 25.0 | No |
| `z_offset_range` | 15.0 | 30.0 | No |

---

## Core Modules

### reconstruction.py - Core Reconstruction

Core reconstruction functions.

```python
from linumpy.reconstruction import (
    get_tiles_ids,
    get_mosaic_info,
    getLargestCC
)

# Get tile IDs from directory
tiles, tile_ids = get_tiles_ids(directory, z=None)
# tiles: list of Path objects
# tile_ids: list of (mx, my, mz) tuples

# Get mosaic information
mosaic_info = get_mosaic_info(
    directory, 
    z=0, 
    overlap_fraction=0.2,
    use_stage_positions=True
)
# Returns dict with:
# - mosaic_shape, tile_positions, etc.
# - mosaic_xmin_mm, mosaic_ymin_mm
# - tile_resolution

# Get largest connected component
largest_cc = getLargestCC(binary_mask)
```

### segmentation.py - Segmentation

Image segmentation tools.

```python
from linumpy.segmentation import (
    segment_tissue,
    create_mask
)
```

### utils_images.py - Image Utilities

General image processing utilities.

```python
from linumpy.utils_images import (
    apply_xy_shift,
    normalize_image
)

# Apply XY shift to image
shifted = apply_xy_shift(
    image,       # Source image
    reference,   # Reference (determines output shape)
    dy,          # Y shift in pixels
    dx           # X shift in pixels
)
```

---

## Common Patterns

### Reading and Processing a Mosaic Grid

```python
from linumpy.io.zarr import read_omezarr, save_omezarr
import numpy as np

# Read
image, resolution = read_omezarr("input.ome.zarr")

# Process (example: normalize)
data = image[:]  # Load into memory
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Save
import dask.array as da
save_omezarr(da.from_array(data), "output.ome.zarr", resolution)
```

### Parallel Processing with Multiple Tiles

```python
from linumpy.reconstruction import get_tiles_ids
from linumpy.microscope.oct import OCT
from tqdm.contrib.concurrent import process_map

def process_tile(tile_path):
    tile = OCT(tile_path)
    # Process tile...
    return result

tiles, tile_ids = get_tiles_ids(directory)
results = process_map(process_tile, tiles, max_workers=8)
```

### Registration Workflow

```python
from linumpy.io.zarr import read_omezarr
from linumpy.stitching.registration import register_2d_images_sitk, apply_transform

# Load images
fixed, _ = read_omezarr("fixed.ome.zarr")
moving, _ = read_omezarr("moving.ome.zarr")

# Register
transform, _, error = register_2d_images_sitk(
    fixed[0],  # 2D slice
    moving[0],
    method='affine',
    metric='MSE'
)

# Apply to full volume
registered = apply_transform(moving, transform)
```

---

## Type Hints

Most functions include type hints for better IDE support:

```python
def read_omezarr(path: str | Path) -> tuple[da.Array, tuple[float, ...]]:
    ...

def apply_xy_shift(
    image: np.ndarray,
    reference: np.ndarray,
    dy: float,
    dx: float
) -> np.ndarray:
    ...
```

---

## Error Handling

Functions typically raise standard Python exceptions:

```python
try:
    image, res = read_omezarr("nonexistent.ome.zarr")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid format: {e}")
```

---

## Dependencies

Key dependencies used by the library:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `dask` | Lazy/parallel arrays |
| `zarr` | Chunked array storage |
| `SimpleITK` | Image registration |
| `scikit-image` | Image processing |
| `scipy` | Scientific computing |
| `pandas` | Data manipulation |
| `tqdm` | Progress bars |

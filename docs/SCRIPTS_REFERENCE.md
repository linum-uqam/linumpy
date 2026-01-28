# Scripts Reference


---

## Overview

linumpy provides a comprehensive set of command-line scripts for microscopy data processing. All scripts follow a consistent interface with `--help` for usage information.

---

## Script Categories

1. [Data Conversion](#data-conversion)
2. [Mosaic Grid Operations](#mosaic-grid-operations)
3. [Preprocessing](#preprocessing)
4. [Reconstruction](#reconstruction)
5. [Registration & Stitching](#registration--stitching)
6. [Visualization](#visualization)
7. [Utilities](#utilities)
8. [GPU-Accelerated Scripts](#gpu-accelerated-scripts)

---

## Data Conversion

### linum_convert_tiff_to_nifti.py

Convert TIFF stack to NIfTI format.

```bash
linum_convert_tiff_to_nifti.py <input.tiff> <output.nii.gz>
```

### linum_convert_tiff_to_omezarr.py

Convert TIFF to OME-Zarr format.

```bash
linum_convert_tiff_to_omezarr.py <input.tiff> <output.ome.zarr>
```

### linum_convert_nifti_to_nrrd.py

Convert NIfTI to NRRD format.

```bash
linum_convert_nifti_to_nrrd.py <input.nii.gz> <output.nrrd>
```

### linum_convert_nifti_to_zarr.py

Convert NIfTI to Zarr format.

```bash
linum_convert_nifti_to_zarr.py <input.nii.gz> <output.zarr>
```

### linum_convert_omezarr_to_nifti.py

Convert OME-Zarr to NIfTI format.

```bash
linum_convert_omezarr_to_nifti.py <input.ome.zarr> <output.nii.gz>
```

### linum_convert_zarr_to_omezarr.py

Convert Zarr to OME-Zarr format.

```bash
linum_convert_zarr_to_omezarr.py <input.zarr> <output.ome.zarr>
```

### linum_convert_bin_to_nii.py

Convert binary file to NIfTI.

```bash
linum_convert_bin_to_nii.py <input.bin> <output.nii.gz>
```

---

## Mosaic Grid Operations

### linum_create_mosaic_grid_2d.py

Create 2D mosaic grid from tiles.

```bash
linum_create_mosaic_grid_2d.py <output.ome.zarr> --from_tiles_list <tiles...>
```

### linum_create_mosaic_grid_3d.py

Create 3D mosaic grid from raw OCT tiles.

```bash
linum_create_mosaic_grid_3d.py <output.ome.zarr> \
    --from_tiles_list <tiles...> \
    --resolution <res> \
    --n_processes <n> \
    --axial_resolution <axial_res> \
    --sharding_factor <factor>
```

**Options:**
- `--resolution`: Output resolution in µm/pixel (-1 for full)
- `--n_processes`: Number of parallel processes
- `--axial_resolution`: Axial resolution in µm
- `--sharding_factor`: Zarr sharding factor
- `--fix_galvo_shift`: Correct galvo shifts
- `--fix_camera_shift`: Correct camera shifts

### linum_create_all_mosaic_grids_2d.py

Create all 2D mosaic grids from a tiles directory.

```bash
linum_create_all_mosaic_grids_2d.py <tiles_dir> <output_dir>
```

### linum_resample_mosaic_grid.py

Resample mosaic grid to different resolution.

```bash
linum_resample_mosaic_grid.py <input.ome.zarr> <output.ome.zarr> -r <resolution>
```

### linum_resample_mosaic_grid_gpu.py

GPU-accelerated version of mosaic grid resampling (5-12x speedup).

```bash
linum_resample_mosaic_grid_gpu.py <input.ome.zarr> <output.ome.zarr> -r <resolution> --use_gpu
```

Falls back to CPU if GPU is not available.

---

## Preprocessing

### linum_fix_illumination_3d.py

Correct illumination inhomogeneity using BaSiC algorithm.

```bash
linum_fix_illumination_3d.py <input.ome.zarr> <output.ome.zarr> \
    --n_processes <n> \
    --percentile_max <pmax>
```

### linum_detect_focal_curvature.py

Detect and correct focal plane curvature.

```bash
linum_detect_focal_curvature.py <input.ome.zarr> <output.ome.zarr>
```

### linum_compensate_illumination.py

Apply illumination compensation.

```bash
linum_compensate_illumination.py <input> <output> <illumination_profile>
```

### linum_estimate_illumination.py

Estimate illumination profile from data.

```bash
linum_estimate_illumination.py <input> <output_profile>
```

### linum_compensate_attenuation.py

Compensate for signal attenuation with depth.

```bash
linum_compensate_attenuation.py <input> <output>
```

### linum_compute_attenuation.py

Compute attenuation profile.

```bash
linum_compute_attenuation.py <input> <output_profile>
```

### linum_compute_attenuation_bias_field.py

Compute attenuation bias field.

```bash
linum_compute_attenuation_bias_field.py <input> <output_field>
```

### linum_compensate_psf_model_free.py

Model-free PSF compensation (beam profile correction).

```bash
linum_compensate_psf_model_free.py <input.ome.zarr> <output.ome.zarr> \
    --percentile_max <pmax>
```

### linum_compensate_psf_from_model.py

PSF compensation using a model.

```bash
linum_compensate_psf_from_model.py <input> <output> <psf_model>
```

### linum_clip_percentile.py

Clip image intensities at percentiles.

```bash
linum_clip_percentile.py <input> <output> --lower <low> --upper <high>
```

### linum_crop_tiles.py

Crop tiles to specific region.

```bash
linum_crop_tiles.py <input_dir> <output_dir> --region <x1> <y1> <x2> <y2>
```

### linum_crop_3d_mosaic_below_interface.py

Crop 3D mosaic below sample interface.

```bash
linum_crop_3d_mosaic_below_interface.py <input.ome.zarr> <output.ome.zarr> \
    --depth <depth_um> \
    --crop_before_interface \
    --percentile_max <pmax>
```

### linum_normalize_intensities_per_slice.py

Normalize intensities per slice.

```bash
linum_normalize_intensities_per_slice.py <input.ome.zarr> <output.ome.zarr> \
    --percentile_max <pmax>
```

### linum_intensity_normalization.py

General intensity normalization.

```bash
linum_intensity_normalization.py <input> <output>
```

---

## Reconstruction

### linum_aip.py

Generate Average Intensity Projection.

```bash
linum_aip.py <input.ome.zarr> <output.ome.zarr>
```

### linum_stitch_2d.py

Stitch 2D tiles into mosaic.

```bash
linum_stitch_2d.py <input.ome.zarr> <output>
```

### linum_stitch_3d.py

Stitch 3D tiles into volume.

```bash
linum_stitch_3d.py <mosaic_grid.ome.zarr> <transform.npy> <output.ome.zarr>
```

### linum_stack_slices.py

Stack 2D slices into 3D volume.

```bash
linum_stack_slices.py <slices_dir> <output> --xy_shifts <shifts.csv>
```

### linum_stack_slices_3d.py

Stack 3D mosaics into final volume with analysis-optimized multi-resolution pyramid.

```bash
linum_stack_slices_3d.py <mosaics_dir> <transforms_dir> <output.ome.zarr> \
    [--blend] [--overlap <n>] [--pyramid_resolutions 10 25 50 100]
```

**Options:**
- `--blend`: Enable diffusion blending between slices
- `--overlap`: Maximum overlap voxels for blending
- `--pyramid_resolutions`: Target resolutions (µm) for pyramid levels (default: 10 25 50 100)
- `--n_levels`: Number of traditional power-of-2 downsample levels (alternative to pyramid_resolutions)

**Pyramid Resolution Modes:**

1. **Custom resolutions (default)**: Specify exact analysis-friendly resolutions
   ```bash
   linum_stack_slices_3d.py mosaics/ transforms/ output.ome.zarr \
       --pyramid_resolutions 10 25 50 100
   ```

2. **Traditional power-of-2**: Use `--n_levels` for 2x downsampling
   ```bash
   linum_stack_slices_3d.py mosaics/ transforms/ output.ome.zarr \
       --n_levels 4
   ```

**Note:** Only resolutions ≥ the base resolution will be created. The base resolution is automatically determined from the input data.

---

## Registration & Stitching

### linum_estimate_transform.py

Estimate XY transformation from mosaic grid.

```bash
linum_estimate_transform.py <aip.ome.zarr> <output_transform.npy>
```

### linum_estimate_transform_pairwise.py

Pairwise registration between consecutive slices. Uses phase correlation for initial alignment followed by intensity-based refinement.

```bash
linum_estimate_transform_pairwise.py <fixed.ome.zarr> <moving.ome.zarr> <output_dir> \
    [--moving_slice_index <idx>] \
    [--transform <type>] \
    [--metric <metric>] \
    [--max_translation <pixels>] \
    [--max_rotation <degrees>] \
    [--use_masks] \
    [--moving_mask <mask>] \
    [--fixed_mask <mask>]
```

**Key Options:**
- `--transform`: `translation` (default, recommended), `euler`, or `affine`
- `--metric`: `CC` (default, recommended), `MSE`, or `MI`
- `--max_translation`: Maximum allowed translation in pixels (default: 50.0)
- `--max_rotation`: Maximum allowed rotation in degrees (default: 2.0)
- `--use_masks`: Use tissue masks to focus registration on tissue regions

**Example:**

```bash
# Translation-only registration with masks (recommended)
linum_estimate_transform_pairwise.py fixed.ome.zarr moving.ome.zarr output/ \
    --transform translation --metric CC \
    --use_masks --moving_mask moving_mask.ome.zarr --fixed_mask fixed_mask.ome.zarr \
    --max_translation 30

# Simple registration without masks
linum_estimate_transform_pairwise.py fixed.ome.zarr moving.ome.zarr output/ \
    --transform translation --metric CC
```

### linum_estimate_xy_shift_from_metadata.py

Estimate XY shifts from tile metadata.

```bash
linum_estimate_xy_shift_from_metadata.py <tiles_dir> <output.csv> \
    --n_processes <n>
```

### linum_align_mosaics_3d_from_shifts.py

Align mosaics to common space using shifts file. This script brings all slices into a common coordinate system based on the physical positions recorded by the microscope.

```bash
linum_align_mosaics_3d_from_shifts.py <mosaics_dir> <shifts.csv> <output_dir> \
    [--slice_config <config.csv>] \
    [--filter_outliers] \
    [--outlier_method {clamp,median,zero,local,iqr}] \
    [--max_shift_mm <mm>] \
    [--iqr_multiplier <mult>] \
    [--no_center_drift]
```

**Options:**
- `--slice_config`: Optional CSV file to filter which slices to process
- `--filter_outliers`: Enable outlier detection and filtering (recommended)
- `--outlier_method`: Method to handle outliers:
  - `clamp`: Limit shift magnitude to `--max_shift_mm`
  - `median`: Replace with global median of non-outliers
  - `zero`: Replace with zero shift
  - `local`: Replace with local median of neighbors (preserves trends)
  - `iqr`: Auto-detect using IQR statistics and replace with local median (recommended)
- `--max_shift_mm`: Maximum allowed shift in mm (default: 0.5, only used if method != 'iqr')
- `--iqr_multiplier`: IQR multiplier for outlier detection (default: 1.5, only with 'iqr' method)
- `--no_center_drift`: Don't center drift around middle slice

**Example with outlier filtering:**
```bash
linum_align_mosaics_3d_from_shifts.py mosaics/ shifts_xy.csv output/ \
    --slice_config slice_config.csv \
    --filter_outliers \
    --outlier_method iqr
```

**Note:** The shifts file may contain erroneous large shifts due to stage positioning errors. The IQR-based filtering automatically detects these outliers and replaces them with reasonable values based on neighboring shifts.

### linum_apply_slices_transforms.py

Apply transforms to slices.

```bash
linum_apply_slices_transforms.py <input_dir> <transforms_dir> <output_dir>
```

### linum_estimate_slices_transforms_gui.py

GUI for manual slice transform estimation.

```bash
linum_estimate_slices_transforms_gui.py <slices_dir>
```

### linum_create_masks.py

Create binary masks for registration. Masks are saved with pyramid levels matching the input image.

```bash
linum_create_masks.py <input.ome.zarr> <output.ome.zarr> \
    --sigma <sigma> \
    --selem_radius <radius> \
    --min_size <size> \
    [--normalize] \
    [--n_levels <levels>] \
    [--preview <path.png>]
```

**Options:**
- `--sigma`: Gaussian smoothing sigma (default: 5.0)
- `--selem_radius`: Structuring element radius (default: 1)
- `--min_size`: Minimum object size in pixels (default: 100)
- `--normalize`: Normalize image before processing
- `--n_levels`: Number of pyramid levels (default: matches input image)
- `--preview`: Path to save a preview PNG for visual verification

---

## Slice Configuration

### linum_generate_slice_config.py

Generate slice configuration file with optional galvo shift detection.

```bash
# From mosaic grids
linum_generate_slice_config.py <mosaics_dir> <output.csv>

# From raw tiles
linum_generate_slice_config.py <tiles_dir> <output.csv> --from_tiles

# From shifts file
linum_generate_slice_config.py <shifts.csv> <output.csv> --from_shifts

# With exclusions
linum_generate_slice_config.py <input> <output.csv> --exclude 1 2 5

# With galvo detection (adds galvo_confidence and galvo_fix columns)
linum_generate_slice_config.py <tiles_dir> <output.csv> --from_tiles --detect_galvo

# From shifts with galvo detection
linum_generate_slice_config.py <shifts.csv> <output.csv> --from_shifts \
    --detect_galvo --tiles_dir <tiles_dir>

# Custom galvo threshold
linum_generate_slice_config.py <tiles_dir> <output.csv> --from_tiles \
    --detect_galvo --galvo_threshold 0.6
```

**Galvo Detection Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--detect_galvo` | off | Enable galvo shift detection |
| `--tiles_dir` | - | Raw tiles directory (if input is shifts file) |
| `--galvo_threshold` | 0.5 | Confidence threshold for galvo fix |

---

## Visualization

### linum_view_omezarr.py

Interactive OME-Zarr viewer.

```bash
linum_view_omezarr.py <input.ome.zarr>
```

### linum_view_zarr.py

Interactive Zarr viewer.

```bash
linum_view_zarr.py <input.zarr>
```

### linum_view_oct_raw_tile.py

View raw OCT tile.

```bash
linum_view_oct_raw_tile.py <tile_dir>
```

### linum_screenshot_omezarr.py

Generate screenshot from OME-Zarr.

```bash
linum_screenshot_omezarr.py <input.ome.zarr> <output.png>
```

### linum_generate_pipeline_report.py

Generate a quality report from pipeline metrics. Aggregates metrics JSON files from all processing steps and produces an HTML or text report.

```bash
# Generate HTML report (default)
linum_generate_pipeline_report.py <pipeline_output_dir> report.html

# Generate text report
linum_generate_pipeline_report.py <pipeline_output_dir> report.txt --format text

# Verbose report with all metric details
linum_generate_pipeline_report.py <pipeline_output_dir> report.html --verbose

# Custom title
linum_generate_pipeline_report.py <pipeline_output_dir> report.html --title "My Pipeline Report"
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | (required) | Directory containing pipeline output with metrics files |
| `output_report` | (required) | Output report file (.html or .txt) |
| `--format` | `auto` | Output format: `html`, `text`, or `auto` (infer from extension) |
| `--title` | `"Pipeline Quality Report"` | Report title |
| `--verbose` | `false` | Include all metric details in report |

---

## Utilities


### linum_resample.py

Resample image to different resolution.

```bash
linum_resample.py <input> <output> -r <resolution>
```

### linum_reorient_nifti_to_ras.py

Reorient NIfTI to RAS orientation.

```bash
linum_reorient_nifti_to_ras.py <input.nii.gz> <output.nii.gz>
```

### linum_axis_XYZ_to_ZYX.py

Transpose axes from XYZ to ZYX.

```bash
linum_axis_XYZ_to_ZYX.py <input> <output>
```

### linum_segment_brain_3d.py

Segment brain tissue in 3D volume.

```bash
linum_segment_brain_3d.py <input> <output_mask>
```

### linum_download_allen.py

Download Allen Brain Atlas data.

```bash
linum_download_allen.py <output_dir>
```

### linum_merge_slices_into_folders.py

Organize slices into folder structure.

```bash
linum_merge_slices_into_folders.py <input_dir> <output_dir>
```

---

## GPU-Accelerated Scripts

GPU-accelerated versions of key scripts. These use NVIDIA CUDA via CuPy for acceleration and automatically fall back to CPU if no GPU is available.

### linum_gpu_info.py

Check GPU availability and run quick benchmarks.

```bash
# Show GPU info
linum_gpu_info.py

# Run quick performance test
linum_gpu_info.py --test

# Output as JSON
linum_gpu_info.py --json
```

### linum_benchmark_gpu.py

Comprehensive CPU vs GPU benchmark suite.

```bash
# Quick benchmark (512x512)
linum_benchmark_gpu.py

# Full benchmark with multiple sizes
linum_benchmark_gpu.py --full

# Custom sizes
linum_benchmark_gpu.py --sizes 1024 2048 4096

# With real data
linum_benchmark_gpu.py --input /path/to/mosaic.ome.zarr

# Save results
linum_benchmark_gpu.py --output results.json --iterations 10
```

| Option | Description |
|--------|-------------|
| `--input` | Path to OME-Zarr for real-data benchmark |
| `--output, -o` | Save results to JSON file |
| `--iterations, -n` | Iterations per test (default: 3) |
| `--full` | Run with multiple sizes |
| `--sizes` | Custom sizes to test |
| `--skip-correctness` | Skip result verification |

### linum_estimate_transform_gpu.py

GPU-accelerated transform estimation using phase correlation.

```bash
linum_estimate_transform_gpu.py <input_images> <output.npy> [--use_gpu] [-v]
```

| Option | Description |
|--------|-------------|
| `--initial_overlap` | Initial overlap fraction (default: 0.3) |
| `--tile_shape` | Tile shape in pixels |
| `--n_samples` | Max tile pairs for optimization |
| `--use_gpu/--no-use_gpu` | Enable/disable GPU |

### linum_create_masks_gpu.py

GPU-accelerated tissue mask creation.

```bash
linum_create_masks_gpu.py <input.ome.zarr> <output.ome.zarr> [options]
```

| Option | Description |
|--------|-------------|
| `--sigma` | Gaussian smoothing sigma (default: 5.0) |
| `--selem_radius` | Structuring element radius (default: 1) |
| `--min_size` | Minimum object size (default: 100) |
| `--normalize` | Normalize before processing |
| `--use_gpu/--no-use_gpu` | Enable/disable GPU |

### linum_create_mosaic_grid_3d_gpu.py

GPU-accelerated mosaic grid creation with galvo detection.

```bash
linum_create_mosaic_grid_3d_gpu.py <output.ome.zarr> --from_tiles_list <tiles> [options]
```

| Option | Description |
|--------|-------------|
| `--resolution` | Output resolution in µm/pixel |
| `--fix_galvo_shift` | Enable galvo correction |
| `--galvo_threshold` | Galvo detection threshold (default: 0.6) |
| `--use_gpu/--no-use_gpu` | Enable/disable GPU |

### GPU Script Comparison

| GPU Script | CPU Equivalent | Accelerated Operations |
|------------|----------------|------------------------|
| `linum_estimate_transform_gpu.py` | `linum_estimate_transform.py` | FFT (9-47x), phase correlation (8-16x) |
| `linum_create_masks_gpu.py` | `linum_create_masks.py` | Gaussian filter (7-20x), binary morphology (7-67x) |
| `linum_create_mosaic_grid_3d_gpu.py` | `linum_create_mosaic_grid_3d.py` | Resize (5-12x) |

*Note: `linum_aip_gpu.py` was removed because mean projection is faster on CPU (0.5x speedup = GPU is 2x slower).*

---

## Common Options

Most scripts support these common options:

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--overwrite`, `-f` | Overwrite existing output |
| `--n_processes`, `-p` | Number of parallel processes |

---

## Getting Help

```bash
# Show script help
linum_<script_name>.py --help

# Example
linum_create_mosaic_grid_3d.py --help
```

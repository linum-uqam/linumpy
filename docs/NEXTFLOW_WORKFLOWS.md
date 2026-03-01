# Nextflow Workflows Guide


---

## Overview

linumpy uses [Nextflow](https://www.nextflow.io/) for orchestrating complex processing pipelines. Nextflow provides:

- **Parallelization**: Automatic parallel execution of independent tasks
- **Portability**: Run on local machines, clusters, or cloud
- **Reproducibility**: Containerized execution with Apptainer/Singularity
- **Fault tolerance**: Automatic retry and error handling

---

## Available Workflows

| Workflow | Location | Purpose |
|----------|----------|---------|
| `preproc_rawtiles.nf` | `workflows/preproc/` | Raw tiles → Mosaic grids |
| `soct_3d_reconst.nf` | `workflows/reconst_3d/` | Mosaic grids → 3D volume |

---

## Prerequisites

### Nextflow Installation

```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash

# Or via conda
conda install -c bioconda nextflow

# Verify installation
nextflow -version
```

**Required version**: >= 23.10

### Apptainer/Singularity (Optional)

For containerized execution:

```bash
# Install Apptainer
sudo apt install apptainer

# Or Singularity
sudo apt install singularity
```

---

## Preprocessing Workflow

### Location

```
workflows/preproc/
├── preproc_rawtiles.nf     # Workflow definition
└── nextflow.config          # Default configuration
```

### Purpose

Converts raw OCT tiles into organized mosaic grids and extracts metadata.

### Running

```bash
cd workflows/preproc

# Basic usage
nextflow run preproc_rawtiles.nf \
    --input /path/to/raw/tiles \
    --output /path/to/output

# With options
nextflow run preproc_rawtiles.nf \
    --input /path/to/raw/tiles \
    --output /path/to/output \
    --processes 8 \
    --resolution 10 \
    --axial_resolution 1.5
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input` | (required) | Raw tiles directory |
| `output` | `"output"` | Output directory |
| `use_old_folder_structure` | `false` | Use flat folder structure |
| `processes` | `1` | Parallel processes per task |
| `axial_resolution` | `1.5` | Axial resolution (µm) |
| `resolution` | `-1` | Output resolution (-1 = full) |
| `sharding_factor` | `4` | Zarr sharding (NxN chunks/shard) |
| `fix_galvo_shift` | `true` | Correct galvo shifts |
| `fix_camera_shift` | `false` | Correct camera shifts |
| `generate_slice_config` | `true` | Generate slice_config.csv |
| `use_gpu` | `true` | Enable GPU acceleration |

### Outputs

```
output/
├── mosaic_grid_3d_z00.ome.zarr/
├── mosaic_grid_3d_z01.ome.zarr/
├── ...
├── shifts_xy.csv
└── slice_config.csv
```

---

## 3D Reconstruction Workflow

### Location

```
workflows/reconst_3d/
├── soct_3d_reconst.nf      # Workflow definition
└── nextflow.config          # Default configuration
```

### Purpose

Processes mosaic grids through multiple correction and stitching steps to produce a final 3D volume.

### Running

```bash
cd workflows/reconst_3d

# Basic usage
nextflow run soct_3d_reconst.nf \
    --input /path/to/mosaic/grids \
    --shifts_xy /path/to/shifts_xy.csv \
    --output /path/to/output

# With slice config
nextflow run soct_3d_reconst.nf \
    --input /path/to/mosaic/grids \
    --shifts_xy /path/to/shifts_xy.csv \
    --slice_config /path/to/slice_config.csv \
    --output /path/to/output

# Full options
nextflow run soct_3d_reconst.nf \
    --input /path/to/mosaic/grids \
    --shifts_xy /path/to/shifts_xy.csv \
    --output /path/to/output \
    --resolution 10 \
    --processes 4 \
    --fix_curvature_enabled true \
    --fix_illum_enabled true \
    --create_registration_masks true
```

### Parameters

#### Input/Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input` | `"."` | Mosaic grids directory |
| `shifts_xy` | `""` | XY shifts file (default: `{input}/shifts_xy.csv`) |
| `slice_config` | `""` | Optional slice config |
| `output` | `"."` | Output directory |
| `subject_name` | `""` | Subject identifier (auto-extracted from path) |
| `processes` | `8` | Parallel Python processes per task |

#### Compute Resources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gpu` | `true` | Enable GPU acceleration (auto-fallback to CPU) |
| `enable_cpu_limits` | `true` | Enable CPU limiting |
| `max_cpus` | `16` | Maximum CPUs to use (0 = no limit) |
| `reserved_cpus` | `4` | CPUs reserved for system overhead |

#### Resolution & Basic Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | `10` | Target resolution (µm/pixel) |
| `clip_percentile_upper` | `99.9` | Upper percentile for intensity clipping |
| `fix_curvature_enabled` | `false` | Detect and compensate focal curvature artifacts |
| `fix_illum_enabled` | `true` | Fix illumination inhomogeneity (BaSiCPy algorithm) |
| `crop_interface_out_depth` | `600` | Maximum tissue depth after interface crop (µm) |
| `normalize_min_contrast` | `0.1` | Min contrast fraction to prevent over-amplification of empty slices (0–1) |

#### Tile Stitching

These parameters control how tiles within each slice are assembled in XY.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_motor_positions_for_stitching` | `true` | Use motor encoder positions for tile layout (recommended) |
| `stitch_overlap_fraction` | `0.2` | Expected tile overlap fraction — should match acquisition settings |
| `stitch_blending_method` | `'diffusion'` | Tile blending: `'none'`, `'average'`, `'diffusion'` |
| `use_refined_stitching` | `true` | Use image registration to refine blend transitions; reduces tile seams |
| `max_blend_refinement_px` | `10` | Maximum sub-pixel refinement shift for blending (pixels) |

#### Common Space Alignment

Aligns each slice into a shared XY canvas using `shifts_xy.csv` motor positions, with optional outlier and step filtering.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_shift_outliers` | `true` | Enable outlier filtering (strongly recommended) |
| `outlier_method` | `'iqr'` | Detection method: `iqr`, `local`, `median`, `clamp`, or `zero` |
| `outlier_iqr_multiplier` | `1.5` | IQR multiplier for outlier detection |
| `max_shift_mm` | `0.5` | Maximum allowed shift (mm) — floor on IQR threshold |
| `common_space_max_step_mm` | `0.5` | Maximum per-step shift change (mm, 0 = disabled) |
| `common_space_step_window` | `3` | Window size for step outlier detection |
| `common_space_step_method` | `'local_median'` | Step correction method: `local_median`, `clamp`, `local_mad` |
| `common_space_step_mad_threshold` | `3.0` | MADs above local median to flag a step outlier (only with `local_mad`) |
| `common_space_excluded_slice_mode` | `'local_median'` | Shift interpolation for excluded slices |
| `common_space_excluded_slice_window` | `2` | Window for excluded slice interpolation |

**Outlier Methods:**
- `iqr` (recommended): Auto-detect outliers using IQR statistics, replace with local median
- `local`: Replace outliers with local median of neighboring shifts
- `median`: Replace outliers with global median
- `clamp`: Limit magnitude to `max_shift_mm` while preserving direction
- `zero`: Replace outliers with zero shift

#### Pairwise Registration

Computes small corrections (rotation, sub-pixel translation) between consecutive slices. The main XY alignment comes from motor positions; these transforms are refinements applied on top.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `registration_transform` | `'euler'` | Transform type: `euler` (XY + rotation) or `translation` (XY only) |
| `registration_max_translation` | `200.0` | Optimizer bound on translation (pixels) |
| `registration_max_rotation` | `5.0` | Optimizer bound on rotation (degrees) |
| `moving_slice_first_index` | `4` | Starting Z-index in the moving volume |
| `registration_slicing_interval_mm` | `0.200` | Physical slice thickness (mm) |
| `registration_allowed_drifting_mm` | `0.100` | Z-search range (mm) |

**Registration Masks:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `create_registration_masks` | `true` | Create tissue masks to focus registration on tissue |
| `mask_smoothing_sigma` | `5.0` | Gaussian smoothing sigma for mask creation (µm) |
| `selem_radius` | `1` | Morphological structuring element radius (pixels) |
| `min_size` | `100` | Minimum mask component size (pixels²) |
| `mask_normalize` | `true` | Normalize intensities before masking |
| `mask_fill_holes` | `'slicewise'` | Hole filling: `none`, `3d`, `slicewise` |

#### Stacking & Output

Choose a stacking method with `stacking_method`:
- **`motor`** (recommended): XY from motor positions (`shifts_xy.csv`), Z from correlation
- **`registration`**: XY and Z both from pairwise registration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stacking_method` | `'motor'` | Stacking method: `motor` or `registration` |
| `stack_blend_enabled` | `true` | Blend overlapping regions between slices |

**Motor stacking (stacking_method = 'motor'):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_expected_z_overlap` | `true` | Use expected Z-overlap instead of correlation-based matching |
| `apply_pairwise_transforms` | `true` | Apply pairwise registration transforms during stacking |
| `apply_rotation_only` | `true` | Apply only the rotation component from registration (keeps XY from motor) |
| `max_rotation_deg` | `5.0` | Clamp rotation values larger than this before application |
| `skip_error_transforms` | `true` | Skip transforms flagged as `error` status (prevents artifacts near interpolated slices) |
| `skip_warning_transforms` | `false` | Skip transforms flagged as `warning` status |
| `stack_accumulate_translations` | `false` | Accumulate pairwise translations as cumulative canvas offsets |
| `stack_max_pairwise_translation` | `0` | Max pairwise translation (pixels) included in accumulation (0 = include all) |
| `stitch_rehoming_enabled` | `false` | Apply one-time segment offset at re-homing event boundaries |
| `stitch_rehoming_threshold_mm` | `0.7` | Motor shift magnitude that identifies a re-homing event (mm) |
| `stitch_rehoming_use_motor` | `false` | Use motor delta instead of pairwise registration for re-homing corrections |
| `stack_smooth_window` | `5` | Moving-average window (slices) for smoothing per-slice rotations (0 = disabled) |

**Registration stacking (stacking_method = 'registration'):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stack_max_overlap` | `10` | Maximum blending overlap in voxels (-1 = unlimited) |
| `stack_no_accumulate_transforms` | `true` | Apply each transform independently (recommended when slices are already XY-aligned) |

**Output pyramid:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pyramid_resolutions` | `[10, 25, 50, 100]` | Target resolutions (µm) for output pyramid levels |
| `pyramid_n_levels` | `null` | Fixed level count (overrides `pyramid_resolutions`) |
| `pyramid_make_isotropic` | `true` | Resample to isotropic voxel spacing |

The `pyramid_resolutions` parameter controls the multi-resolution pyramid in the final 3D volume. Instead of power-of-2 downsampling, specific analysis-friendly resolutions are used:

- **10 µm**: High-resolution analysis
- **25 µm**: Standard analysis resolution
- **50 µm**: Overview and atlas registration
- **100 µm**: Quick visualization and large-scale analysis

**Note:** Only resolutions ≥ the base `resolution` parameter will be included. For example, if `resolution = 25`, then only 25, 50, and 100 µm levels will be created.

#### Z-Intensity Normalization

Corrects slow intensity drift across serial sections after stacking. Disabled by default.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalize_z_slices` | `false` | Enable post-stacking Z-intensity normalization |
| `znorm_mode` | `'histogram'` | Normalization mode: `histogram` (preserves contrast) or `percentile` (linear scaling) |
| `znorm_strength` | `0.5` | Correction mixing strength (0 = passthrough, 1 = full correction) |

**Histogram mode** (`znorm_mode = 'histogram'`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `znorm_tissue_threshold` | `0.02` | Minimum intensity to classify as tissue (below this left unchanged) |

**Percentile mode** (`znorm_mode = 'percentile'`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `znorm_smooth_sigma` | `10.0` | Gaussian smoothing sigma (sections); ~10 corrects ~2mm drift and preserves anatomy |
| `znorm_percentile` | `80.0` | Percentile of non-zero tissue voxels used as intensity reference |
| `znorm_max_scale` | `2.0` | Maximum correction scale factor |
| `znorm_min_scale` | `0.5` | Minimum correction scale factor |

#### Atlas Registration (RAS Alignment)

Register the final reconstructed volume to the Allen Mouse Brain Atlas (CCF) to produce an RAS-aligned OME-Zarr output. Atlas data is downloaded automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `align_to_ras_enabled` | `false` | Enable Allen atlas registration |
| `allen_resolution` | `25` | Atlas resolution for registration (µm): 10, 25, 50, 100 |
| `allen_metric` | `'MI'` | Registration metric: `MI`, `MSE`, `CC`, `AntsCC` |
| `allen_max_iterations` | `1000` | Maximum registration iterations |
| `allen_registration_level` | `2` | Pyramid level of input zarr for registration (0 = full; level 2 ≈ 50 µm, fast) |
| `ras_input_orientation` | `''` | 3-letter orientation code of the INPUT brain volume (see table below) |
| `ras_initial_rotation` | `''` | Initial rotation hint `'Rx Ry Rz'` in degrees (leave empty for automatic MOMENTS initialization) |
| `allen_preview` | `true` | Generate a 3-panel alignment comparison image |

**Output:** `align_to_ras/{subject}_ras.ome.zarr` with all pyramid resolutions.

---

#### RAS Orientation Lookup Table

The `ras_input_orientation` parameter is a 3-letter code describing the anatomical direction each axis of the **input ZYX zarr** points toward. The code is interpreted as:

```
letter 1 → dim0 (zarr Z) = slice stacking direction (perpendicular to cutting plane)
letter 2 → dim1 (zarr Y) = in-plane row direction
letter 3 → dim2 (zarr X) = in-plane column direction
```

Each letter is one of: `R`/`L` (right/left), `A`/`P` (anterior/posterior), `S`/`I` (superior/inferior).

The script `linum_align_to_ras.py` uses the code to permute and flip axes before registration, bringing the volume into approximate RAS space. The `ras_initial_rotation` then seeds the registration optimizer with a coarse rotation, which is essential for oblique cuts.

**Standard setup assumption** used in the table below:

> Brain mounted with dorsal side up. OCT motor rows (zarr Y) scan dorsal→ventral (I). OCT motor columns (zarr X) scan left→right for coronal/axial (R), or posterior→anterior for sagittal (A).

<!--
Orientation code construction:
  letter_map: R→axis0(+), L→axis0(−), A→axis1(+), P→axis1(−), S→axis2(+), I→axis2(−)
  The code permutes/flips zarr ZYX axes so that axis0=R, axis1=A, axis2=S in the output.
-->

##### Cardinal (in-plane) cutting orientations

| Cutting plane | Stack direction | Row dir (Y) | Col dir (X) | `ras_input_orientation` |
|---|---|---|---|---|
| Coronal — anterior→posterior | A→P | Dorsal→Ventral (I) | Left→Right (R) | `PIR` |
| Coronal — posterior→anterior | P→A | Dorsal→Ventral (I) | Left→Right (R) | `AIR` |
| Sagittal — left→right | L→R | Dorsal→Ventral (I) | Posterior→Anterior (A) | `RIA` |
| Sagittal — right→left | R→L | Dorsal→Ventral (I) | Posterior→Anterior (A) | `LIA` |
| Axial/Horizontal — dorsal→ventral | D→V | Anterior→Posterior (P) | Left→Right (R) | `IPR` |
| Axial/Horizontal — ventral→dorsal | V→D | Anterior→Posterior (P) | Left→Right (R) | `SPR` |

> **Important:** The in-plane letters (2nd and 3rd) depend on the physical stage motor orientation and brain mounting. If the output looks mirrored or rotated 90°, swap or negate the in-plane letters. Run `linum_align_to_ras.py --preview-only` to inspect the raw volume orientation before registering.

##### 45° oblique cutting orientations

For cuts between two cardinal planes, use the closest cardinal code plus `ras_initial_rotation` to seed the registration with the approximate tilt angle. The sign depends on which specific diagonal direction the cut follows — verify with `--preview` after registration.

| Cutting plane | Between planes | `ras_input_orientation` | `ras_initial_rotation`¹ | Rotation axis |
|---|---|---|---|---|
| Corono-sagittal 45° | Coronal ↔ Sagittal | `PIR` | `'0 0 ±45'` | Around RAS Superior-Inferior (Rz) |
| Corono-axial 45° | Coronal ↔ Axial | `PIR` | `'±45 0 0'` | Around RAS Right-Left (Rx) |
| Sagitto-axial 45° | Sagittal ↔ Axial | `RIA` | `'0 ±45 0'` | Around RAS Anterior-Posterior (Ry) |

¹ Sign (+ or −) depends on the specific oblique direction. Start with +45 and inspect the preview; negate if the alignment is worse.

**Rotation axis guide** (applied in the approximately-RAS frame after orientation correction):
- `Rx` — tilts the A-P axis toward/away from S-I (e.g., pitch)
- `Ry` — tilts the R-L axis toward/away from S-I (e.g., roll)
- `Rz` — rotates in the axial plane, mixing R-L and A-P (e.g., yaw)

**Example config (coronal A→P, standard setup):**
```groovy
align_to_ras_enabled    = true
ras_input_orientation   = 'PIR'
ras_initial_rotation    = ''        // automatic MOMENTS initialization
allen_resolution        = 25
allen_registration_level = 2        // ~50 µm pyramid level for speed
```

**Example config (corono-sagittal 45° oblique cut):**
```groovy
align_to_ras_enabled    = true
ras_input_orientation   = 'PIR'
ras_initial_rotation    = '0 0 45'  // adjust sign after checking preview
allen_resolution        = 25
allen_registration_level = 2
```

---

#### Previews & Reports

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_preview` | `true` | Generate stitched slice preview images |
| `common_space_preview` | `false` | Generate common space alignment previews |
| `interpolation_preview` | `false` | Generate interpolated slice previews |
| `generate_report` | `true` | Generate HTML quality report after stacking |
| `report_verbose` | `false` | Include detailed per-slice metrics in report |
| `annotated_label_every` | `1` | Label every Nth slice in annotated preview (1 = all slices) |
| `annotated_show_lines` | `false` | Draw slice boundary lines on annotated preview |

#### Debugging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `analyze_shifts` | `false` | Generate shifts analysis report and drift plots |
| `mask_preview` | `false` | Save mask preview images alongside mask zarrs |
| `debug_slices` | `""` | Comma-separated slice IDs or ranges to process (e.g. `"25,26"` or `"25-29"`); leave empty to process all |

The `analyze_shifts` option runs drift analysis on the shifts file before processing, producing:
- A text report with statistics and outlier detection
- A PNG plot showing drift patterns
- A filtered shifts CSV file

#### Diagnostic Mode

Diagnostic mode enables additional analysis processes for troubleshooting reconstruction artifacts (edge mismatches, overhangs, alignment issues).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diagnostic_mode` | `false` | Master switch: enables all diagnostic analyses |
| `analyze_rotation_drift` | `false` | Analyze cumulative rotation between slices |
| `analyze_acquisition_rotation` | `false` | Analyze acquisition-time rotation from shifts + registration |
| `analyze_tile_dilation` | `false` | Analyze tile position refinements for scale drift (requires `use_refined_stitching=false`) |
| `motor_only_stitch` | `false` | Stitch slices using motor positions only (no image registration) |
| `motor_only_stack` | `false` | Stack slices using motor positions only (no pairwise registration) |
| `compare_stitching` | `false` | Compare motor-only vs refined stitching side-by-side |

Diagnostic parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `motor_only_overlap` | `0.2` | Expected tile overlap for motor-only diagnostics (matches `stitch_overlap_fraction`) |
| `motor_only_stitch_blending` | `'diffusion'` | Blending for motor-only stitching: `none`, `average`, `diffusion` |
| `motor_only_stack_blending` | `'none'` | Blending for motor-only stacking: `none`, `average`, `max`, `feather` |
| `diagnostic_rotation_threshold` | `2.0` | Rotation warning threshold (degrees) |
| `save_refinement_data` | `false` | Save refined stitching transform data as JSON |
| `comparison_tile_step` | `60` | Tile step for seam detection in stitching comparison |

Diagnostic outputs are written to `{output}/diagnostics/` and include rotation plots, dilation reports, motor-only stitch results, and side-by-side comparisons.

#### GPU Acceleration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gpu` | `true` | Enable GPU acceleration (auto-fallback to CPU) |

### Outputs

```
output/
├── README/readme.txt
├── resample_mosaic_grid/
├── fix_focal_curvature/
├── fix_illumination/
├── generate_aip/
├── estimate_xy_transformation/
├── stitch_3d/
├── beam_profile_correction/
├── crop_interface/
├── normalize/
├── bring_to_common_space/
├── create_registration_masks/
├── register_pairwise/
├── stack/
│   ├── 3d_volume.ome.zarr
│   ├── 3d_volume.ome.zarr.zip
│   └── 3d_volume.png
├── normalize_z_intensity/              # Only when normalize_z_slices = true
│   └── 3d_volume_znorm.ome.zarr
├── align_to_ras/                       # Only when align_to_ras_enabled = true
│   ├── {subject}_ras.ome.zarr          # RAS-aligned volume (all pyramid levels)
│   ├── {subject}_ras_transform.tfm     # Registration transform (SimpleITK)
│   └── {subject}_ras_preview.png       # 3-panel alignment comparison
├── diagnostics/                        # Only when diagnostic_mode = true or individual flags set
│   ├── rotation_analysis/
│   ├── acquisition_rotation/
│   ├── dilation_analysis/
│   ├── aggregated_dilation/
│   ├── motor_only_stitch/
│   ├── motor_only_stack/
│   └── stitch_comparison/
└── {subject}_quality_report.html
```

---

## GPU Acceleration

Both workflows support GPU acceleration using NVIDIA CUDA via CuPy. GPU processing is enabled by default and automatically falls back to CPU if no GPU is available.

### GPU-Accelerated Processes

| Workflow | Process | GPU Operations |
|----------|---------|----------------|
| `preproc_rawtiles.nf` | `create_mosaic_grid` | Galvo detection, volume resize |
| `soct_3d_reconst.nf` | `generate_aip` | Mean projection |
| `soct_3d_reconst.nf` | `estimate_xy_transformation` | Phase correlation (FFT) |
| `soct_3d_reconst.nf` | `create_registration_masks` | Gaussian filter, morphology |

### Usage

```bash
# GPU enabled (default)
nextflow run preproc_rawtiles.nf --input /data --output /output

# Disable GPU
nextflow run preproc_rawtiles.nf --input /data --output /output --use_gpu false

# 3D reconstruction with GPU
nextflow run soct_3d_reconst.nf --input /mosaics --output /output --use_gpu true
```

### Config-Based Control

```groovy
// In nextflow.config
params {
    use_gpu = true   // Enable GPU (default)
    // use_gpu = false  // Force CPU only
}
```

### Requirements

For GPU support:
- NVIDIA GPU with CUDA support
- CuPy installed: `pip install cupy-cuda12x`
- See [GPU_ACCELERATION.md](GPU_ACCELERATION.md) for detailed setup

### Expected Speedups

On NVIDIA A6000 (48GB):

| Operation | Speedup |
|-----------|---------|
| Phase correlation | 10-15x |
| Volume resize | 5-10x |
| AIP projection | 3-4x |
| Mask creation | 2-4x |

---

## CPU Core Management

The pipelines provide fine-grained control over CPU usage, allowing you to reserve cores for system overhead and manage the interplay between Nextflow parallelism and Python multiprocessing.

### Configuration Options

Both pipelines support two approaches:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_cpus` | `null` | Explicit maximum CPUs to use (takes precedence) |
| `reserved_cpus` | `2` | Number of cores to keep free for overhead |
| `processes` | `1` | Python processes per Nextflow task |

### Usage Examples

#### Reserve Cores for Overhead (Recommended)

```bash
# Keep 2 cores free for system overhead (default)
nextflow run soct_3d_reconst.nf \
    --input /path/to/data \
    --reserved_cpus 2

# Keep 4 cores free on a heavily-loaded system
nextflow run soct_3d_reconst.nf \
    --input /path/to/data \
    --reserved_cpus 4
```

#### Set Explicit Core Limit

```bash
# Use exactly 16 cores maximum
nextflow run soct_3d_reconst.nf \
    --input /path/to/data \
    --max_cpus 16
```

### Understanding the Interplay

The total CPU usage depends on three factors:

1. **Nextflow parallelism**: How many tasks run simultaneously
2. **Python processes per task**: The `processes` parameter
3. **Thread libraries**: NumPy/SciPy threading (OMP, MKL, OpenBLAS)

The effective formula is:
```
Total threads ≈ (Nextflow parallel tasks) × (processes) × (threads per process)
```

The pipeline automatically:
- Sets `LINUMPY_MAX_CPUS` or `LINUMPY_RESERVED_CPUS` environment variables for Python scripts
- Configures `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` to prevent thread oversubscription

### Disabling CPU Limits

If the CPU limiting system causes issues (e.g., unexpectedly slow performance), you can disable it entirely:

```bash
# Disable CPU limits - all cores will be used
nextflow run workflow.nf --enable_cpu_limits false
```

This will skip all environment variable settings and let processes use all available cores.

### Recommended Configurations

| System Type | reserved_cpus | processes | Notes |
|-------------|--------------|-----------|-------|
| Workstation (8-16 cores) | 2 | 2-4 | Good balance |
| Server (32+ cores) | 4 | 4-8 | Leave room for I/O |
| Shared system | 8+ | 2 | Conservative to avoid impacting others |
| Dedicated processing | 1 | auto | Maximum throughput |

### Environment Variables

Python scripts in linumpy respect these environment variables:

| Variable | Description |
|----------|-------------|
| `LINUMPY_MAX_CPUS` | Maximum CPUs to use (explicit limit) |
| `LINUMPY_RESERVED_CPUS` | CPUs to reserve for overhead |
| `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` | Thread limit for SimpleITK operations |

These can also be set manually when running scripts directly:

```bash
# Reserve 4 cores when running standalone scripts
LINUMPY_RESERVED_CPUS=4 linum_create_mosaic_grid_3d.py input.ome.zarr output.ome.zarr

# Or set explicit max
LINUMPY_MAX_CPUS=8 linum_stitch_3d.py mosaic_grid.ome.zarr transform.npy output.ome.zarr
```

---

## Configuration Files

### nextflow.config Structure

```groovy
manifest {
    nextflowVersion = '>= 23.10'
}

params {
    // Default parameter values
    input = "."
    output = "."
    // ... more parameters
}

process {
    // Process-level settings
    publishDir = {"$params.output/$slice_id/$task.process"}
    scratch = true
    errorStrategy = { task.attempt <= 2 ? 'retry' : 'ignore' }
    maxRetries = 2
}

apptainer {
    autoMounts = true
    enabled = true
}

profiles {
    // Environment-specific profiles
    calliste {
        // HPC cluster settings
    }
}
```

### Using Custom Config

```bash
# Use custom config file
nextflow run workflow.nf -c my_config.config

# Override specific parameters
nextflow run workflow.nf --resolution 5 --processes 8
```

---

## Execution Profiles

### Local Execution

```bash
nextflow run workflow.nf
```

### HPC Cluster (SLURM)

```groovy
// In nextflow.config
profiles {
    slurm {
        process.executor = 'slurm'
        process.queue = 'normal'
        process.memory = '16 GB'
        process.cpus = 4
    }
}
```

```bash
nextflow run workflow.nf -profile slurm
```

### Containerized Execution

```groovy
// In nextflow.config
apptainer {
    enabled = true
    cacheDir = '/path/to/cache'
}
```

```bash
nextflow run workflow.nf -with-apptainer linumpy.sif
```

---

## Monitoring and Debugging

### Progress Monitoring

```bash
# Real-time progress
nextflow run workflow.nf

# With execution report
nextflow run workflow.nf -with-report report.html

# With timeline
nextflow run workflow.nf -with-timeline timeline.html

# With DAG visualization
nextflow run workflow.nf -with-dag dag.png
```

### Resume Failed Runs

```bash
# Resume from last checkpoint
nextflow run workflow.nf -resume
```

### Clean Up

```bash
# Clean work directory
nextflow clean -f

# Clean specific run
nextflow clean -f <run_name>
```

### Log Files

```
.nextflow.log         # Main log file
.nextflow/            # Nextflow cache and history
work/                 # Task working directories
```

---

## Common Issues

### Out of Memory

```groovy
// Increase memory in config
process {
    memory = '32 GB'
}
```

### Disk Space

```bash
# Check work directory size
du -sh work/

# Clean after successful run
rm -rf work/
```

### Container Issues

```bash
# Pull container manually
apptainer pull linumpy.sif docker://ghcr.io/linum/linumpy:latest

# Run with explicit container
nextflow run workflow.nf -with-apptainer linumpy.sif
```

### Permission Errors

```bash
# Check file permissions
ls -la work/

# Fix ownership
sudo chown -R $USER:$USER work/
```

---

## Best Practices

### 1. Use Version Control for Configs

```bash
# Track your custom configs
git add nextflow.config
git commit -m "Add custom pipeline config"
```

### 2. Test with Small Data First

```bash
# Run on subset
nextflow run workflow.nf --input /path/to/test_data
```

### 3. Monitor Resource Usage

```bash
# With resource report
nextflow run workflow.nf -with-report -with-trace
```

### 4. Use Profiles for Different Environments

```groovy
profiles {
    local { /* laptop settings */ }
    hpc { /* cluster settings */ }
    cloud { /* AWS/GCP settings */ }
}
```

### 5. Keep Work Directory on Fast Storage

```bash
# Set work directory
nextflow run workflow.nf -w /fast/storage/work
```

---

## Reference

- [Nextflow Documentation](https://www.nextflow.io/docs/latest/)
- [Nextflow Patterns](https://nextflow-io.github.io/patterns/)
- [nf-core Guidelines](https://nf-co.re/docs/)

# Nextflow Workflows Guide


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
    --axial_resolution 1.36
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input` | (required) | Raw tiles directory |
| `output` | `"output"` | Output directory |
| `use_old_folder_structure` | `false` | Use flat folder structure |
| `use_gpu` | `true` | Enable GPU acceleration (auto-fallback to CPU) |
| `processes` | `1` | Parallel Python processes per task (CPU mode only) |
| `max_mosaic_forks` | `4` | Max concurrent `create_mosaic_grid` GPU jobs |
| `max_aip_forks` | `4` | Max concurrent `generate_aip` GPU jobs |
| `axial_resolution` | `1.36` | Axial resolution (µm) |
| `resolution` | `-1` | Output resolution (-1 = full native resolution) |
| `sharding_factor` | `4` | Zarr sharding (NxN chunks/shard) |
| `fix_galvo_shift` | `true` | Correct galvo shifts |
| `fix_camera_shift` | `false` | Correct camera shifts |
| `preprocess` | `false` | Apply rotation/flip preprocessing (true for legacy data) |
| `galvo_confidence_threshold` | `0.6` | Minimum confidence to apply galvo fix |
| `generate_slice_config` | `true` | Generate slice_config.csv |
| `exclude_first_slices` | `1` | Number of leading slices to mark as excluded |
| `detect_galvo` | `false` | Include galvo detection results in slice_config.csv |
| `generate_previews` | `false` | Generate orthogonal view previews of mosaic grids |
| `generate_aips` | `false` | Generate AIP images from mosaic grids for QC |

### Outputs

```
output/
├── mosaic_grid_3d_z00.ome.zarr/
├── mosaic_grid_3d_z01.ome.zarr/
├── ...
├── shifts_xy.csv
├── slice_config.csv
├── aips/                          # Only when generate_aips = true
│   ├── aip_z00.png
│   └── ...
└── previews/                      # Only when generate_previews = true
    ├── mosaic_grid_z00_preview.png
    └── ...
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
    --fix_illum_enabled true
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


#### Tile Stitching

These parameters control how tiles within each slice are assembled in XY.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_motor_positions_for_stitching` | `true` | Use motor encoder positions for tile layout (recommended) |
| `stitch_overlap_fraction` | `0.2` | Expected tile overlap fraction — should match acquisition settings |
| `stitch_blending_method` | `'diffusion'` | Tile blending: `'none'`, `'average'`, `'diffusion'` |
| `max_blend_refinement_px` | `10` | Maximum sub-pixel refinement shift for blending (pixels) |

**Global tile-placement transform** (optional). When enabled, one 2×2 affine is
fitted across a pool of mid-brain mosaic grids (instrument geometry is slice-
invariant) and re-used for every slice. This removes per-slice scale/rotation
jitter that the default refined stitcher introduces when the LS fit is
underdetermined on small or sparse grids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_global_transform` | `false` | Enable pooled global affine estimation |
| `stitch_global_transform_slices` | `''` | Optional comma-separated slice IDs to pool from (empty = all) |
| `stitch_global_transform_histogram_match` | `true` | Match overlap histograms before phase correlation |
| `stitch_global_transform_max_empty_fraction` | `0.9` | Otsu-based empty-overlap filter fraction (null = simpler check) |
| `stitch_global_transform_n_samples` | `2048` | Max pooled pairs for the LS fit (0 = use all) |
| `stitch_global_transform_seed` | `0` | Random seed for pair sub-sampling |

#### Common Space Alignment

Aligns each slice into a shared XY canvas using `shifts_xy.csv` motor positions.
Erroneous shifts are corrected **upstream** via re-homing detection rather than
post-hoc outlier filtering.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_rehoming` | `true` | Correct encoder glitch spikes (round-trip steps) before alignment |
| `rehoming_return_fraction` | `0.4` | Spike sensitivity: lower = fewer corrections (adjacent step must reverse > (1−this) of current) |
| `rehoming_max_shift_mm` | `0.5` | Steps below this magnitude are not checked for spikes |
| `tile_fov_mm` | `null` | Tile field-of-view (mm); set only for legacy shifts files where `xmin_mm` jumped by whole tile columns |
| `tile_fov_tolerance` | `0.05` | Fractional tolerance around each tile-FOV multiple |
| `common_space_excluded_slice_mode` | `'local_median'` | Handling for excluded-slice shifts: `keep`, `local_median`, `median`, `zero` |
| `common_space_excluded_slice_window` | `2` | Window size for `local_median` replacement |
| `common_space_refine_unreliable` | `false` | Re-estimate `reliable=0` transitions with 2-D phase cross-correlation |
| `common_space_refine_max_discrepancy_px` | `0` | Reject image-based estimates differing from motor by more than this (0 = accept all) |
| `common_space_refine_min_correlation` | `0.0` | Minimum NCC to accept an image-based refinement |

#### Missing Slice Interpolation

Fill single-slice gaps in the normalized slice sequence. Two-or-more consecutive
gaps are not interpolated (insufficient information) and remain as holes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interpolate_missing_slices` | `true` | Interpolate single-slice gaps |
| `interpolation_method` | `'zmorph'` | Method: `zmorph` (z-aware morphing), `weighted`, `average` |
| `interpolation_blend_method` | `'gaussian'` | Blend: `gaussian` (feathered edges) or `linear` |
| `interpolation_registration_metric` | `'MSE'` | Similarity metric for boundary-plane registration |
| `interpolation_max_iterations` | `1000` | Maximum registration iterations |
| `interpolation_overlap_search_window` | `5` | Z-planes to search at each boundary for best overlap pair |
| `interpolation_min_overlap_correlation` | `0.3` | Pre-registration NCC threshold below which zmorph falls back |
| `interpolation_reference_slab_size` | `3` | Planes averaged around boundary reference plane |
| `interpolation_min_foreground_fraction` | `0.1` | Minimum foreground fraction for a boundary plane |
| `interpolation_min_ncc_improvement` | `0.05` | Min post-reg NCC improvement to accept the transform |

When zmorph's quality gates fail the slot is left as a genuine gap (no zarr
output); a manifest fragment and diagnostics JSON are still emitted. See
[SLICE_INTERPOLATION_FEATURE.md](SLICE_INTERPOLATION_FEATURE.md) for details.

#### Automatic Slice Quality Assessment

Runs `linum_assess_slice_quality` on normalized slices and stamps a
`slice_config.csv` that marks degraded slices for exclusion before the common-
space step.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_assess_quality` | `false` | Enable automatic quality assessment |
| `auto_assess_min_quality` | `0.3` | Exclude slices with quality score below this |
| `auto_assess_exclude_first` | `1` | Exclude first N calibration slices automatically |
| `auto_assess_roi_size` | `1024` | Centre-crop size in XY for quality metrics (0 = full plane) |

#### Pairwise Registration

Computes small corrections (rotation, sub-pixel translation) between consecutive slices. The main XY alignment comes from motor positions; these transforms are refinements applied on top.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `registration_transform` | `'euler'` | Transform type: `euler` (XY + rotation) or `translation` (XY only) |
| `registration_max_translation` | `200.0` | Optimizer bound on translation (pixels) |
| `registration_max_rotation` | `5.0` | Optimizer bound on rotation (degrees) |
| `registration_initial_alignment` | `'both'` | Initial alignment before refinement: `none`, `com`, `gradient`, or `both` |
| `moving_slice_first_index` | `4` | Starting Z-index in the moving volume |
| `registration_slicing_interval_mm` | `0.200` | Physical slice thickness (mm) |
| `registration_allowed_drifting_mm` | `0.100` | Z-search range (mm) |

#### Stacking & Output

Stacking assembles all common-space slices into a 3D volume using motor positions
for XY placement, pairwise registration for rotation/translation refinement,
and correlation or physics-based Z-matching.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stack_blend_enabled` | `true` | Blend overlapping regions between slices |
| `blend_refinement_px` | `0` | Z-blend refinement: phase-correlation XY correction in the overlap zone before blending (0 = disabled) |
| `stack_blend_z_refine_vox` | `5` | Z-blend position refinement: search up to N voxels below the expected boundary for the best-correlated plane |

**Motor stacking / transform application:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_expected_z_overlap` | `true` | Use expected Z-overlap instead of correlation-based matching |
| `apply_pairwise_transforms` | `true` | Apply pairwise registration transforms during stacking |
| `apply_rotation_only` | `false` | Apply only the rotation component (keeps XY from motor) |
| `max_rotation_deg` | `5.0` | Clamp rotation values larger than this before application |

**Confidence-based transform degradation.** A confidence score (0–1) is computed from Z-correlation, translation magnitude and rotation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transform_confidence_high` | `0.6` | Above this: full transform applied |
| `transform_confidence_low` | `0.3` | Between low and high: rotation-only; below low: skipped |
| `z_overlap_min_corr` | `0.5` | Fall back to expected Z-overlap below this NCC score |
| `blend_z_refine_min_confidence` | `0.5` | Min confidence to run blend Z-refinement (else use expected overlap) |

**Transform gating:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skip_error_transforms` | `true` | Skip transforms flagged `overall_status="error"` |
| `skip_warning_transforms` | `true` | Skip transforms flagged `overall_status="warning"` |
| `load_transform_min_zcorr` | `0.0` | Metric-based gating: min z_correlation to load a transform (0 = disabled) |
| `load_transform_max_rotation` | `0.0` | Metric-based gating: max rotation (degrees) (0 = disabled) |

**Auto-exclude extended low-quality clusters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_exclude_enabled` | `true` | Enable automatic cluster detection |
| `auto_exclude_consecutive` | `3` | Min consecutive low-quality pairs to trigger exclusion |
| `auto_exclude_z_corr` | `0.6` | Z-correlation threshold below which a pair is low-quality |

**Translation accumulation & smoothing:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stack_accumulate_translations` | `true` | Accumulate pairwise translations as cumulative canvas offsets |
| `stack_confidence_weight_translations` | `true` | Weight each translation by its confidence before accumulating |
| `stack_max_cumulative_drift_px` | `50` | Max cumulative translation drift from motor baseline (0 = unlimited) |
| `stack_max_pairwise_translation` | `0` | Values near this limit are assumed optimizer-boundary hits and zeroed out (0 = accumulate all) |
| `stack_smooth_window` | `5` | Moving-average window (slices) for smoothing per-slice rotations (0 = disabled) |
| `stack_translation_smooth_sigma` | `3.0` | Gaussian sigma (slices) for smoothing accumulated translations (0 = disabled) |
| `stack_translation_min_zcorr` | `0.2` | Min z_correlation to use a slice's translation in accumulation |

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

#### Bias Field Correction

Corrects slow intensity drift and bias field across serial sections after stacking using N4 bias field correction (SimpleITK). Disabled by default.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `correct_bias_field` | `false` | Enable post-stacking N4 bias field correction |
| `bias_mode` | `'two_pass'` | Correction mode: `per_section` (N4 per thick section), `global` (single volume pass), or `two_pass` (per-section then global) |
| `bias_strength` | `1.0` | Correction mixing strength (0 = passthrough, 1 = full correction) |

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

#### Manual Alignment Export

Export a lightweight data package for interactive manual alignment of pairwise
slice transforms (`tools/manual-align/`). When manually corrected transforms
exist, the stack step can pick them up and optionally re-refine them with a
tight image-based registration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `export_manual_align` | `false` | Export manual alignment data after `register_pairwise` |
| `manual_align_level` | `1` | Pyramid level for AIP export (0 = full, 1 = 2×, …) |
| `manual_transforms_dir` | `''` | Directory of manually-corrected transforms; overrides automated ones for matching slices |
| `refine_manual_transforms` | `false` | Re-run pairwise registration on manual pairs, seeded from the manual transform |
| `refine_max_translation_px` | `10` | Max residual translation searched during refinement (pixels) |
| `refine_max_rotation_deg` | `2.0` | Max residual rotation searched during refinement (degrees) |

#### Previews & Reports

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_preview` | `true` | Generate stitched slice preview images |
| `common_space_preview` | `false` | Generate common space alignment previews |
| `rehoming_diagnostics` | `false` | Save `rehoming_report.json` + `rehoming_plot.png` |
| `interpolation_preview` | `false` | Generate interpolated slice previews |
| `generate_report` | `true` | Generate HTML quality report after stacking |
| `report_verbose` | `false` | Include detailed per-slice metrics in report |
| `report_format` | `'zip'` | `'html'` (lightweight) or `'zip'` (HTML + bundled previews) |
| `annotated_label_every` | `1` | Label every Nth slice in annotated preview (1 = all slices) |
| `annotated_show_lines` | `false` | Draw slice boundary lines on annotated preview |

#### Debugging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `analyze_shifts` | `true` | Generate shifts analysis report and drift plots |
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
├── analyze_shifts/                     # Only when analyze_shifts = true
├── resample_mosaic_grid/
├── fix_focal_curvature/
├── fix_illumination/
├── stitch_3d_with_refinement/
├── previews/stitched_slices/           # Only when stitch_preview = true
├── beam_profile_correction/
├── crop_interface/
├── normalize/
├── detect_rehoming_events/             # Only when detect_rehoming = true
├── auto_assess_quality/                # Only when auto_assess_quality = true
├── bring_to_common_space/
├── common_space_previews/              # Only when common_space_preview = true
├── interpolate_missing_slice/          # Only when interpolate_missing_slices = true
├── finalise_interpolation/
├── register_pairwise/
├── auto_exclude_slices/                # Only when auto_exclude_enabled = true
├── stack/
│   ├── {subject}.ome.zarr
│   ├── {subject}.ome.zarr.zip
│   ├── {subject}.png
│   └── {subject}_annotated.png
├── correct_bias_field/                 # Only when correct_bias_field = true
│   └── {subject}_corrected.ome.zarr
├── align_to_ras/                       # Only when align_to_ras_enabled = true
│   ├── {subject}_ras.ome.zarr
│   ├── {subject}_ras_transform.tfm
│   └── {subject}_ras_preview.png
├── diagnostics/                        # Only when diagnostic_mode = true or individual flags set
│   ├── rotation_analysis/
│   ├── acquisition_rotation/
│   ├── motor_only_stitch/
│   ├── refined_stitch/
│   ├── motor_only_stack/
│   └── stitch_comparison/
└── {subject}_quality_report.html       # Only when generate_report = true
```

---

## GPU Acceleration

Both workflows support GPU acceleration using NVIDIA CUDA via CuPy. GPU processing is enabled by default and automatically falls back to CPU if no GPU is available.

### GPU-Accelerated Processes

| Workflow | Process | GPU Operations |
|----------|---------|----------------|
| `preproc_rawtiles.nf` | `create_mosaic_grid` | Galvo detection, volume resize |
| `preproc_rawtiles.nf` | `generate_aip` | Mean projection |
| `soct_3d_reconst.nf` | `resample_mosaic_grid` | Volume resize |
| `soct_3d_reconst.nf` | `fix_illumination` | BaSiCPy background correction (PyTorch on GPU) |
| `soct_3d_reconst.nf` | `normalize` | Intensity normalization, percentile clipping |

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
- CuPy installed: `uv pip install cupy-cuda12x`
- See [GPU_ACCELERATION.md](GPU_ACCELERATION.md) for detailed setup

### Expected Speedups

On NVIDIA A6000 (48GB):

| Operation | Speedup |
|-----------|---------|
| Phase correlation | 10-15x |
| Volume resize | 5-10x |
| AIP projection | 3-4x |

---

## CPU Core Management

The pipelines provide fine-grained control over CPU usage, allowing you to reserve cores for system overhead and manage the interplay between Nextflow parallelism and Python multiprocessing.

### Configuration Options

Both pipelines support two approaches. Defaults differ between workflows:
the **preproc** pipeline ships with `max_cpus = null` and `reserved_cpus = 2`,
while the **3D reconstruction** pipeline uses `max_cpus = 16` and
`reserved_cpus = 4` (see `workflows/<pipeline>/nextflow.config`).

| Parameter | preproc default | reconst_3d default | Description |
|-----------|-----------------|--------------------|-------------|
| `max_cpus` | `null` | `16` | Explicit maximum CPUs to use (takes precedence) |
| `reserved_cpus` | `2` | `4` | Number of cores to keep free for overhead |
| `processes` | `1` | `1` | Python processes per Nextflow task |

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

### Reconstruction Robustness Presets

The 3D reconstruction workflow ships with three preset profiles that bundle
related stacking/registration parameters:

| Profile | When to use |
|---------|-------------|
| `conservative` (default behaviour) | Trusts motor positions for XY, applies rotation-only from registration, skips unreliable transforms, interpolates single-slice gaps. Recommended starting point. |
| `aggressive` | Uses full pairwise transforms (XY + rotation) and accumulates them cumulatively. Best alignment when registration is reliable, degrades badly when it is not. |
| `minimal` | Motor-only stacking — ignores all pairwise registration. Most stable and fastest; use when motor positions are reliable and registration consistently fails. |

```bash
# Pick a robustness preset
nextflow run soct_3d_reconst.nf -profile conservative --input ... --output ...
nextflow run soct_3d_reconst.nf -profile aggressive   --input ... --output ...
nextflow run soct_3d_reconst.nf -profile minimal      --input ... --output ...
```

Individual parameters may still be overridden on the command line on top of a
profile.

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

## Authoring Notes (linumpy reconstruction workflow)

These notes apply specifically to `workflows/reconst_3d/soct_3d_reconst.nf` and
explain a few patterns and pitfalls that the workflow file otherwise had to
re-document inline. Refer to this section before changing the channel topology.

### Value vs queue channels

Nextflow has two channel kinds, with very different consumption semantics:

- **Queue channels** are consumed once. When two operators read from the same
  queue channel, only the first one observes the data; the second sees an
  empty channel. Any process that depends (directly or transitively) on an
  empty channel is silently skipped — there is no error.
- **Value channels** can be consumed any number of times by any number of
  downstream operators. They are the safe default for inputs that fan out.

In the reconstruction workflow several channels fan out to multiple
consumers. They must therefore be value channels:

| Channel             | Consumers                                                              |
|---------------------|------------------------------------------------------------------------|
| `shifts_xy`         | `analyze_shifts`, `detect_rehoming`, `bring_to_common_space`, `stack`, `analyze_acquisition_rotation`, `stack_motor_only` |
| `slice_config_channel` / `current_slice_config` | `auto_assess`, `detect_rehoming`, `bring_to_common_space`, `finalise_interpolation`, `auto_exclude_slices`, `stack` |
| `no_transform`      | `stitch_3d_with_refinement` (combined with every slice tuple)           |

To create a value channel from a file path, use `channel.value(file(path))`,
**not** `channel.of(file(path))` or `channel.fromPath(path)` — the latter two
produce queue channels that exhaust after the first consumer.

### Auto-promotion of process outputs

Nextflow DSL2 auto-promotes a process output to a value channel when **all**
of the process's inputs are value channels. The reconstruction workflow
relies on this: once the source channels above are value channels, every
downstream `process.out` we re-assign to `current_slice_config` is also a
value channel — no `.first()` is needed to convert it.

If you ever introduce a queue input into one of those processes (e.g. by
collecting per-slice tuples without `.collect()`), the corresponding output
will revert to a queue channel and `current_slice_config` will silently
exhaust the next time it is consumed twice.

### `.first()` is a last resort

`.first()` converts a queue channel to a value channel by emitting only the
first value. We avoid it in the reconstruction workflow because:

1. When applied to a value channel it triggers the warning
   `WARN: The operator first is useless when applied to a value channel which returns a single value by definition`.
2. Whenever it is *needed* it indicates that a source or upstream process is
   producing a queue channel where it shouldn't — usually a sign that one of
   the source-channel rules above was violated.

Prefer fixing the upstream channel kind. Reach for `.first()` only when an
external API genuinely returns a queue channel that you cannot influence.

### `tuple path(...), path(...)` + `.combine()` flattens lists

When a process input is declared as a tuple of paths and the upstream
channel is built with `.combine()` against a `.collect()`-ed list:

```groovy
auto_assess_inputs = normalize.out.normalized.map { _id, p -> p }.collect()  // list of paths
auto_assess_quality(auto_assess_inputs.combine(existing_slice_config))
```

…Nextflow flattens the collected list into the tuple binding, so the first
zarr in the list is bound to the slot that was supposed to receive the
config CSV (we observed this as
`IsADirectoryError: Is a directory: 'slice_z02_normalize.ome.zarr'`).

The fix is to declare each item as a separate input and pass them as
separate positional arguments:

```groovy
process auto_assess_quality {
    input:
    path "inputs/*"
    path existing_slice_config
}

auto_assess_quality(auto_assess_inputs, channel.value(existing_slice_config_file))
```

The same pattern applies to `finalise_interpolation` and
`auto_exclude_slices`.

### Slice config flow

Several stages produce or consume a per-slice configuration CSV
(`slice_config.csv`). The flow is:

```
slice_config_channel
   └── (auto_assess_quality, optional) ──┐
                                         ▼
                       effective_slice_config (value)
                                         │
                                         ▼
                          current_slice_config (value)
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
      detect_rehoming           bring_to_common_space     finalise_interpolation
              │                                                    │
              └────► current_slice_config (re-bound) ◄──────────────┘
                                         │
                                         ▼
                              auto_exclude_slices
                                         │
                                         ▼
                            stack_slice_config (value)
                                         │
                                         ▼
                                       stack
```

Every assignment along that flow is a value channel, so the same config can
be merged into `bring_to_common_space` and combined into `stack_input`
without exhaustion.

### Hard-skip behaviour for failed interpolation

`interpolate_missing_slice` produces an *optional* `zarr` output. When
zmorph's quality gates reject the interpolation it emits a manifest fragment
with `interpolation_failed=true` and no zarr; `finalise_interpolation`
stamps that into `slice_config_final.csv` and the slot stays a genuine gap
in the stacked volume. See [`SLICE_INTERPOLATION_FEATURE.md`](SLICE_INTERPOLATION_FEATURE.md)
for the full policy.

### `finalise_interpolation` is published-only

`finalise_interpolation.out` is **not** rebound to `current_slice_config`
even though it logically refines it. Two reasons:

1. When there are no single-slice gaps, `interpolate_missing_slice` is not
   invoked; its `.manifest` channel never emits; `manifest.collect()`
   therefore does not emit either; `finalise_interpolation` is not invoked;
   and `finalise_interpolation.out` is an empty channel. Rebinding
   `current_slice_config` to that empty channel propagates the emptiness
   downstream and **silently skips `stack`** (and everything after it).
2. `linum_stack_slices_motor.py` only reads `use` and `auto_excluded` from
   the slice config (via `slice_config_io.force_skip_slices`).
   `finalise_interpolation` only adds `interpolated` and
   `interpolation_failed`, so it does not change any column that `stack`
   acts on. The published `slice_config_final.csv` is consumed directly
   from the output directory by `linum_generate_pipeline_report.py`, which
   gracefully falls back to `slice_config.csv` if the final file is absent.

Treat `finalise_interpolation` as an artifact-emitting side effect; do not
rebind its output into the channel chain.

### File layout (linumpy reconstruction workflow)

```
workflows/reconst_3d/
├── soct_3d_reconst.nf   main workflow + per-stage processes
├── diagnostics.nf       optional diagnostic processes (rotation analyses,
│                        motor-only stitch / stack, comparison)
└── nextflow.config      defaults; subject overrides live next to the data
```

`soct_3d_reconst.nf` is organised top-down:

1. **Helper functions** — small Groovy utilities reused across processes
   (`gpuScript`, `pyramidArgs`, `annotatedScreenshotArgs`, `extractSliceId`,
   per-concern `stack*Args` builders, etc.). Keep new helpers here so the
   pipeline body stays declarative.
2. **`include`** of `diagnostics.nf` for the optional analyses.
3. **Process definitions**, grouped by stage (utility → preprocessing →
   stitching → corrections → alignment → registration → stacking).
4. **`workflow {}`** — the actual stage-by-stage wiring.

When adding a new process: prefer extending an existing helper to copy-pasting
shell-arg blocks (e.g. add another `stack*Args()` rather than an inline
60-line `if` chain inside the process script).

### GPU flag

Each process that performs GPU-accelerated work passes `--use_gpu` or
`--no-use_gpu` based on `params.use_gpu`. Use the pattern:

```groovy
def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
"""
linum_foo.py ... ${gpu_flag}
"""
```

There is no longer a separate `<stem>_gpu.py` script; all GPU-capable scripts
have a single unified name and accept `--use_gpu`/`--no-use_gpu`.

---

## Reference

- [Nextflow Documentation](https://www.nextflow.io/docs/latest/)
- [Nextflow Patterns](https://nextflow-io.github.io/patterns/)
- [nf-core Guidelines](https://nf-co.re/docs/)

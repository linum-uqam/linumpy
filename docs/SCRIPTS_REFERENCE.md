# Scripts Reference


## Overview

linumpy provides a comprehensive set of command-line scripts for microscopy data processing. All scripts follow a consistent interface with `--help` for usage information.

> **Source layout.** Most scripts live in `scripts/`. Diagnostic and
> benchmark tools live in `scripts/diagnostics/`. All scripts are exposed
> as console entry points by `pyproject.toml`, so they can be invoked
> directly by name regardless of source location.

---

## Script Categories

1. [Data Conversion](#data-conversion)
2. [Mosaic Grid Operations](#mosaic-grid-operations)
3. [Preprocessing](#preprocessing)
4. [Reconstruction](#reconstruction)
5. [Registration & Stitching](#registration--stitching)
6. [Diagnostics](#diagnostics)
7. [Slice Configuration](#slice-configuration)
8. [Visualization](#visualization)
9. [Utilities](#utilities)
10. [GPU-Accelerated Scripts](#gpu-accelerated-scripts)

---

## Data Conversion

### linum-convert-tiff-to-nifti

Convert TIFF stack to NIfTI format.

```bash
linum-convert-tiff-to-nifti <input.tiff> <output.nii.gz>
```

### linum-convert-tiff-to-omezarr

Convert TIFF to OME-Zarr format.

```bash
linum-convert-tiff-to-omezarr <input.tiff> <output.ome.zarr>
```

### linum-convert-nifti-to-nrrd

Convert NIfTI to NRRD format.

```bash
linum-convert-nifti-to-nrrd <input.nii.gz> <output.nrrd>
```

### linum-convert-nifti-to-zarr

Convert NIfTI to Zarr format.

```bash
linum-convert-nifti-to-zarr <input.nii.gz> <output.zarr>
```

### linum-convert-omezarr-to-nifti

Convert OME-Zarr to NIfTI format.

```bash
linum-convert-omezarr-to-nifti <input.ome.zarr> <output.nii.gz>
```

### linum-convert-zarr-to-omezarr

Convert Zarr to OME-Zarr format.

```bash
linum-convert-zarr-to-omezarr <input.zarr> <output.ome.zarr>
```

### linum-convert-bin-to-nii

Convert binary file to NIfTI.

```bash
linum-convert-bin-to-nii <input.bin> <output.nii.gz>
```

---

## Mosaic Grid Operations

### linum-create-mosaic-grid-2d

Create 2D mosaic grid from tiles.

```bash
linum-create-mosaic-grid-2d <output.ome.zarr> --from_tiles_list <tiles...>
```

### linum-create-mosaic-grid-3d

Create 3D mosaic grid from raw OCT tiles.

```bash
linum-create-mosaic-grid-3d <output.ome.zarr> \
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

### linum-create-all-mosaic-grids-2d

Create all 2D mosaic grids from a tiles directory.

```bash
linum-create-all-mosaic-grids-2d <tiles_dir> <output_dir>
```

### linum-generate-mosaic-aips

Generate AIP (Average Intensity Projection) PNG previews from a directory of mosaic grid OME-Zarr files. Useful for quick QC visualization of all slices at once.

```bash
linum-generate-mosaic-aips <mosaics_dir> <output_dir> \
    [--level <pyramid_level>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--level` | `0` | Pyramid level to use (higher = faster, lower resolution) |

### linum-resample-mosaic-grid

Resample mosaic grid to different resolution.

```bash
linum-resample-mosaic-grid <input.ome.zarr> <output.ome.zarr> -r <resolution>
```

---

## Preprocessing

### linum-fix-illumination-3d

Correct illumination inhomogeneity using BaSiC algorithm.

```bash
linum-fix-illumination-3d <input.ome.zarr> <output.ome.zarr> \
    --n_processes <n> \
    --percentile_max <pmax>
```

### linum-detect-focal-curvature

Detect and correct focal plane curvature in a 3D mosaic.

```bash
linum-detect-focal-curvature <input.ome.zarr> <output.ome.zarr> \
    [--n_levels <n>] \
    [--n_processes <n>] \
    [--block_size <n>] \
    [--use_log] \
    [--use_gpu] \
    [--verbose]
```

### linum-compensate-illumination

Apply illumination compensation.

```bash
linum-compensate-illumination <input> <output> <illumination_profile>
```

### linum-estimate-illumination

Estimate illumination profile from data.

```bash
linum-estimate-illumination <input> <output_profile>
```

### linum-compensate-attenuation

Compensate for signal attenuation with depth using a precomputed bias field.

```bash
linum-compensate-attenuation <input.ome.zarr> <bias.ome.zarr> <output.ome.zarr>
```

### linum-compensate-attenuation-inplace

Per-slice depth-attenuation compensation from a single OME-Zarr volume. Computes the Vermeer 2014 (or Liu 2019 / Li 2020) attenuation map and applies the gain in one pass. This is the script invoked by the Nextflow reconstruction workflow.

```bash
linum-compensate-attenuation-inplace <input.ome.zarr> <output.ome.zarr> \
    [--method {li,liu,smith,vermeer}] \
    [--strength <0.0-1.0>] \
    [--k <voxels>] \
    [--zshift <voxels>] \
    [--snr_threshold_db <dB>] \
    [--min_bias <float>] \
    [--mask_smoothing_sigma <float>] \
    [--n_levels <n>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `li` | Depth-resolved estimator: `li` (Li 2020, default — noise-floor subtraction + SNR-based A-line truncation on top of Liu 2019), `liu` (Liu 2019 exact form), `smith` (legacy linumpy default), `vermeer` (bare estimator, no regularization). See `docs/ATTENUATION_METHODS.md`. |
| `--strength` | `0.3` | Multiplicative scale on the optical-depth correction (0..1). 1.0 = textbook Vermeer formula; <1 attenuates the correction. ~0.30 yields a near-flat depth profile on cropped 600 µm brain slices. |
| `--k` | `10` | XY median-filter kernel (voxels) applied before the Vermeer fit. `0` disables denoising. |
| `--zshift` | `3` | Voxels under the water/tissue interface to ignore when fitting the exponential tail. Smaller values keep more of the shallow tissue. |
| `--snr_threshold_db` | `6.0` | Per-voxel SNR threshold (dB) for A-line truncation in `li`. Voxels with SNR below this are excluded from the fit. |
| `--min_bias` | `0.05` | Floor applied to the bias field before division. Caps the maximum gain at `1/min_bias` and prevents noise amplification in deep voxels. |
| `--mask_smoothing_sigma` | `2.0` | Gaussian sigma (XY voxels) for the Otsu tissue mask. |
| `--n_levels` | `0` | Pyramid levels in the output (0 = single resolution). |

### linum-compute-attenuation

Compute attenuation profile.

```bash
linum-compute-attenuation <input> <output_profile>
```

### linum-compute-attenuation-bias-field

Compute attenuation bias field.

```bash
linum-compute-attenuation-bias-field <input> <output_field>
```

### linum-compensate-psf-model-free

Model-free PSF compensation (beam profile correction).

```bash
linum-compensate-psf-model-free <input.ome.zarr> <output.ome.zarr> \
    --percentile_max <pmax>
```

### linum-compensate-psf-from-model

Fit a confocal-PSF parametric model (focal depth + Rayleigh length) on a few axial profiles, then divide each A-line by the synthesized 3D PSF.

```bash
linum-compensate-psf-from-model <input.ome.zarr> <output.ome.zarr> \
    [--zr_initial <µm>] \
    [--percentile_max <p>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--zr_initial` | `610` | Initial Rayleigh length seed (µm) for the fit. Use `1060` for the 10× Mitutoyo objective on the current rig. Sensitive to acquisition geometry — refit per instrument when in doubt (see script docstring). |

### linum-clip-percentile

Clip image intensities at percentiles.

```bash
linum-clip-percentile <input> <output> --lower <low> --upper <high>
```

### linum-crop-tiles

Crop tiles to specific region.

```bash
linum-crop-tiles <input_dir> <output_dir> --region <x1> <y1> <x2> <y2>
```

### linum-crop-3d-mosaic-below-interface

Crop 3D mosaic below sample interface.

```bash
linum-crop-3d-mosaic-below-interface <input.ome.zarr> <output.ome.zarr> \
    --depth <depth_um> \
    --crop_before_interface \
    --percentile_max <pmax>
```

### linum-normalize-intensities-per-slice

Normalize intensities per slice.

```bash
linum-normalize-intensities-per-slice <input.ome.zarr> <output.ome.zarr> \
    --percentile_max <pmax>
```

### linum-intensity-normalization

General intensity normalization.

```bash
linum-intensity-normalization <input> <output>
```

### linum-correct-bias-field

Correct slow intensity drift and bias field across serial sections after stacking using N4 bias field correction (SimpleITK). Three modes are supported: `per_section` (N4 applied independently per thick section), `global` (single N4 pass over the whole volume), and `two_pass` (per-section pass followed by a global pass).

```bash
linum-correct-bias-field <input.ome.zarr> <output.ome.zarr> \
    [--mode {per_section,global,two_pass}] \
    [--strength <0.0-1.0>] \
    [--n_serial_slices <n>] \
    [--n_processes <n>] \
    [--shrink_factor <n>] \
    [--n_iterations <n> [<n> ...]] \
    [--spline_distance_mm <val>] \
    [--mask_smoothing_sigma <val>] \
    [--save_bias_field <path>] \
    [--n_levels <n>] \
    [--verbose]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `two_pass` | Correction mode: `per_section`, `global`, or `two_pass` |
| `--strength` | `1.0` | Mixing strength (0 = passthrough, 1 = full correction) |
| `--n_serial_slices` | `1` | Slices per section for `per_section` mode |
| `--n_processes` | `1` | Number of parallel worker processes |
| `--shrink_factor` | `4` | Downsampling factor before N4 fitting (faster, less memory) |
| `--n_iterations` | `[50,50,50,50]` | N4 iterations per fitting level (length = number of levels) |
| `--spline_distance_mm` | auto | B-spline control point spacing (default: 2.0 mm per-section, 10.0 mm global) |
| `--mask_smoothing_sigma` | `2.0` | Gaussian sigma for tissue mask smoothing |
| `--save_bias_field` | — | If given, write the estimated bias field to this OME-Zarr path |
| `--n_levels` | `None` | Pyramid levels in output OME-Zarr (auto-chosen if unset) |

---

## Reconstruction

### linum-aip

Generate Average Intensity Projection.

```bash
linum-aip <input.ome.zarr> <output.ome.zarr>
```

### linum-stitch-2d

Stitch 2D tiles into mosaic.

```bash
linum-stitch-2d <input.ome.zarr> <output>
```

### linum-stitch-3d

Stitch 3D tiles into volume using a pre-computed transform.

```bash
linum-stitch-3d <mosaic_grid.ome.zarr> <transform.npy> <output.ome.zarr>
```

### linum-stitch-3d-refined

Stitch 3D tiles with image-registration-refined blend transitions to reduce visible tile seams.

```bash
linum-stitch-3d-refined <mosaic_grid.ome.zarr> <output.ome.zarr> \
    [--overlap_fraction <frac>] \
    [--blending_method {none,average,diffusion}] \
    [--refinement_mode blend_shift] \
    [--max_refinement_px <pixels>] \
    [--input_transform <transform.npy>] \
    [--output_refinements <refinements.json>]
```

| Option | Description |
|--------|-------------|
| `--overlap_fraction` | Expected tile overlap fraction (default: 0.2) |
| `--blending_method` | Tile blending method: `none`, `average`, `diffusion` |
| `--refinement_mode` | How refinement shifts are used (e.g. `blend_shift`) |
| `--max_refinement_px` | Maximum sub-pixel refinement shift (pixels) |
| `--input_transform` | Pre-computed global 2×2 affine (from `linum-estimate-global-transform`) |
| `--output_refinements` | Optional JSON file to save refinement data |

### linum-estimate-global-transform

Estimate a single 2×2 tile-placement affine pooled across many 3D mosaic grids.
Instrument geometry is slice-invariant, so using one fitted transform for every
slice removes per-slice scale/rotation jitter that the default refined stitcher
introduces when the least-squares fit is underdetermined on small or sparse grids.

```bash
linum-estimate-global-transform <mosaics_dir> <output_transform.npy> \
    [--slices <id1,id2,...>] \
    [--histogram_match] \
    [--max_empty_fraction <frac>] \
    [--n_samples <n>] \
    [--seed <seed>]
```

The output `.npy` can be passed to `linum-stitch-3d-refined --input_transform`.

### linum-analyze-stitch-affine

Per-slice affine diagnostic for the refined stitching step. Inspects
`estimate_xy_transformation` outputs (or the refined stitcher's
`refinements.json`) and reports scale / rotation drift across slices.

```bash
linum-analyze-stitch-affine <input_dir> <output_dir>
```

### linum-stitch-motor-only

Stitch tiles using motor encoder positions only (no image registration). Useful for diagnostics and comparing against refined stitching.

```bash
linum-stitch-motor-only <mosaic_grid.ome.zarr> <output.ome.zarr> \
    [--overlap_fraction <frac>] \
    [--blending_method {none,average,diffusion}]
```

### linum-stack-slices-2d

Stack 2D AIPs into a 3D volume using `shifts_xy.csv`.

```bash
linum-stack-slices-2d <slices_dir> <output> --xy_shifts <shifts.csv>
```

### linum-stack-slices-3d

> **Deprecated.** Use `linum-stack-slices-motor` with `--no_xy_shift` instead.

Stack 3D mosaics into final volume using pairwise registration transforms.
This script is superseded by `linum-stack-slices-motor`, which provides
confidence-based transform degradation, translation filtering/accumulation,
rotation smoothing, auto-exclude, and richer diagnostics.

### linum-stack-slices-motor

Stack slices into a 3D volume using motor positions for XY placement and correlation-based Z matching. This is the primary stacking script used by the pipeline.

```bash
linum-stack-slices-motor <slices_dir> <shifts_file> <output.ome.zarr> \
    [--transforms_dir <dir>] \
    [--rotation_only] \
    [--max_rotation_deg 1.0] \
    [--accumulate_translations] \
    [--max_pairwise_translation 0] \
    [--confidence_weight_translations] \
    [--max_cumulative_drift_px 0] \
    [--smooth_window 0] \
    [--translation_smooth_sigma 0] \
    [--skip_error_transforms] \
    [--skip_warning_transforms] \
    [--no_xy_shift] \
    [--slicing_interval_mm 0.200] \
    [--search_range_mm 0.100] \
    [--use_expected_overlap] \
    [--z_overlap_min_corr 0.5] \
    [--moving_z_first_index 8] \
    [--blend] \
    [--blend_depth] \
    [--blend_refinement_px 0] \
    [--blend_z_refine_vox 0] \
    [--pyramid_resolutions 10 25 50 100] \
    [--make_isotropic | --no_isotropic] \
    [--max_slices <n>] \
    [--output_z_matches] \
    [--output_stacking_decisions] \
    [--confidence_high 0.6] \
    [--confidence_low 0.3] \
    [--blend_z_refine_min_confidence 0.5] \
    [--slice_config <config.csv>] \
    [--load_min_zcorr <val>] \
    [--load_max_rotation <deg>] \
    [--translation_min_zcorr <val>] \
    [--manual_transforms_dir <dir>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transforms_dir` | — | Directory of pairwise registration transforms |
| `--rotation_only` | off | Apply only the rotation component from pairwise transforms |
| `--max_rotation_deg` | `1.0` | Clamp rotations larger than this value (degrees) |
| `--accumulate_translations` | off | Accumulate pairwise translations as cumulative canvas offsets |
| `--max_pairwise_translation` | `0` | Zero out translations near this optimizer-boundary limit (0 = accumulate all) |
| `--confidence_weight_translations` | off | Weight translations by confidence before accumulating |
| `--max_cumulative_drift_px` | `0` | Cap cumulative drift from motor baseline (0 = unlimited) |
| `--smooth_window` | `0` | Moving-average window (slices) for per-slice rotation smoothing (0 = disabled) |
| `--translation_smooth_sigma` | `0` | Gaussian sigma (slices) for smoothing accumulated translations (0 = disabled) |
| `--skip_error_transforms` | off | Skip transforms flagged `overall_status="error"` |
| `--skip_warning_transforms` | off | Skip transforms flagged `overall_status="warning"` |
| `--no_xy_shift` | off | Ignore XY shifts from motor CSV (stack without XY displacement) |
| `--slice_config` | — | CSV to filter which slices are included / motor-only |
| `--load_max_rotation` | — | Metric-based gate: skip transforms with rotation above this threshold |

### linum-stack-motor-only

Stack slices using motor positions only (no pairwise registration). Used for diagnostics to isolate the motor-position contribution.

```bash
linum-stack-motor-only <slices_dir> <shifts_file> <output.ome.zarr> \
    [--blending {none,average,max,feather}] \
    [--preview <preview.png>]
```

---

## Registration & Stitching

### linum-estimate-transform

Estimate XY transformation from mosaic grid.

```bash
linum-estimate-transform <aip.ome.zarr> <output_transform.npy>
```

### linum-estimate-xy-shift-from-metadata

Estimate XY shifts from tile metadata.

```bash
linum-estimate-xy-shift-from-metadata <tiles_dir> <output.csv> \
    --n_processes <n>
```

### linum-align-mosaics-3d-from-shifts

Align mosaics to a common XY canvas using the shifts CSV. Each mosaic is resampled to a common shape and translated using cumulative shifts; when slices are excluded via `--slice_config`, their shifts are accumulated so the remaining slices stay aligned.

Large erroneous shifts should be corrected **upstream** with
`linum-detect-rehoming` (see below) before running this script. This script
no longer implements outlier filtering directly.

```bash
linum-align-mosaics-3d-from-shifts <mosaics_dir> <shifts.csv> <output_dir> \
    [--slice_config <config.csv>] \
    [--excluded_slice_mode {keep,local_median,median,zero}] \
    [--excluded_slice_window <n>] \
    [--no_center_drift] \
    [--refine_unreliable] \
    [--refine_max_discrepancy_px <px>] \
    [--refine_min_correlation <0-1>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--slice_config` | — | Optional slice-config CSV to filter slices |
| `--excluded_slice_mode` | `keep` | Handling for shifts of excluded slices: `keep`, `local_median`, `median`, `zero` |
| `--excluded_slice_window` | `2` | Neighbor window for `local_median` replacement |
| `--no_center_drift` | off | Disable centering drift around middle slice |
| `--refine_unreliable` | off | For transitions flagged `reliable=0`, use 2-D phase correlation to replace the metadata shift (requires scikit-image) |
| `--refine_max_discrepancy_px` | `0` | Reject image-based estimates differing from metadata by more than this many pixels (0 = accept all) |
| `--refine_min_correlation` | `0.0` | Minimum NCC to accept an image-based refinement |

### linum-detect-rehoming

Detect and correct two classes of spurious inter-slice shifts in a shifts CSV:

1. **Mosaic grid expansion** (`--tile_fov_mm`): the acquisition software may add
   or remove a tile column between slices, causing `xmin_mm` to jump by ±N × tile_FOV
   even though the tissue did not move. These steps are persistent and look like
   valid re-homing events to the spike detector; correct them first.
2. **Encoder glitch spikes**: a large step immediately self-cancelled by the next
   step (no real repositioning). Detected via a return-fraction criterion and
   zeroed out, while genuine re-homing events (large step that stays) are preserved.

```bash
linum-detect-rehoming <in_shifts.csv> <out_shifts.csv> \
    [--tile_fov_mm <mm>] \
    [--tile_fov_tolerance <frac>] \
    [--return_fraction <frac>] \
    [--max_shift_mm <mm>] \
    [--report_json <report.json>] \
    [--plot <plot.png>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tile_fov_mm` | — | Tile field-of-view in mm; enables mosaic-expansion correction (legacy shifts files only) |
| `--tile_fov_tolerance` | `0.05` | Fractional tolerance around each tile-FOV multiple |
| `--return_fraction` | `0.4` | Spike sensitivity: adjacent step must reverse > (1 − return_fraction) of current step |
| `--max_shift_mm` | `0.5` | Steps below this magnitude are not checked for spikes |
| `--report_json` | — | Write detection report with per-slice decisions |
| `--plot` | — | Save a PNG comparing original and corrected shifts |

Output CSV columns include a `reliable` flag (0 when the corrected step is still
large or uncertain), consumed downstream by
`linum-align-mosaics-3d-from-shifts --refine_unreliable`.

### linum-auto-exclude-slices

Detect extended clusters of consecutive low-quality pairwise registrations and
produce a slice-config fragment listing slice IDs to force-skip (motor-only)
during stacking. Reads `pairwise_registration_metrics.json` files from each
`register_pairwise` output subdirectory.

```bash
linum-auto-exclude-slices <register_pairwise_dir> <output_slice_config.csv> \
    [--existing_slice_config <config.csv>] \
    [--consecutive <n>] \
    [--z_corr <threshold>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--existing_slice_config` | — | Merge auto-exclusion flags into an existing slice config |
| `--consecutive` | `3` | Minimum consecutive low-quality pairs to trigger exclusion |
| `--z_corr` | `0.6` | Z-correlation threshold below which a pair is low-quality |

### linum-interpolate-missing-slice

Interpolate a single missing slice from its two neighbours. Uses **z-aware
morphing** (`zmorph`) by default: an affine transform between the boundary
planes of the neighbours is computed and applied fractionally through the gap
so each output plane smoothly morphs from the before-neighbour to the after-
neighbour. When zmorph's quality gates reject the fit, the slot is left as a
genuine gap (no zarr output) and the failure is stamped into
`slice_config_final.csv`. `average` and `weighted` are available as explicit
baselines.

See {doc}`SLICE_INTERPOLATION_FEATURE` for the
physical model and parameter-tuning guidance.

```bash
linum-interpolate-missing-slice <slice_before.ome.zarr> <slice_after.ome.zarr> <output.ome.zarr> \
    [--method {zmorph,average,weighted}] \
    [--blend_method {gaussian,linear}] \
    [--registration_metric {MSE,MI,CC,AntsCC}] \
    [--max_iterations <n>] \
    [--overlap_search_window <n>] \
    [--min_overlap_correlation <0-1>] \
    [--reference_slab_size <n>] \
    [--min_foreground_fraction <0-1>] \
    [--min_ncc_improvement <val>] \
    [--manifest <fragment.csv>] \
    [--diagnostics <diag.json>]

# Finalise mode: merge per-slice manifest fragments into slice_config.csv
linum-interpolate-missing-slice --finalise \
    --slice_config_in <in.csv> --slice_config_out <out.csv> \
    --fragments_dir <manifests/>
```

### linum-export-manual-align

Export a lightweight data package (AIP images + automated transforms) for
interactive manual alignment. Consumed by the `tools/manual-align/` web tool;
outputs land in `export_manual_align/` when `--export_manual_align` is set in
the reconstruction pipeline.

```bash
linum-export-manual-align <common_space_dir> <register_pairwise_dir> <output_dir> \
    [--level <pyramid_level>] \
    [--slice_config <config.csv>]
```

### linum-refine-manual-transforms

Refine manually-corrected pairwise slice transforms with tight image-based
registration. For each pair with a manual transform, warps the moving slice
with the manual transform then runs a small-search-window registration to
correct sub-pixel / sub-degree residuals. The composed transform is emitted
with `source="manual_refined"`. Pairs without a manual transform are copied
through unchanged.

```bash
linum-refine-manual-transforms <slices_dir> <transforms_dir> <out_dir> \
    [--manual_transforms_dir <dir>] \
    [--max_translation_px <px>] \
    [--max_rotation_deg <deg>]
```

### linum-apply-slices-transforms

Apply transforms to slices.

```bash
linum-apply-slices-transforms <input_dir> <transforms_dir> <output_dir>
```

### linum-estimate-slices-transforms-gui

GUI for manual slice transform estimation.

```bash
linum-estimate-slices-transforms-gui <slices_dir>
```

### linum-register-pairwise

Perform pairwise registration between consecutive slices to compute small rotation and Z-overlap corrections. This is the primary registration script used by the motor-based reconstruction pipeline. It **does not** compute large XY translations (those are handled by motor positions from `shifts_xy.csv`).

Two outputs are produced per pair:
- `transform.tfm`: SimpleITK transform (rotation + sub-pixel translation)
- `offsets.txt`: Z-index correspondence between fixed and moving slices
- `metrics.json`: Registration quality metrics

```bash
linum-register-pairwise <fixed.ome.zarr> <moving.ome.zarr> <output_dir> \
    [--slicing_interval_mm 0.200] \
    [--search_range_mm 0.100] \
    [--moving_z_index 0] \
    [--enable_rotation | --no-enable_rotation] \
    [--max_rotation_deg 5.0] \
    [--max_translation_px 20.0] \
    [--initial_alignment {none,com,gradient,both}] \
    [--out_transform transform.tfm] \
    [--out_offsets offsets.txt] \
    [--screenshot <path>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--slicing_interval_mm` | `0.200` | Expected physical slice thickness in mm |
| `--search_range_mm` | `0.100` | Z search range around expected overlap |
| `--moving_z_index` | `0` | Starting Z-index in the moving volume |
| `--enable_rotation` | on | Enable rotation in the transform (use `--no-enable_rotation` to disable) |
| `--max_rotation_deg` | `5.0` | Maximum rotation to consider (degrees) |
| `--max_translation_px` | `20.0` | Maximum translation per axis (pixels) |
| `--initial_alignment` | `both` | Pre-registration alignment: `none`, `com`, `gradient`, or `both` |
| `--out_transform` | `transform.tfm` | Output SimpleITK transform path |
| `--out_offsets` | `offsets.txt` | Output Z-index correspondence path |
| `--screenshot` | — | Save a PNG of the registration result |

---

### linum-align-to-ras

Align a 3D brain volume to RAS orientation by rigid registration to the Allen Mouse Brain Atlas (CCF). The result is an OME-Zarr at all pyramid resolutions in RAS space.

```bash
linum-align-to-ras <input.ome.zarr> <output_ras.ome.zarr> \
    [--allen-resolution {10,25,50,100}] \
    [--metric {MI,MSE,CC,AntsCC}] \
    [--max-iterations N] \
    [--level L] \
    [--input-orientation <CODE>] \
    [--initial-rotation RX RY RZ] \
    [--preview <preview.png>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--allen-resolution` | `100` | Allen atlas resolution in µm: 10, 25, 50, 100 |
| `--metric` | `MI` | Registration metric: `MI`, `MSE`, `CC`, `AntsCC` |
| `--max-iterations` | `1000` | Maximum optimizer iterations |
| `--level` | `0` | Pyramid level for registration (0 = full) |
| `--input-orientation` | — | 3-letter orientation code (see below) |
| `--initial-rotation` | `0 0 0` | Initial rotation hint Rx Ry Rz (degrees) |
| `--preview` | — | Save a 3-panel alignment comparison image |
| `--preview-only` | — | Only generate preview without registering |
| `--store-transform-only` | — | Store transform in metadata without resampling |
| `--verbose` | — | Print registration progress |

**Orientation codes** — see the [RAS Orientation Lookup Table](NEXTFLOW_WORKFLOWS.md#ras-orientation-lookup-table) for a complete reference.

---

## Slice Configuration

### linum-generate-slice-config

Generate slice configuration file with optional galvo shift detection.

```bash
# From mosaic grids
linum-generate-slice-config <mosaics_dir> <output.csv>

# From raw tiles
linum-generate-slice-config <tiles_dir> <output.csv> --from_tiles

# From shifts file
linum-generate-slice-config <shifts.csv> <output.csv> --from_shifts

# With exclusions
linum-generate-slice-config <input> <output.csv> --exclude 1 2 5

# With galvo detection (adds galvo_confidence and galvo_fix columns)
linum-generate-slice-config <tiles_dir> <output.csv> --from_tiles --detect_galvo

# From shifts with galvo detection
linum-generate-slice-config <shifts.csv> <output.csv> --from_shifts \
    --detect_galvo --tiles_dir <tiles_dir>

# Custom galvo threshold
linum-generate-slice-config <tiles_dir> <output.csv> --from_tiles \
    --detect_galvo --galvo_threshold 0.6
```

**Galvo Detection Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--detect_galvo` | off | Enable galvo shift detection |
| `--tiles_dir` | - | Raw tiles directory (if input is shifts file) |
| `--galvo_threshold` | 0.5 | Confidence threshold for galvo fix |

---

## Diagnostics

Scripts for troubleshooting reconstruction artifacts. These are typically invoked by the pipeline's diagnostic mode but can also be run standalone.

### linum-analyze-shifts

Analyze XY shifts from a shifts file and generate a drift analysis report with summary statistics, outlier detection, and cumulative drift visualization.

```bash
linum-analyze-shifts <shifts_xy.csv> <output_dir> \
    [--resolution <um>] \
    [--iqr_multiplier <mult>] \
    [--slice_config <config.csv>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--resolution` | `10.0` | Resolution in µm/pixel (for mm → pixel conversion) |
| `--iqr_multiplier` | `1.5` | IQR multiplier for outlier detection |
| `--slice_config` | — | Optional slice config CSV to filter slices |

### linum-assess-slice-quality

Assess mosaic grid slice quality and optionally create or update a slice configuration file. Uses SSIM, edge preservation, and variance consistency metrics.

```bash
linum-assess-slice-quality <mosaics_dir> <output_slice_config.csv> \
    [--min_quality <0.0-1.0>] \
    [--exclude_first <n>] \
    [--update_existing]
```

| Option | Description |
|--------|-------------|
| `--min_quality` | Automatically exclude slices below this quality score |
| `--exclude_first` | Exclude the first N calibration slices |
| `--update_existing` | Update an existing slice config with quality info |

### linum-analyze-registration-transforms

Analyze cumulative rotation and translation across pairwise registration transforms to detect drift.

```bash
linum-analyze-registration-transforms <register_pairwise_dir> <output_dir> \
    [--resolution <um>] \
    [--rotation_threshold <degrees>]
```

### linum-analyze-acquisition-rotation

Analyze acquisition-time rotation from the shifts file combined with registration outputs.

```bash
linum-analyze-acquisition-rotation <shifts_file> <output_dir> \
    [--registration_dir <register_pairwise_dir>] \
    [--resolution <um>]
```

### linum-analyze-tile-dilation

Analyze tile position refinements to detect scale drift (mosaic dilation).

```bash
linum-analyze-tile-dilation <mosaic_grid.ome.zarr> <transform.npy> <output_dir> \
    [--resolution <um>] \
    [--overlap_fraction <frac>] \
    [--slice_id <id>]
```

### linum-aggregate-dilation-analysis

Aggregate per-slice tile dilation analysis results across the full sample.

```bash
linum-aggregate-dilation-analysis <input_dir> <output_dir> \
    [--pattern <glob_pattern>]
```

### linum-compare-stitching

Compare motor-only vs refined stitching side-by-side by computing seam sharpness metrics and generating comparison visualizations.

```bash
linum-compare-stitching <motor_stitch.ome.zarr> <refined_stitch.ome.zarr> <output_dir> \
    [--label1 <name>] \
    [--label2 <name>] \
    [--tile_step <pixels>]
```

### linum-diagnose-pipeline

High-level pipeline diagnostic script. Aggregates metrics and produces a summarized diagnostic report.

```bash
linum-diagnose-pipeline <pipeline_output_dir> <output_dir> \
    [--resolution <um>]
```

### linum-diagnose-reconstruction

Detailed reconstruction diagnostic script. Checks registration transforms, rotation drift, and alignment quality.

```bash
linum-diagnose-reconstruction <pipeline_output_dir> <output_dir> \
    [--resolution <um>] \
    [--rotation_threshold <degrees>]
```

---

## Visualization

### linum-view-omezarr

Interactive OME-Zarr viewer.

```bash
linum-view-omezarr <input.ome.zarr>
```

### linum-view-zarr

Interactive Zarr viewer.

```bash
linum-view-zarr <input.zarr>
```

### linum-view-oct-raw-tile

View raw OCT tile.

```bash
linum-view-oct-raw-tile <tile_dir>
```

### linum-screenshot-omezarr

Generate screenshot from OME-Zarr.

```bash
linum-screenshot-omezarr <input.ome.zarr> <output.png>
```

### linum-screenshot-omezarr-annotated

Generate annotated orthogonal view screenshots from an OME-Zarr volume. Adds Z-slice index labels to the coronal and sagittal views so each input slice can be easily identified in the reconstruction.

```bash
linum-screenshot-omezarr-annotated <input.ome.zarr> <output.png> \
    [--x_slice <idx>] \
    [--y_slice <idx>] \
    [--n_slices <n>] \
    [--slice_ids <id1,id2,...>] \
    [--font_size <size>] \
    [--label_every <n>] \
    [--show_lines]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--x_slice` | center | X-axis slice index for ZY view |
| `--y_slice` | center | Y-axis slice index for ZX view |
| `--n_slices` | auto | Number of input slices (auto-detected from OME-Zarr metadata) |
| `--slice_ids` | — | Comma-separated actual slice IDs (e.g. `05,12,18`) |
| `--font_size` | `7` | Font size for slice labels |
| `--label_every` | `1` | Label every Nth slice |
| `--show_lines` | off | Draw horizontal lines at slice boundaries |

### linum-generate-pipeline-report

Generate a quality report from pipeline metrics. Aggregates metrics JSON files from all processing steps and produces an HTML or text report.

```bash
# Generate HTML report (default)
linum-generate-pipeline-report <pipeline_output_dir> report.html

# Generate text report
linum-generate-pipeline-report <pipeline_output_dir> report.txt --format text

# Verbose report with all metric details
linum-generate-pipeline-report <pipeline_output_dir> report.html --verbose

# Custom title
linum-generate-pipeline-report <pipeline_output_dir> report.html --title "My Pipeline Report"
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


### linum-clean-raw-data

Clean up raw S-OCT acquisitions by removing processed `.bin` files and keeping
only the metadata (`metadata.json`, `info.txt`). Moves quick-stitch images to
`quick_stitches/` and slice directories to `metadata/`, preserving the overall
directory structure. Useful for archiving subjects once mosaic grids have been
generated.

```bash
linum-clean-raw-data <data_directory> [--dry-run] [-v]
```

### linum-fix-galvo-shift-zarr

Fix galvo-shift artefacts directly on an already-assembled mosaic grid
OME-Zarr when the raw `.bin` files are no longer available. Each zarr chunk
corresponds to one OCT tile, so the same dark-band detector used for raw tiles
is applied to a sample of chunks, and the fix is a circular roll (`np.roll`)
of each chunk. `--mode undo` reverses a previously applied fix (for false-
positive detections).

```bash
linum-fix-galvo-shift-zarr <input.ome.zarr> <output.ome.zarr> \
    [--detect_only] \
    [--mode {apply,undo}] \
    [--band_shift <pixels>] \
    [--band_width <pixels>] \
    [--n_pixel_return <n>] \
    [--galvo_threshold <0-1>]
```

### linum-extract-pyramid-levels

Extract one or more pyramid levels from an OME-Zarr volume and save as NIfTI files. Useful for exporting analysis-specific resolutions without converting the full volume.

```bash
# List available pyramid levels
linum-extract-pyramid-levels <input.ome.zarr> --list

# Extract specific levels
linum-extract-pyramid-levels <input.ome.zarr> 0 2
```

Output files are named `<zarr_stem>_level<N>_<resolution>.nii.gz` and saved next to the input.

### linum-resample-nifti

Resample a NIfTI image to a target isotropic resolution.

```bash
linum-resample-nifti <input.nii.gz> <output.nii.gz> -r <resolution>
```

For resampling 2D mosaic grids see `linum-resample-mosaic-grid`.

### linum-reorient-nifti-to-ras

Reorient NIfTI to RAS orientation.

```bash
linum-reorient-nifti-to-ras <input.nii.gz> <output.nii.gz>
```

### linum-axis-xyz-to-zyx

Transpose axes from XYZ to ZYX.

```bash
linum-axis-xyz-to-zyx <input> <output>
```

### linum-segment-brain-3d

Segment brain tissue in 3D volume.

```bash
linum-segment-brain-3d <input> <output_mask>
```

### linum-download-allen

Download Allen Brain Atlas data.

```bash
linum-download-allen <output_dir>
```

### linum-merge-slices-into-folders

Organize slices into folder structure.

```bash
linum-merge-slices-into-folders <input_dir> <output_dir>
```

### linum-suggest-params

Suggest 3D reconstruction pipeline parameters from raw input files. Analyses the motor-positions file (`shifts_xy.csv`) and, optionally, the raw data directory produced by the preprocessing pipeline to automatically estimate suitable `nextflow.config` parameters.

```bash
linum-suggest-params <shifts_file> <output_dir> \
    [--data_dir <raw_data_dir>] \
    [--n_calibration_slices N] \
    [--axial_res_um UM] \
    [--resolution_um UM] \
    [-f]
```

| Option | Default | Description |
|--------|---------|-------------|
| `shifts_file` | (required) | Motor-positions CSV file (`shifts_xy.csv`) |
| `output_dir` | (required) | Directory for the report and suggested config snippet |
| `--data_dir` | — | Raw data directory (contains `state.json` and `slice_z##/` subdirectories) |
| `--n_calibration_slices` | `1` | Leading calibration slices to skip when reading per-slice metadata (`slice_z00` is always calibration) |
| `--axial_res_um` | `3.5` | OCT axial resolution in µm/pixel |
| `--resolution_um` | — | Override target pipeline resolution in µm/pixel (derived automatically if omitted) |
| `-f`, `--overwrite` | — | Overwrite existing output directory |

**Estimated parameters:**

From `shifts_xy.csv`:
- `max_shift_mm` — IQR upper bound of normal inter-slice shifts (used for `rehoming_max_shift_mm`)
- `common_space_max_step_mm` — 95th percentile of consecutive normal shift changes (used for `common_space_excluded_slice_mode` tuning)
- `interpolate_missing_slices` — suggested based on gap pattern in shift data

From `--data_dir` (raw data directory):
- `registration_slicing_interval_mm` — from `slice_thickness` in `metadata.json` / `state.json`
- `stitch_overlap_fraction` — from `overlap_fraction` in `metadata.json` / `state.json`
- `resolution` — smallest standard resolution ≥ native lateral pixel size
- `crop_interface_out_depth` — estimate based on OCT depth and focus position (must be verified)

**Outputs:**
- `param_estimation_report.txt` — human-readable analysis report with shift statistics
- `suggested_params.config` — annotated `nextflow.config` parameter block ready to copy-paste

**Example:**
```bash
linum-suggest-params /data/sub-01/shifts_xy.csv /tmp/sub-01_params \
    --data_dir /data/sub-01/raw \
    --n_calibration_slices 1
```

---

## GPU-Accelerated Scripts

GPU-accelerated versions of key scripts. These use NVIDIA CUDA via CuPy for acceleration and automatically fall back to CPU if no GPU is available.

### linum-gpu-info

Check GPU availability and run quick benchmarks.

```bash
# Show GPU info
linum-gpu-info

# Run quick performance test
linum-gpu-info --test

# Output as JSON
linum-gpu-info --json
```

### linum-benchmark-gpu

Comprehensive CPU vs GPU benchmark suite.

```bash
# Quick benchmark (512x512)
linum-benchmark-gpu

# Full benchmark with multiple sizes
linum-benchmark-gpu --full

# Custom sizes
linum-benchmark-gpu --sizes 1024 2048 4096

# With real data
linum-benchmark-gpu --input /path/to/mosaic.ome.zarr

# Save results
linum-benchmark-gpu --output results.json --iterations 10
```

| Option | Description |
|--------|-------------|
| `--input` | Path to OME-Zarr for real-data benchmark |
| `--output, -o` | Save results to JSON file |
| `--iterations, -n` | Iterations per test (default: 3) |
| `--full` | Run with multiple sizes |
| `--sizes` | Custom sizes to test |
| `--skip-correctness` | Skip result verification |

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
linum-create-mosaic-grid-3d --help
```

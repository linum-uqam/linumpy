# 2.5D Reconstruction Workflow

---

## Overview

The 2.5D reconstruction workflow (`soct_2.5d_reconst.nf`) converts a set of per-slice 2D mosaic
grids (TIFF format) into a stacked 3D volume using illumination correction, image-based tile
placement, and XY-shift-based stacking.

This is the **legacy** workflow. New acquisitions should use the full
[3D Reconstruction Workflow](NEXTFLOW_WORKFLOWS.md#3d-reconstruction-workflow)
(`soct_3d_reconst.nf`), which operates on OME-Zarr mosaic grids and provides much richer
correction and diagnostics.

---

## When to Use This Workflow

- Data acquired as TIFF-format mosaic grids (`mosaic_grid_z*.tiff`)
- Re-processing older datasets that pre-date the OME-Zarr pipeline
- Quick 2D stitching + stacking without the full 3D pipeline overhead

---

## Location

```
workflows/reconst_2.5d/
├── soct_2.5d_reconst.nf           # Workflow definition
├── soct_2.5d_reconst_beluga.config # Compute Canada / Beluga cluster config
└── soct_2.5d_reconst_docker.config # Docker-based config
```

---

## Input

| Item | Description |
|------|-------------|
| `{directory}/mosaicgrids/` | Directory of `mosaic_grid_z*.tiff` files, one per slice |
| `{directory}/shifts_xy.csv` | XY inter-slice shifts file (standard linumpy format) |

---

## Running

```bash
# Basic usage (run from workflow directory or with full path)
nextflow run soct_2.5d_reconst.nf \
    --directory /path/to/subject

# With custom output directory and resolution
nextflow run soct_2.5d_reconst.nf \
    --directory /path/to/subject \
    --output_directory /path/to/output

# Resume after partial run
nextflow run soct_2.5d_reconst.nf \
    --directory /path/to/subject \
    -resume
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `directory` | `"."` | Root subject directory |
| `input_directory` | `{directory}/mosaicgrids` | Directory containing `mosaic_grid_z*.tiff` files |
| `xy_shift_file` | `{directory}/shifts_xy.csv` | Inter-slice XY shifts file |
| `output_directory` | `{directory}` | Output directory |

### Tile Shape

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_nx` | `400` | Tile width in pixels |
| `tile_ny` | `400` | Tile height in pixels |
| `spacing_xy` | `1.875` | Lateral pixel spacing (µm) |
| `spacing_z` | `200.0` | Axial (slice) spacing (µm) |

### Tile Cropping

A border is removed from each tile before processing to avoid edge artifacts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `xmin` | `10` | Left crop (pixels) |
| `xmax` | `390` | Right crop (pixels); effective tile width = `xmax - xmin` = 380 px |
| `ymin` | `10` | Top crop (pixels) |
| `ymax` | `390` | Bottom crop (pixels); effective tile height = `ymax - ymin` = 380 px |

### Illumination Bias

| Parameter | Default | Description |
|-----------|---------|-------------|
| `illum_n_samples` | `512` | Tiles sampled to estimate flat-field |
| `pos_n_samples` | `512` | Tiles sampled to estimate dark-field |
| `basic_working_size` | `128` | Internal BaSIC working resolution (pixels) |

### Stitching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_overlap` | `0.2` | Expected tile overlap fraction for position estimation |

### Output Resolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution_nifti` | `10.0` | Isotropic resolution for the resampled NIfTI output (µm) |

---

## Pipeline Processes

The workflow runs processes in a linear sequence:

```
crop_tiles
    └─→ estimate_illumination_bias
             └─→ compensate_illumination_bias
                       │
                       ├─→ estimate_position (pools all compensated grids)
                       │
                       └─→ stitch_mosaic (per-slice, uses shared position transform)
                                  └─→ stack_mosaic
                                            ├─→ compress_stack  → stack.zarr.zip
                                            ├─→ convert_to_omezarr
                                            │         └─→ resample_stack → stack_10um.nii.gz
                                            └─→ (stack.zarr itself)
```

### 1. `crop_tiles`

Crops a border from each tile within the mosaic grid TIFF to remove edge artifacts.

```
linum_crop_tiles.py <mosaic_dir> <output.tiff> --xmin --xmax --ymin --ymax --tile_shape
```

**Input**: `mosaic_grid_z*.tiff` directory  
**Output**: `{basename}_cropped.tiff`

### 2. `estimate_illumination_bias`

Estimates per-tile flat-field and dark-field using the BaSIC algorithm.

```
linum_estimate_illumination.py <mosaic.tiff> <flatfield.nii.gz> --tile_shape --output_darkfield
```

**Input**: Cropped mosaic grid (per slice)  
**Output**: `{key}_flatfield.nii.gz`, `{key}_darkfield.nii.gz`

### 3. `compensate_illumination_bias`

Applies the estimated flat/dark field correction to each mosaic grid.

```
linum_compensate_illumination.py <mosaic.tiff> <output.nii.gz> --flatfield --darkfield --tile_shape
```

**Input**: Cropped mosaic + flat/dark field  
**Output**: `{key}_mosaic_grid_compensated.nii.gz`

### 4. `estimate_position`

Pools all compensated mosaic grids to estimate a single shared tile-placement transform (`.npy`).
This single transform is applied to all slices, avoiding per-slice jitter.

```
linum_estimate_transform.py <all_mosaics...> <position_transform.npy> --tile_shape --initial_overlap
```

**Input**: All compensated mosaic grids (collected)  
**Output**: `position_transform.npy`

### 5. `stitch_mosaic`

Stitches each compensated mosaic grid into a 2D slice using the shared position transform.
Blending method is `diffusion`.

```
linum_stitch_2d.py <mosaic.nii.gz> <transform.npy> <output.nii.gz> --blending_method diffusion --tile_shape
```

**Input**: Compensated mosaic (per slice) + position transform  
**Output**: `{key}_stitched.nii.gz`

### 6. `stack_mosaic`

Stacks all stitched 2D slices into a 3D volume using XY shifts from `shifts_xy.csv`.

```
linum_stack_slices.py <all_stitched...> stack.zarr --xy_shifts --resolution_xy --resolution_z
```

**Input**: All stitched slices (collected) + shifts CSV  
**Output**: `stack.zarr`

### 7. `compress_stack`

Compresses the Zarr stack to a ZIP archive for transfer.

**Input**: `stack.zarr`  
**Output**: `stack.zarr.zip`

### 8. `convert_to_omezarr`

Converts the Zarr stack to OME-Zarr format for visualization in napari/neuroglancer.

**Input**: `stack.zarr`  
**Output**: `stack.ome.zarr`

### 9. `resample_stack`

Resamples the OME-Zarr to isotropic resolution and exports a NIfTI file.

```
linum_convert_omezarr_to_nifti.py stack.ome.zarr stack_10um.nii.gz --resolution 10.0
```

**Input**: `stack.ome.zarr`  
**Output**: `stack_10um.nii.gz`

---

## Outputs

| File | Description |
|------|-------------|
| `stack.zarr` | Full-resolution 3D volume (Zarr format) |
| `stack.zarr.zip` | Compressed archive of `stack.zarr` |
| `stack.ome.zarr` | OME-Zarr for visualization |
| `stack_10um.nii.gz` | Isotropic 10 µm NIfTI for atlas registration and analysis |

---

## Cluster / Container Configs

| Config | Use case |
|--------|----------|
| `soct_2.5d_reconst_beluga.config` | Compute Canada Beluga cluster (SLURM) |
| `soct_2.5d_reconst_docker.config` | Docker container execution |

```bash
# Beluga cluster
nextflow run soct_2.5d_reconst.nf \
    --directory /path/to/subject \
    -c soct_2.5d_reconst_beluga.config

# Docker
nextflow run soct_2.5d_reconst.nf \
    --directory /path/to/subject \
    -c soct_2.5d_reconst_docker.config
```

---

## Differences from the 3D Workflow

| Aspect | 2.5D workflow | 3D workflow |
|--------|---------------|-------------|
| Input format | TIFF mosaic grids | OME-Zarr mosaic grids |
| Tile illumination | BaSIC (per-slice) | BaSIC (per-slice via `fix_illumination`) |
| Tile placement | Image-based (phase correlation) | Motor positions + optional image refinement |
| Slice alignment | Motor XY shifts only | Motor + pairwise registration |
| Quality assessment | None | Optional auto quality assessment |
| Re-homing correction | None | `detect_rehoming` pass |
| Interpolation | None | `interpolate_missing_slice` |
| Atlas registration | None (requires separate NIfTI step) | Integrated `align_to_ras` |
| Output | `stack.zarr`, `stack_10um.nii.gz` | `{subject}.ome.zarr` (multi-resolution) |

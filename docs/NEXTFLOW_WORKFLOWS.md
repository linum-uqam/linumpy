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
| `shifts_xy` | `"$input/shifts_xy.csv"` | XY shifts file |
| `slice_config` | `""` | Optional slice config |
| `output` | `"."` | Output directory |
| `processes` | `1` | Parallel processes |

#### Processing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | `10` | Target resolution (µm/pixel) |
| `clip_percentile_upper` | `99.9` | Upper percentile for clipping |
| `fix_curvature_enabled` | `true` | Fix focal curvature |
| `fix_illum_enabled` | `true` | Fix illumination |
| `crop_interface_out_depth` | `600` | Crop depth (µm) |

#### Registration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `create_registration_masks` | `true` | Create masks |
| `mask_smoothing_sigma` | `5.0` | Mask smoothing (µm) |
| `selem_radius` | `1` | Morphological element radius |
| `min_size` | `100` | Minimum mask component size |
| `mask_normalize` | `true` | Normalize before masking |
| `moving_slice_first_index` | `4` | Skip voxels from top |
| `registration_transform` | `'affine'` | Transform type (`affine`, `euler`, `translation`) |
| `registration_metric` | `'MSE'` | Registration metric (`MSE`, `CC`, `MI`) |
| `registration_max_translation` | `50.0` | Max allowed translation (pixels) |
| `registration_max_rotation` | `2.0` | Max allowed rotation (degrees) |
| `registration_robustness` | `'normal'` | Robustness level (`conservative`, `normal`, `aggressive`) |

**Robustness Levels:**

The `registration_robustness` parameter controls fallback strategies:
- `conservative`: Trust microscope positions, minimal corrections, uses identity for large translations
- `normal`: Standard fallback chain (phase correlation → mask fallback → translation-only)
- `aggressive`: All fallbacks + accepts consistent large translations


#### Shift Outlier Filtering (Common Space)

These parameters control outlier detection and filtering for the `bring_to_common_space` step. The shifts file from the microscope may contain erroneous large shifts that cause slices to drift out of alignment.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_shift_outliers` | `true` | Enable outlier filtering (strongly recommended) |
| `outlier_method` | `'iqr'` | Detection method: `clamp`, `median`, `zero`, `local`, or `iqr` |
| `max_shift_mm` | `0.5` | Max shift threshold in mm (only if method != 'iqr') |
| `outlier_iqr_multiplier` | `1.5` | IQR multiplier for outlier detection (only with 'iqr' method) |

**Outlier Methods:**
- `iqr` (recommended): Auto-detect outliers using IQR statistics, replace with local median
- `local`: Replace outliers with local median of neighboring shifts
- `median`: Replace outliers with global median
- `clamp`: Limit magnitude to `max_shift_mm` while preserving direction
- `zero`: Replace outliers with zero shift

#### Debugging Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `analyze_shifts` | `false` | Generate shifts analysis report and drift plots |
| `mask_preview` | `false` | Generate preview images for masks |
| `common_space_preview` | `true` | Generate preview images after common space alignment |

The `analyze_shifts` option runs drift analysis on the shifts file before processing, producing:
- A text report with statistics and outlier detection
- A PNG plot showing drift patterns
- A filtered shifts CSV file

#### Stacking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stack_blend_enabled` | `false` | Enable blending |
| `stack_max_overlap` | `-1` | Max overlap (-1 = all) |

#### Pyramid Resolutions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pyramid_resolutions` | `[10, 25, 50, 100]` | Target resolutions (µm) for output pyramid levels |

The `pyramid_resolutions` parameter controls the multi-resolution pyramid in the final 3D volume. Instead of power-of-2 downsampling, specific analysis-friendly resolutions are used:

- **10 µm**: High-resolution analysis
- **25 µm**: Standard analysis resolution  
- **50 µm**: Overview and atlas registration
- **100 µm**: Quick visualization and large-scale analysis

**Note:** Only resolutions ≥ the base `resolution` parameter will be included. For example, if `resolution = 25`, then only 25, 50, and 100 µm levels will be created.

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
└── stack/
    ├── 3d_volume.ome.zarr
    ├── 3d_volume.ome.zarr.zip
    └── 3d_volume.png
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

# GPU Acceleration


## Overview

linumpy supports GPU acceleration for compute-intensive operations using NVIDIA CUDA via CuPy. GPU acceleration is **optional** - all functions automatically fall back to CPU (NumPy/SciPy) if:

- CuPy is not installed
- No CUDA-capable GPU is available
- GPU memory is insufficient

```mermaid
flowchart TD
    CALL[GPU-aware function called<br/>backend='auto' default] --> CHK1{CuPy installed?}
    CHK1 -->|no| CPU[Run on CPU<br/>NumPy / SciPy / SimpleITK]
    CHK1 -->|yes| CHK2{CUDA device available?}
    CHK2 -->|no| CPU
    CHK2 -->|yes| PICK[Auto-select least-loaded GPU]
    PICK --> RUN[Run CuPy kernel on device]
    RUN -->|OOM / runtime error| CPU
    RUN -->|success| OUT([Result])
    CPU --> OUT
```

backend selection is per-call (`backend="cpu" | "gpu" | "auto"`); the auto path is the safe default and is what the Nextflow workflows use when `use_gpu=true`.

---

## Quick Start

```bash
# Check your CUDA version
nvidia-smi | grep "CUDA Version"

# Install linumpy with GPU support (choose your CUDA version)
uv pip install 'linumpy[gpu]'           # CUDA 13.x (default)
uv pip install 'linumpy[gpu-cuda12]'    # CUDA 12.x

# Verify GPU
linum_gpu_info.py
linum_diagnose_pipeline.py --benchmark
```

---

## Installation

### Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0+
- CUDA Toolkit 11.x, 12.x, or 13.x
- CuPy matching your CUDA version

**Recommended GPU:** NVIDIA A6000 (48GB) or similar professional GPU.

### CuPy Version Reference

| CUDA Version | CuPy Package | linumpy extra |
|--------------|--------------|---------------|
| CUDA 13.x    | `cupy-cuda13x` | `linumpy[gpu]` (default) |
| CUDA 12.x    | `cupy-cuda12x` | `linumpy[gpu-cuda12]` |

---

## BaSiCPy (fix_illumination)

The `fix_illumination` step uses BaSiCPy 2.x, which now ships with a
**PyTorch backend** (no JAX). BaSiCPy will use a CUDA-enabled PyTorch wheel
automatically when one is installed; otherwise it runs on CPU.

If you only need linumpy's CuPy paths (resampling, FFT, morphology, N4),
no extra steps beyond `pip install 'linumpy[gpu]'` are required. To enable
GPU acceleration of BaSiCPy as well, install a CUDA build of PyTorch:

```bash
# Pick the index URL that matches your CUDA toolkit
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
linum_gpu_info.py
linum_diagnose_pipeline.py --benchmark
```

---

## GPU-Accelerated Scripts

All scripts now accept `--use_gpu` / `--no-use_gpu` (default: `--use_gpu`).
GPU acceleration is enabled automatically when a CUDA device is detected; no
separate `_gpu.py` variant is needed.

| Script | GPU-accelerated operation | Typical Speedup |
|--------|--------------------------|-----------------|
| `linum_estimate_transform.py` | FFT / phase correlation | 8-47x |
| `linum_create_mosaic_grid_3d.py` | Volume resize | 5-12x |
| `linum_resample_mosaic_grid.py` | Volume resize | 5-12x |
| `linum_normalize_intensities_per_slice.py` | Gaussian filter, Otsu threshold | 4-10x |
| `linum_fix_illumination_3d.py` | BaSiCPy via PyTorch/CUDA | 2-5x |
| `linum_assess_slice_quality.py` | SSIM, morphology | 3-8x |
| `linum_aip_png.py` | Mean projection | ≤1x |
| `linum_generate_mosaic_aips.py` | Mean projection | ≤1x |
| `linum_correct_bias_field.py` | N4 bias field estimation | varies |
| `linum_estimate_global_transform.py` | Phase correlation | 8-16x |

---

## GPU-Accelerated Operations

### Major Improvements (7-70x speedup)

| Operation | Function | Typical Speedup |
|-----------|----------|-----------------|
| Binary Morphology | `binary_closing()`, etc. | 7-67x |
| FFT/iFFT | `fft2()`, `ifft2()` | 9-47x |
| Gaussian Filter | `gaussian_filter()` | 7-20x |
| Phase Correlation | `phase_correlation()` | 8-16x |
| Resampling | `resize()` | 5-12x |

### Medium Improvements (4-10x speedup)

| Operation | Typical Speedup |
|-----------|-----------------|
| Normalization | 4-10x |
| Percentile Clipping | 4-10x |
| Interpolation | 5-10x |
| Mask Creation | 2-4x |

### No GPU Benefit (use CPU)

| Operation | Reason |
|-----------|--------|
| Mean/Max Projection | Simple reduction, transfer overhead dominates |
| Galvo Detection | Simple computation |

---

## Multi-GPU Systems

```bash
# Show status of all GPUs
linum_gpu_info.py --status

# Select best GPU (most free memory)
linum_gpu_info.py --select-best

# Use specific GPU via environment
CUDA_VISIBLE_DEVICES=1 nextflow run pipeline.nf --use_gpu true
```

---

## Memory Management

The NVIDIA A6000's 48GB VRAM typically holds entire mosaic grids. For larger volumes:

```python
import cupy as cp

# Clear GPU memory cache
cp.get_default_memory_pool().free_all_blocks()

# Check memory usage
mempool = cp.get_default_memory_pool()
print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
```

---

## Troubleshooting

### GPU Not Detected

```bash
nvidia-smi
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
linum_gpu_info.py
```

### BaSiCPy / PyTorch CUDA Issues

If `linum_fix_illumination_3d.py` falls back to CPU unexpectedly, verify the
PyTorch CUDA build is installed and visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
linum_diagnose_pipeline.py --debug-cuda
```

Reinstall PyTorch from the matching CUDA index URL (see the BaSiCPy section
above) if `torch.cuda.is_available()` returns `False`.

### Out of Memory

- Reduce batch size / chunk size
- Use `cp.get_default_memory_pool().free_all_blocks()` between operations
- Set `use_gpu=False` for specific operations

---

## Reference Benchmarks

Actual benchmarks on NVIDIA RTX A6000 (48GB):

| Operation | Image Size | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| FFT2 | 2048×2048 | 207ms | 4.4ms | 47x |
| Phase Correlation | 2048×2048 | 3876ms | 240ms | 16x |
| Gaussian Filter | 2048×2048 | 79ms | 4.1ms | 20x |
| Binary Closing | 2048×2048 | 99ms | 1.5ms | 67x |
| Resize | 2048→1024 | 29ms | 3.4ms | 9x |
| Mean Projection | 100×2048×2048 | 71ms | 146ms | **0.5x** |

---

## Module Reference

```python
from linumpy.gpu import (
    GPU_AVAILABLE,      # bool: Is GPU available?
    CUPY_AVAILABLE,     # bool: Is CuPy installed?
    GPU_DEVICE_NAME,    # str: GPU name
    GPU_MEMORY_GB,      # float: GPU memory in GB
    to_gpu, to_cpu,     # Transfer functions
    print_gpu_info,     # Print info
)

# Submodules
from linumpy.gpu.fft_ops import fft2, ifft2, phase_correlation
from linumpy.gpu.interpolation import resize, affine_transform
from linumpy.gpu.morphology import binary_closing, gaussian_filter
from linumpy.gpu.array_ops import normalize_percentile, clip_percentile
from linumpy.gpu.zarr_io import read_zarr_to_gpu
```

---

## Fast zarr → GPU loading

For zarr arrays on local NVMe, `linumpy.gpu.zarr_io.read_zarr_to_gpu` is the recommended entry point. It dispatches to the fastest backend available at runtime:

```python
from linumpy.gpu.zarr_io import read_zarr_to_gpu

dev = read_zarr_to_gpu("/scratch_nvme/volume.zarr")
# dev is a cupy.ndarray
```

Backend implementations live in their own modules:

- `linumpy.gpu.kvikio_zarr` — kvikio / GPUDirect Storage reader (uncompressed zarr v2/v3 only).

Selection order (when `prefer="auto"`):

1. **kvikio (GPUDirect Storage, native mode)** — chunks DMA'd directly from NVMe into GPU memory. Fastest. Requires `kvikio` installed, GDS native mode enabled (`/etc/cufile.json`: `allow_compat_mode=false`), and an uncompressed zarr.
2. **`zarr.config.enable_gpu()`** — host I/O with on-host decode, single H→D copy. Works for any zarr (compressed or not) and is the automatic fallback when GDS is unavailable, in compat mode, or the array is compressed.

You can force a specific path with `prefer="kvikio"` or `prefer="zarr-gpu"`. The legacy `cupy.asarray(zarr.open_array(...)[:])` path is kept only as a reference baseline in `linum_benchmark_kvikio_zarr.py`.

### Reference benchmark

`scripts/linum_benchmark_kvikio_zarr.py` measures all three paths. On a 16 GiB float32 zarr v3 (256³ chunks) on local NVMe ext4 with an RTX A6000:

| Path | Cold | Warm |
|---|---|---|
| kvikio (GDS native) | 8.2 GiB/s | **9.9 GiB/s** |
| `zarr.config.enable_gpu()` | 6.3 GiB/s | 7.1 GiB/s |
| `zarr → numpy → cupy.asarray` | 1.1 GiB/s | 2.8 GiB/s |

In compat mode (GDS bounce-buffer), kvikio drops to ~4 GiB/s — slower than the `zarr-gpu` path. The auto selector detects this via `kvikio.defaults.compat_mode()` and prefers `zarr-gpu` in that case.

### Installing the GDS path

```bash
# CuPy + linumpy GPU support
uv pip install 'linumpy[gpu]'           # CUDA 13.x (default)
uv pip install 'linumpy[gpu-cuda12]'    # CUDA 12.x

# kvikio (optional, only needed for the GDS fast path)
uv pip install 'linumpy[gds]'           # CUDA 13.x (default)
uv pip install 'linumpy[gds-cuda12]'    # CUDA 12.x
```

### Enabling native GDS

Native GDS additionally requires:

- A CUDA-aware filesystem on the source path (ext4 / xfs on local NVMe is fine; NFS, overlayfs, encrypted FS are not).
- `/etc/cufile.json`: `properties.use_compat_mode = false`.
- IOMMU disabled or in passthrough (`amd_iommu=off iommu=off` on AMD; `intel_iommu=off` on Intel).
- nvidia-fs DKMS module loaded with matching ABI; verify `dmesg | grep nvidia_fs` shows no "no extended symbol version" warnings after driver/kernel updates.

If any of these are missing, kvikio silently falls back to a POSIX bounce-buffer (compat mode) and `read_zarr_to_gpu` will route around it.

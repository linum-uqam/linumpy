# GPU Acceleration


---

## Overview

linumpy supports GPU acceleration for compute-intensive operations using NVIDIA CUDA via CuPy. GPU acceleration is **optional** - all functions automatically fall back to CPU (NumPy/SciPy) if:

- CuPy is not installed
- No CUDA-capable GPU is available
- GPU memory is insufficient

---

## Quick Start

```bash
# Check your CUDA version
nvidia-smi | grep "CUDA Version"

# Install linumpy with GPU support (choose your CUDA version)
pip install linumpy[gpu]           # CUDA 12.x (default)
pip install linumpy[gpu-cuda11]    # CUDA 11.x
pip install linumpy[gpu-cuda13]    # CUDA 13.x (requires extra setup for JAX)

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

| CUDA Version | CuPy Package |
|--------------|--------------|
| CUDA 11.x | `cupy-cuda11x` |
| CUDA 12.x | `cupy-cuda12x` |
| CUDA 13.x | `cupy-cuda13x` |

---

## JAX GPU for BaSiCPy (fix_illumination)

The `fix_illumination` step uses BaSiCPy which is built on JAX. JAX GPU requires additional setup.

### Important: JAX 0.4.23 Library Requirements

BaSiCPy requires `jax<=0.4.23`. JAX 0.4.23 was compiled against specific library versions:
- cuSOLVER 11 (libcusolver.so.11)
- cuSPARSE 12 (libcusparse.so.12)
- cuFFT 11 (libcufft.so.11)
- cuBLAS 12 (libcublas.so.12)
- cuDNN 8 (libcudnn.so.8)

These exact versions are only available in **specific pinned versions** of the `nvidia-xxx-cu12` packages. Newer versions of these packages have different `.so` versions that are **incompatible**.

### Automated Setup (Recommended)

```bash
# Run the fix script - handles everything
source shell_scripts/fix_jax_cuda_plugin.sh
```

This script:
1. Removes conflicting nvidia packages
2. Installs JAX 0.4.23 with **pinned nvidia package versions**:
   - `nvidia-cublas-cu12==12.3.4.1`
   - `nvidia-cudnn-cu12==8.9.7.29`
   - `nvidia-cusolver-cu12==11.5.4.101`
   - etc.
3. Applies patchelf fix (required for Linux 6.x+ kernels)
4. Sets up LD_LIBRARY_PATH
5. Tests JAX CUDA with SVD operation

### Manual Setup

If you prefer manual setup:

```bash
# 1. Uninstall all conflicting packages
pip uninstall -y jax jaxlib jax-cuda12-plugin nvidia-cusolver nvidia-cufft \
    nvidia-cusparse nvidia-cublas nvidia-cuda-runtime nvidia-cudnn nvidia-nvjitlink \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 nvidia-nvjitlink-cu12

# 2. Install JAX 0.4.23 with CUDA wheel
pip install 'jax==0.4.23' 'jaxlib==0.4.23+cuda12.cudnn89' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install PINNED nvidia package versions (critical - newer versions won't work!)
pip install \
    'nvidia-cublas-cu12==12.3.4.1' \
    'nvidia-cuda-cupti-cu12==12.3.101' \
    'nvidia-cuda-runtime-cu12==12.3.101' \
    'nvidia-cudnn-cu12==8.9.7.29' \
    'nvidia-cufft-cu12==11.0.12.1' \
    'nvidia-cusolver-cu12==11.5.4.101' \
    'nvidia-cusparse-cu12==12.2.0.103' \
    'nvidia-nccl-cu12==2.19.3' \
    'nvidia-nvjitlink-cu12==12.3.101'

# 4. Apply patchelf fix (required for modern Linux kernels)
sudo apt install patchelf
JAXLIB_PATH=$(python -c "import jaxlib; print(jaxlib.__path__[0])")
find "$JAXLIB_PATH" -name "*.so" -exec patchelf --clear-execstack {} \;
find $(python -c "import site; print(site.getsitepackages()[0])")/jax_plugins \
    -name "*.so" -exec patchelf --clear-execstack {} \;

# 5. Set LD_LIBRARY_PATH (before running JAX/BaSiCPy)
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SP}/nvidia/cublas/lib:${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/cusolver/lib:${SP}/nvidia/cusparse/lib:${SP}/nvidia/cufft/lib:${SP}/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# 6. Test
python -c "import jax; print(jax.devices()); import jax.numpy as jnp; print(jnp.linalg.svd(jnp.eye(2)))"
```

### Verify Installation

```bash
# Check GPU availability
linum_gpu_info.py

# Run full pipeline diagnostics
linum_diagnose_pipeline.py --benchmark
```

---

## GPU-Accelerated Scripts

| GPU Script | CPU Equivalent | Typical Speedup |
|------------|----------------|-----------------|
| `linum_estimate_transform_gpu.py` | `linum_estimate_transform.py` | 8-47x |
| `linum_create_masks_gpu.py` | `linum_create_masks.py` | 7-67x |
| `linum_create_mosaic_grid_3d_gpu.py` | `linum_create_mosaic_grid_3d.py` | 5-12x |
| `linum_resample_mosaic_grid_gpu.py` | `linum_resample_mosaic_grid.py` | 5-12x |
| `linum_normalize_intensities_per_slice_gpu.py` | `linum_normalize_intensities_per_slice.py` | 4-10x |
| `linum_fix_illumination_3d_gpu.py` | `linum_fix_illumination_3d.py` | 2-5x |
| `linum_assess_slice_quality_gpu.py` | `linum_assess_slice_quality.py` | 3-8x |
| `linum_aip_gpu.py` | `linum_aip_png.py` | ≤1x (mean projection; transfer overhead dominates for typical sizes) |

### Usage

```bash
# Use GPU (default)
linum_create_masks_gpu.py input.ome.zarr output.ome.zarr

# Disable GPU (force CPU)
linum_create_masks_gpu.py input.ome.zarr output.ome.zarr --no-use_gpu
```

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

### JAX CUDA Issues

```bash
# Run the fix script
source shell_scripts/fix_jax_cuda_plugin.sh

# Or check diagnostics
linum_diagnose_pipeline.py --debug-cuda
```

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
```

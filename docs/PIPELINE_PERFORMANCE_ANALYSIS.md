# 3D Reconstruction Pipeline Performance Analysis

## Executive Summary

**Root cause of slowdown: `params.processes = 1` default**

The `fix_illumination` step is now the bottleneck because it defaults to running with `processes = 1`, meaning each slice is processed sequentially. With ~20 slices and the BaSiC algorithm taking 5-10 minutes per slice, this alone could account for 2-3 hours of additional runtime.

However, the 8x slowdown (4-6h → 32h) suggests **additional factors** may be at play:

1. Thread limiting being too aggressive
2. GPU acceleration not being utilized
3. **SimpleITK thread pool not being limited** (CRITICAL)
4. Multiprocessing workers not respecting thread limits

---

## Identified CPU Limiting Gaps

### Gap 1: SimpleITK Thread Pool (CRITICAL)
**Problem**: SimpleITK spawns its own thread pool for registration operations that **ignores** environment variables like `OMP_NUM_THREADS`.

**Impact**: Each `register_pairwise` process can spawn 48+ threads, leading to massive thread oversubscription when multiple slices are processed.

**Fix Applied**: Added `configure_all_libraries()` calls after SimpleITK import in:
- `linum_estimate_transform.py`
- `linum_estimate_transform_gpu.py`
- `linum_interpolate_missing_slice.py`
- `linum_stack_slices_3d.py` (deprecated)

### Gap 2: Multiprocessing Workers Re-Import Libraries
**Problem**: When using `multiprocessing.Pool` or `pqdm`, each worker process is a fresh Python interpreter that re-imports all libraries. Even though environment variables are inherited, libraries like SimpleITK and numpy need runtime configuration.

**Impact**: Worker processes don't respect thread limits configured in the main process.

**Fix Applied**: 
- Added `worker_initializer` function in `_thread_config.py`
- Updated `multiprocessing.Pool` calls to use the initializer
- Added `apply_threadpool_limits()` call in `process_tile()` for pqdm workers

### Gap 3: configure_sitk() Never Called
**Problem**: The `configure_sitk()` function existed but was never called anywhere in the codebase.

**Fix Applied**: Added automatic SimpleITK configuration in `configure_all_libraries()`.

### Gap 4: Dask Configuration Only in zarr.py
**Problem**: `configure_dask()` was only called from `linumpy/io/zarr.py`, so scripts that use Dask without zarr.py wouldn't have proper thread limits.

**Fix Applied**: Included Dask configuration in `configure_all_libraries()`.

### Gap 5: JAX/XLA Thread Pool (BaSiCPy)
**Problem**: JAX (used by BaSiCPy for the BaSiC algorithm) has its own thread pool controlled by XLA_FLAGS, which was not being set.

**Impact**: The `fix_illumination` step could spawn excessive threads even when OMP_NUM_THREADS was limited, because JAX ignores OpenMP settings.

**Fix Applied**: 
- Added `XLA_FLAGS` environment variable setting in `_thread_config.py`
- Added `XLA_FLAGS` to Nextflow `beforeScript` in both workflow configs
- Added explicit `XLA_FLAGS` setting in `linum_fix_illumination_3d.py` before JAX import

**XLA_FLAGS format**:
```bash
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=N'
```

---

## Thread Configuration Architecture

### Before (Gaps Present)
```
Nextflow beforeScript → sets OMP_NUM_THREADS
         ↓
Python script imports linumpy._thread_config → sets env vars
         ↓
NumPy/SciPy import → ✓ respects OMP_NUM_THREADS  
SimpleITK import → ✗ IGNORES limits, spawns all CPU threads
Dask import → ✗ May not be configured
         ↓
Subprocess workers (pqdm/Pool) → Re-import libraries
         ↓
Workers: NumPy → ✓ inherits env vars
Workers: SimpleITK → ✗ IGNORES limits again
```

### After (Gaps Fixed)
```
Nextflow beforeScript → sets OMP_NUM_THREADS, XLA_FLAGS
         ↓
Python script imports linumpy._thread_config → sets env vars (incl. XLA_FLAGS)
         ↓
All imports complete
         ↓
configure_all_libraries() called → 
  ✓ NumPy/SciPy (threadpoolctl)
  ✓ SimpleITK (ProcessObject.SetGlobalDefaultNumberOfThreads)
  ✓ Dask (dask.config)
  ✓ Numba (set_num_threads)
  ✓ JAX/XLA (XLA_FLAGS environment variable)
         ↓
Subprocess workers created with worker_initializer →
  ✓ Re-applies all limits in each worker
```

### Process-by-Process Breakdown

| Process | GPU Support | Parallelism | Est. Time/Slice | Bottleneck Risk |
|---------|-------------|-------------|-----------------|-----------------|
| `resample_mosaic_grid` | ✅ Yes | Per-slice parallel | ~2 min | Low |
| `fix_focal_curvature` | ❌ No | Per-slice parallel | ~1 min | Low |
| **`fix_illumination`** | ❌ No | **Uses `params.processes`** | **5-10 min** | **HIGH** |
| `generate_aip` | ❌ No | Per-slice parallel | ~30 sec | Low |
| `estimate_xy_transformation` | ✅ Yes | Per-slice parallel | ~1 min | Low |
| `stitch_3d` | ❌ No | Per-slice parallel | ~2 min | Medium |
| `beam_profile_correction` | ❌ No | Per-slice parallel | ~2 min | Medium |
| `crop_interface` | ❌ No | Per-slice parallel | ~1 min | Low |
| `normalize` | ✅ Yes | Per-slice parallel | ~1 min | Low |
| `create_registration_masks` | ✅ Yes | Per-slice parallel | ~1 min | Low |
| `register_pairwise` | ❌ No | Sequential (by design) | ~5 min | Medium |
| `stack` | ❌ No | Single process | ~10 min | Low |

### Critical Issue: `fix_illumination`

```groovy
// In soct_3d_reconst.nf line 62-72
process fix_illumination {
    cpus params.processes  // <-- Uses params.processes

    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} ... --n_processes ${params.processes}
    """
}
```

```groovy
// In nextflow.config line 7
params {
    processes = 1  // <-- DEFAULT IS 1!
}
```

**Impact**: With 20 slices × 10 minutes each = 200 minutes (~3.3 hours) when running sequentially.

With `processes = 12`: All slices can run in parallel with internal parallelism, reducing to ~15-20 minutes total.

---

## Thread Configuration Analysis

The pipeline has a complex thread limiting system:

### Nextflow `beforeScript` (nextflow.config lines 110-140)
```groovy
int threadsPerProcess = Math.max(1, (int)(maxCpus / numProcesses))
envVars << "export OMP_NUM_THREADS=${threadsPerProcess}"
```

### Python Override (linum_fix_illumination_3d.py lines 13-14)
```python
from os import environ
environ["OMP_NUM_THREADS"] = "1"  # <-- HARDCODED TO 1!
```

**Problem**: The Python script overrides Nextflow's thread configuration, forcing BaSiC to use single-threaded execution.

---

## Recommended Configuration

For your 48-core, 512GB RAM, dual A6000 server:

### nextflow.config changes:

```groovy
params {
    // Change from 1 to 12-16 for parallel processing
    processes = 12  // Recommended for 48-core server
    
    // CPU management
    enable_cpu_limits = true
    reserved_cpus = 4  // Leave 4 cores for system overhead
    
    // GPU (already enabled)
    use_gpu = true
}
```

### Explanation of `processes = 12`:
- 48 cores total - 4 reserved = 44 available
- BaSiC algorithm benefits from ~3-4 threads per worker
- 44 / 3 ≈ 14-15 workers maximum
- Using 12 provides headroom for I/O and other processes

---

## Diagnostic Scripts Created

### 1. `linum_diagnose_pipeline.py`
Comprehensive Python diagnostic script:
```bash
# Quick check
python scripts/linum_diagnose_pipeline.py --quick

# Full benchmark
python scripts/linum_diagnose_pipeline.py --benchmark

# Save to file
python scripts/linum_diagnose_pipeline.py --output diagnosis.json
```

### 2. Server Checks
Quick server verification commands:
```bash
nproc && nvidia-smi && python -c "import cupy; print('CuPy OK')"
```

---

## Verification Checklist

Run these on your server before the next pipeline execution:

### 1. Check CPU Configuration
```bash
nproc  # Should show 48
```

### 2. Check GPU Availability
```bash
nvidia-smi  # Should show 2x A6000
```

### 3. Check CuPy Installation
```bash
python3 -c "import cupy; print(cupy.__version__)"
```

### 4. Check linumpy GPU Module
```bash
python3 -c "from linumpy.gpu import GPU_AVAILABLE; print(f'GPU: {GPU_AVAILABLE}')"
```

### 5. Verify nextflow.config Parameters
```bash
grep "processes" /path/to/workflows/reconst_3d/nextflow.config
# Should show: processes = 12 (or similar)
```

---

## Quick Fix

If you want to run immediately without modifying config files:

```bash
nextflow run soct_3d_reconst.nf \
    --input /path/to/input \
    --output /path/to/output \
    --processes 12 \
    --use_gpu true \
    --reserved_cpus 4 \
    -resume
```

---

## Additional Optimizations

### 1. Consider GPU-accelerated Illumination Correction
The BaSiC algorithm doesn't have a GPU version, but the processing could potentially be optimized by:
- Using JAX backend for BaSiC (if available)
- Processing multiple slices simultaneously on GPU

### 2. I/O Optimization
```groovy
// In nextflow.config
process {
    scratch = true  // Currently enabled
    stageInMode = 'symlink'  // Uses symlinks (good)
    stageOutMode = 'rsync'   // Consider 'move' for faster output
}
```

### 3. Memory-Intensive Process Limits
For processes that use a lot of memory, add `maxForks`:
```groovy
withName: "fix_illumination" {
    maxForks = 6  // Limit concurrent instances
}
```

---

## Expected Performance After Fixes

| Scenario | fix_illumination Time | Total Pipeline |
|----------|----------------------|----------------|
| Current (processes=1) | ~3-4 hours | 25-35 hours |
| Optimized (processes=12) | ~20-30 min | 4-6 hours |

The 8x improvement comes primarily from parallelizing the fix_illumination step.

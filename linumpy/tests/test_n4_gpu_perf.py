"""Performance benchmark: CPU SimpleITK N4 vs GPU CuPy N4 port.

These tests are skipped when CUDA is unavailable.

The synthetic volume is sized so both backends complete in tens of
seconds, not minutes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.intensity.bias_field import n4_correct, n4_correct_per_section


def _make_perf_volume(shape=(64, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zg, yg, xg = np.mgrid[0:z, 0:y, 0:x].astype(np.float32)
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zg - cz) / (z / 3)) ** 2 + ((yg - cy) / (y / 3)) ** 2 + ((xg - cx) / (x / 3)) ** 2)
    truth = np.where(r < 1.0, 1.0, 0.3).astype(np.float32) + rng.normal(0, 0.02, shape).astype(np.float32)
    bias = (1.0 + 0.5 * (zg / z + 0.5 * yg / y - 0.5 * xg / x)).astype(np.float32)
    mask = r < 1.2
    return (truth * bias).astype(np.float32), mask


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_n4_gpu_faster_than_cpu_synthetic():
    """On a 128x512x512 synthetic volume (realistic OCT slab), GPU N4 should
    be at least 2x faster than the SimpleITK CPU implementation.  Measured
    speedup at this size is ~3.3x; we assert 2x to allow run-to-run variance.
    Tiny volumes (e.g. 64x128x128) are dominated by CUDA launch overhead and
    do NOT exercise the perf benefit of the GPU implementation."""
    vol, mask = _make_perf_volume(shape=(128, 512, 512))
    n_iters = [25, 25, 25]
    spline_dist = 20.0

    # Warm-up (CUDA / cuFFT plan caches)
    n4_correct(vol[:8], mask[:8], shrink_factor=2, n_iterations=[5], backend="gpu", spline_distance_mm=spline_dist)

    t0 = time.perf_counter()
    cpu_corr, _ = n4_correct(vol, mask, shrink_factor=2, n_iterations=n_iters, backend="cpu", spline_distance_mm=spline_dist)
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_corr, _ = n4_correct(vol, mask, shrink_factor=2, n_iterations=n_iters, backend="gpu", spline_distance_mm=spline_dist)
    gpu_time = time.perf_counter() - t0

    speedup = cpu_time / max(gpu_time, 1e-6)
    print(f"\nN4 perf: cpu={cpu_time:.2f}s gpu={gpu_time:.2f}s speedup={speedup:.2f}x")
    assert np.isfinite(cpu_corr).all()
    assert np.isfinite(gpu_corr).all()
    assert speedup >= 2.0, f"Expected >=2x speedup, got {speedup:.2f}x (cpu={cpu_time:.2f}s, gpu={gpu_time:.2f}s)"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_n4_gpu_per_section_speedup():
    """Per-section GPU should beat per-section single-process CPU by >=1.5x.
    (Multiprocessing CPU may approach GPU throughput; we compare against
    single-process to isolate per-section overhead.)"""
    vol, mask = _make_perf_volume(shape=(32, 512, 512))

    # Warm-up
    n4_correct_per_section(
        vol[:8], n_serial_slices=1, mask=mask[:8], n_processes=1, shrink_factor=2, n_iterations=[3], backend="gpu"
    )

    t0 = time.perf_counter()
    cpu_corr, _ = n4_correct_per_section(
        vol,
        n_serial_slices=4,
        mask=mask,
        n_processes=1,
        shrink_factor=2,
        n_iterations=[10],
        spline_distance_mm=15.0,
        backend="cpu",
    )
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_corr, _ = n4_correct_per_section(
        vol,
        n_serial_slices=4,
        mask=mask,
        n_processes=1,  # forced internally
        shrink_factor=2,
        n_iterations=[10],
        spline_distance_mm=15.0,
        backend="gpu",
    )
    gpu_time = time.perf_counter() - t0

    speedup = cpu_time / max(gpu_time, 1e-6)
    print(f"\nN4 per-section perf: cpu={cpu_time:.2f}s gpu={gpu_time:.2f}s speedup={speedup:.2f}x")
    assert np.isfinite(cpu_corr).all()
    assert np.isfinite(gpu_corr).all()
    assert speedup >= 1.5, f"Per-section: expected >=1.5x speedup, got {speedup:.2f}x"

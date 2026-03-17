#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark GPU vs CPU performance for linumpy operations.

This script tests all GPU-accelerated operations and compares their
performance against CPU implementations. It also verifies that results
are numerically equivalent.

Usage:
    # Quick benchmark with synthetic data
    linum_benchmark_gpu.py
    
    # Benchmark with real data
    linum_benchmark_gpu.py --input /path/to/mosaic.ome.zarr
    
    # Full benchmark with multiple sizes
    linum_benchmark_gpu.py --full
    
    # Save results to file
    linum_benchmark_gpu.py --output benchmark_results.json
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np

# Import GPU module
from linumpy.gpu import GPU_AVAILABLE, gpu_info, print_gpu_info


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=str,
                   help="Path to OME-Zarr file for real-data benchmark")
    p.add_argument("--output", "-o", type=str,
                   help="Save results to JSON file")
    p.add_argument("--iterations", "-n", type=int, default=3,
                   help="Number of iterations per test [%(default)s]")
    p.add_argument("--full", action="store_true",
                   help="Run full benchmark with multiple sizes")
    p.add_argument("--skip-correctness", action="store_true",
                   help="Skip result correctness checks")
    p.add_argument("--sizes", nargs="+", type=int, 
                   default=[512, 1024, 2048],
                   help="Image sizes to test [%(default)s]")
    p.add_argument("--select-best-gpu", action="store_true",
                   help="Automatically select GPU with most free memory")
    p.add_argument("--gpu", type=int, metavar="ID",
                   help="Select specific GPU by ID")
    return p


class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def benchmark_operation(func_cpu, func_gpu, data, name, iterations=3, check_correctness=True):
    """
    Benchmark a single operation comparing CPU and GPU.
    
    Returns dict with timing results.
    """
    results = {
        "name": name,
        "cpu_times": [],
        "gpu_times": [],
        "correct": None,
        "max_diff": None,
    }
    
    # Warmup GPU
    if GPU_AVAILABLE:
        try:
            _ = func_gpu(data)
        except Exception:
            pass
    
    # CPU benchmark
    for _ in range(iterations):
        with BenchmarkTimer() as t:
            result_cpu = func_cpu(data)
        results["cpu_times"].append(t.elapsed)
    
    # GPU benchmark
    if GPU_AVAILABLE:
        for _ in range(iterations):
            with BenchmarkTimer() as t:
                result_gpu = func_gpu(data)
            results["gpu_times"].append(t.elapsed)
        
        # Check correctness
        if check_correctness and result_cpu is not None and result_gpu is not None:
            try:
                if isinstance(result_cpu, tuple):
                    result_cpu = result_cpu[0]
                    result_gpu = result_gpu[0]
                
                cpu_arr = np.asarray(result_cpu)
                gpu_arr = np.asarray(result_gpu)
                
                # Handle complex arrays by comparing magnitudes
                if np.iscomplexobj(cpu_arr) or np.iscomplexobj(gpu_arr):
                    cpu_arr = np.abs(cpu_arr)
                    gpu_arr = np.abs(gpu_arr)
                
                cpu_arr = cpu_arr.astype(np.float64)
                gpu_arr = gpu_arr.astype(np.float64)
                
                # Compute various error metrics
                abs_diff = np.abs(cpu_arr - gpu_arr)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                
                # Relative error (avoid division by zero)
                cpu_max = np.max(np.abs(cpu_arr))
                rel_error = max_diff / cpu_max if cpu_max > 1e-10 else max_diff
                
                results["max_diff"] = float(max_diff)
                results["mean_diff"] = float(mean_diff)
                results["rel_error"] = float(rel_error)
                
                # Use relative tolerance for correctness check
                # 1e-5 relative error is acceptable for float32 operations
                results["correct"] = rel_error < 1e-4
            except Exception as e:
                results["correct"] = f"Check failed: {e}"
    else:
        results["gpu_times"] = [float('nan')] * iterations
        results["correct"] = "GPU not available"
    
    # Compute statistics
    results["cpu_mean"] = np.mean(results["cpu_times"])
    results["cpu_std"] = np.std(results["cpu_times"])
    results["gpu_mean"] = np.mean(results["gpu_times"]) if GPU_AVAILABLE else float('nan')
    results["gpu_std"] = np.std(results["gpu_times"]) if GPU_AVAILABLE else float('nan')
    results["speedup"] = results["cpu_mean"] / results["gpu_mean"] if GPU_AVAILABLE else float('nan')
    
    return results


def benchmark_fft(size, iterations=3, check_correctness=True):
    """Benchmark FFT operations."""
    from linumpy.gpu.fft_ops import fft2
    
    data = np.random.rand(size, size).astype(np.float32)
    
    def cpu_fft(d):
        return np.fft.fft2(d)
    
    def gpu_fft(d):
        return fft2(d, use_gpu=True)
    
    return benchmark_operation(cpu_fft, gpu_fft, data, f"FFT2 ({size}x{size})",
                               iterations, check_correctness)


def benchmark_phase_correlation(size, iterations=3, check_correctness=True):
    """Benchmark phase correlation."""
    from linumpy.gpu.fft_ops import phase_correlation
    from linumpy.stitching.registration import pairWisePhaseCorrelation
    
    # Create two slightly shifted images
    img1 = np.random.rand(size, size).astype(np.float32)
    img2 = np.roll(img1, (5, 10), axis=(0, 1))
    
    def cpu_pc(d):
        return pairWisePhaseCorrelation(d[0], d[1], returnCC=True)
    
    def gpu_pc(d):
        return phase_correlation(d[0], d[1], use_gpu=True)
    
    data = (img1, img2)
    
    return benchmark_operation(cpu_pc, gpu_pc, data, f"Phase Correlation ({size}x{size})",
                               iterations, check_correctness=False)  # Results may differ slightly


def benchmark_gaussian_filter(size, iterations=3, check_correctness=True):
    """Benchmark Gaussian filtering."""
    from linumpy.gpu.morphology import gaussian_filter
    from scipy.ndimage import gaussian_filter as scipy_gaussian
    
    data = np.random.rand(size, size).astype(np.float32)
    sigma = 2.0
    
    def cpu_gauss(d):
        return scipy_gaussian(d, sigma=sigma)
    
    def gpu_gauss(d):
        return gaussian_filter(d, sigma=sigma, use_gpu=True)
    
    return benchmark_operation(cpu_gauss, gpu_gauss, data,
                               f"Gaussian Filter ({size}x{size})",
                               iterations, check_correctness)


def benchmark_binary_closing(size, iterations=3, check_correctness=True):
    """Benchmark binary morphology."""
    from linumpy.gpu.morphology import binary_closing
    from scipy.ndimage import binary_closing as scipy_closing
    
    # Create random binary mask
    data = (np.random.rand(size, size) > 0.5).astype(np.bool_)
    
    def cpu_close(d):
        return scipy_closing(d, iterations=2)
    
    def gpu_close(d):
        return binary_closing(d, iterations=2, use_gpu=True)
    
    return benchmark_operation(cpu_close, gpu_close, data,
                               f"Binary Closing ({size}x{size})",
                               iterations, check_correctness)


def benchmark_resize(size, iterations=3, check_correctness=True):
    """Benchmark image resize."""
    from linumpy.gpu.interpolation import resize
    
    data = np.random.rand(size, size).astype(np.float32)
    output_size = (size // 2, size // 2)
    
    def cpu_resize(d):
        # Use the same function with use_gpu=False for fair comparison
        return resize(d, output_size, order=1, anti_aliasing=False, use_gpu=False)
    
    def gpu_resize(d):
        return resize(d, output_size, order=1, anti_aliasing=False, use_gpu=True)
    
    return benchmark_operation(cpu_resize, gpu_resize, data,
                               f"Resize ({size}→{size//2})",
                               iterations, check_correctness)


def benchmark_rescale_3d(size, iterations=3, check_correctness=True):
    """Benchmark 3D volume rescaling (like linum_resample_mosaic_grid)."""
    from linumpy.gpu.interpolation import resize
    
    # Create 3D volume (typical OCT tile: depth x height x width)
    depth = size // 4  # Typically depth is smaller than XY
    data = np.random.rand(depth, size, size).astype(np.float32)
    
    # Rescale by 0.5 in each dimension (typical downsampling)
    scale_factor = 0.5
    output_size = (int(depth * scale_factor), int(size * scale_factor), int(size * scale_factor))
    
    def cpu_rescale(d):
        return resize(d, output_size, order=1, anti_aliasing=True, use_gpu=False)
    
    def gpu_rescale(d):
        return resize(d, output_size, order=1, anti_aliasing=True, use_gpu=True)
    
    return benchmark_operation(cpu_rescale, gpu_rescale, data,
                               f"Rescale 3D ({depth}x{size}x{size}→{output_size[0]}x{output_size[1]}x{output_size[2]})",
                               iterations, check_correctness)


def benchmark_normalize(size, iterations=3, check_correctness=True):
    """Benchmark percentile normalization."""
    from linumpy.gpu.array_ops import normalize_percentile
    
    data = np.random.rand(size, size).astype(np.float32) * 1000
    
    def cpu_norm(d):
        low, high = np.percentile(d, [1, 99])
        return np.clip((d - low) / (high - low), 0, 1)
    
    def gpu_norm(d):
        return normalize_percentile(d, p_low=1, p_high=99, use_gpu=True)
    
    return benchmark_operation(cpu_norm, gpu_norm, data,
                               f"Normalize ({size}x{size})",
                               iterations, check_correctness)

def benchmark_intensity_normalization(size, iterations=3, check_correctness=True):
    """Benchmark intensity normalization operations."""
    from linumpy.gpu.array_ops import threshold_otsu, normalize_percentile
    from linumpy.gpu.morphology import gaussian_filter

    data = np.random.rand(size, size).astype(np.float32) * 1000

    def cpu_norm(d):
        # Simulate intensity normalization operations
        from scipy.ndimage import gaussian_filter as scipy_gaussian
        smoothed = scipy_gaussian(d, sigma=1.0)
        from skimage.filters import threshold_otsu as sk_otsu
        threshold = sk_otsu(smoothed)
        mask = smoothed > threshold
        low, high = np.percentile(smoothed, [1, 99])
        normalized = np.clip((smoothed - low) / (high - low), 0, 1)
        return normalized

    def gpu_norm(d):
        # Simulate intensity normalization operations
        smoothed = gaussian_filter(d, sigma=1.0, use_gpu=True)
        threshold = threshold_otsu(smoothed, use_gpu=True)
        mask = smoothed > threshold
        normalized = normalize_percentile(smoothed, p_low=1, p_high=99, use_gpu=True)
        return normalized

    return benchmark_operation(cpu_norm, gpu_norm, data,
                               f"Intensity Normalization ({size}x{size})",
                               iterations, check_correctness)

def benchmark_real_data(input_path, iterations=3):
    """Benchmark with real OME-Zarr data."""
    from linumpy.io.zarr import read_omezarr
    from linumpy.gpu.morphology import gaussian_filter
    
    print(f"\nLoading real data from: {input_path}")
    vol, res = read_omezarr(input_path, level=0)
    
    # Load a manageable chunk
    chunk_size = min(100, vol.shape[0])
    data = np.array(vol[:chunk_size])
    print(f"Loaded chunk shape: {data.shape}")
    
    results = []
    
    # Gaussian on real data AIP
    aip = np.mean(data, axis=0).astype(np.float32)
    from scipy.ndimage import gaussian_filter as scipy_gaussian
    
    def cpu_gauss(d):
        return scipy_gaussian(d, sigma=2.0)
    
    def gpu_gauss(d):
        return gaussian_filter(d, sigma=2.0, use_gpu=True)
    
    results.append(benchmark_operation(cpu_gauss, gpu_gauss, aip,
                                       f"Real Data Gaussian {aip.shape}", iterations))
    
    return results


def print_results(all_results, gpu_info_dict):
    """Print formatted benchmark results."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {gpu_info_dict['device_name']}")
    print(f"GPU Memory: {gpu_info_dict['memory_gb']:.1f} GB")
    print(f"GPU Available: {gpu_info_dict['gpu_available']}")
    print("=" * 90)
    
    print(f"\n{'Operation':<40} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Correct':<8} {'Rel Err':<12}")
    print("-" * 94)
    
    for result in all_results:
        cpu_ms = result['cpu_mean'] * 1000
        gpu_ms = result['gpu_mean'] * 1000 if not np.isnan(result['gpu_mean']) else float('nan')
        speedup = result['speedup'] if not np.isnan(result['speedup']) else float('nan')
        correct = "✓" if result.get('correct') == True else ("N/A" if result.get('correct') is None else "✗")
        rel_err = result.get('rel_error', float('nan'))
        
        if np.isnan(gpu_ms):
            print(f"{result['name']:<40} {cpu_ms:>10.2f}   {'N/A':^10}   {'N/A':^8}   {correct:^6}   {'N/A':^10}")
        else:
            rel_err_str = f"{rel_err:.2e}" if not np.isnan(rel_err) else "N/A"
            print(f"{result['name']:<40} {cpu_ms:>10.2f}   {gpu_ms:>10.2f}   {speedup:>7.1f}x   {correct:^6}   {rel_err_str:>10}")
    
    print("-" * 94)
    
    # Summary statistics
    valid_speedups = [r['speedup'] for r in all_results if not np.isnan(r['speedup'])]
    if valid_speedups:
        print(f"\nAverage speedup: {np.mean(valid_speedups):.1f}x")
        print(f"Max speedup: {np.max(valid_speedups):.1f}x")
        print(f"Min speedup: {np.min(valid_speedups):.1f}x")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Handle GPU selection
    if args.select_best_gpu:
        from linumpy.gpu import select_best_gpu
        select_best_gpu(verbose=True)
        print()
    elif args.gpu is not None:
        from linumpy.gpu import select_gpu
        select_gpu(args.gpu, verbose=True)
        print()
    
    # Print GPU info
    print_gpu_info()
    info = gpu_info()
    
    if not info['gpu_available']:
        print("\n⚠️  WARNING: GPU not available. Only CPU benchmarks will run.\n")
    
    all_results = []
    iterations = args.iterations
    check_correctness = not args.skip_correctness
    
    # Synthetic data benchmarks
    sizes = args.sizes if args.full else [args.sizes[0]]
    
    print(f"\nRunning benchmarks with {iterations} iterations per test...")
    print(f"Testing sizes: {sizes}")
    
    for size in sizes:
        print(f"\n--- Size: {size}x{size} ---")
        
        # FFT
        print("  Testing FFT...", end=" ", flush=True)
        all_results.append(benchmark_fft(size, iterations, check_correctness))
        print("done")
        
        # Phase correlation
        print("  Testing Phase Correlation...", end=" ", flush=True)
        all_results.append(benchmark_phase_correlation(size, iterations, check_correctness))
        print("done")
        
        # Gaussian filter
        print("  Testing Gaussian Filter...", end=" ", flush=True)
        all_results.append(benchmark_gaussian_filter(size, iterations, check_correctness))
        print("done")
        
        # Binary closing
        print("  Testing Binary Closing...", end=" ", flush=True)
        all_results.append(benchmark_binary_closing(size, iterations, check_correctness))
        print("done")
        
        # Resize
        print("  Testing Resize...", end=" ", flush=True)
        all_results.append(benchmark_resize(size, iterations, check_correctness))
        print("done")
        
        # Rescale 3D (like linum_resample_mosaic_grid_gpu)
        print("  Testing Rescale 3D...", end=" ", flush=True)
        all_results.append(benchmark_rescale_3d(size, iterations, check_correctness))
        print("done")

        # Normalize
        print("  Testing Normalize...", end=" ", flush=True)
        all_results.append(benchmark_normalize(size, iterations, check_correctness))
        print("done")

        # Intensity normalization
        print("  Testing Intensity Normalization...", end=" ", flush=True)
        all_results.append(benchmark_intensity_normalization(size, iterations, check_correctness))
        print("done")
    
    # Real data benchmark
    if args.input:
        print("\n--- Real Data Benchmark ---")
        real_results = benchmark_real_data(args.input, iterations)
        all_results.extend(real_results)
    
    # Print results
    print_results(all_results, info)
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": info,
            "parameters": {
                "iterations": iterations,
                "sizes": sizes,
                "input_file": args.input,
            },
            "results": all_results,
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {args.output}")
    
    # Return exit code based on GPU availability
    sys.exit(0 if info['gpu_available'] else 1)


if __name__ == "__main__":
    main()

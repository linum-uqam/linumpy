#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print GPU availability and configuration information for linumpy.

This script checks if GPU acceleration is available and prints
diagnostic information useful for troubleshooting.

Examples:
    # Show basic GPU info
    linum_gpu_info.py
    
    # Show detailed status of all GPUs with memory usage
    linum_gpu_info.py --status
    
    # List all available GPUs
    linum_gpu_info.py --list
    
    # Select GPU with most free memory (for multi-GPU systems)
    linum_gpu_info.py --select-best
    
    # Select specific GPU by ID
    linum_gpu_info.py --select 1
    
    # Run quick performance test
    linum_gpu_info.py --test
    
    # Output as JSON (useful for scripting)
    linum_gpu_info.py --json
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import sys


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--json", action="store_true",
                   help="Output as JSON")
    p.add_argument("--test", action="store_true",
                   help="Run a quick GPU test")
    p.add_argument("--status", action="store_true",
                   help="Show detailed status of all GPUs")
    p.add_argument("--list", action="store_true",
                   help="List all available GPUs")
    p.add_argument("--select-best", action="store_true",
                   help="Select GPU with most free memory")
    p.add_argument("--select", type=int, metavar="ID",
                   help="Select specific GPU by ID")
    return p


def run_gpu_test():
    """Run a quick GPU performance test."""
    import time
    import numpy as np
    
    print("\n" + "=" * 50)
    print("GPU Performance Test")
    print("=" * 50)
    
    # Test data
    size = 2048
    data = np.random.rand(size, size).astype(np.float32)
    
    # CPU FFT
    start = time.time()
    for _ in range(10):
        _ = np.fft.fft2(data)
    cpu_time = (time.time() - start) / 10
    print(f"CPU FFT ({size}x{size}): {cpu_time*1000:.2f} ms")
    
    # GPU FFT
    try:
        import cupy as cp
        
        data_gpu = cp.asarray(data)
        
        # Warmup
        _ = cp.fft.fft2(data_gpu)
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        for _ in range(10):
            _ = cp.fft.fft2(data_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) / 10
        
        print(f"GPU FFT ({size}x{size}): {gpu_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
    except Exception as e:
        print(f"GPU test failed: {e}")
    
    print("=" * 50)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    from linumpy.gpu import (
        gpu_info, print_gpu_info, print_gpu_status,
        list_gpus, select_best_gpu, select_gpu
    )
    
    # Handle GPU selection first
    if args.select_best:
        select_best_gpu(verbose=True)
        print()
    elif args.select is not None:
        select_gpu(args.select, verbose=True)
        print()
    
    # Handle output modes
    if args.json:
        import json
        info = gpu_info()
        info['all_gpus'] = list_gpus()
        print(json.dumps(info, indent=2))
    elif args.status:
        print_gpu_status()
    elif args.list:
        gpus = list_gpus()
        if gpus:
            print(f"Found {len(gpus)} GPU(s):\n")
            for gpu in gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']}")
                print(f"         {gpu['free_gb']:.1f} GB free / {gpu['total_gb']:.1f} GB total")
        else:
            print("No GPUs found")
    else:
        print_gpu_info()
    
    if args.test:
        run_gpu_test()
    
    # Return exit code based on GPU availability
    info = gpu_info()
    sys.exit(0 if info['gpu_available'] else 1)


if __name__ == "__main__":
    main()

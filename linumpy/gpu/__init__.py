"""
GPU acceleration module for linumpy.

This module provides GPU-accelerated versions of compute-intensive operations
using CuPy. All functions have automatic fallback to CPU (NumPy) if:
- CuPy is not installed
- No CUDA-capable GPU is available
- GPU memory is insufficient

Usage:
    from linumpy.gpu import GPU_AVAILABLE, get_array_module
    
    # Check if GPU is available
    if GPU_AVAILABLE:
        print("GPU acceleration enabled")
    
    # Get appropriate array module (cupy or numpy)
    xp = get_array_module(use_gpu=True)
    
    # Use GPU-accelerated functions
    from linumpy.gpu.fft_ops import gpu_phase_correlation
    from linumpy.gpu.interpolation import gpu_affine_transform
    from linumpy.gpu.registration import GPUAcceleratedRegistration

Configuration:
    Set USE_GPU=false environment variable to disable GPU globally.
"""

import os
import warnings

# Check for GPU availability
GPU_AVAILABLE = False
CUPY_AVAILABLE = False
GPU_DEVICE_NAME = "N/A"
GPU_MEMORY_GB = 0

# Allow disabling GPU via environment variable
_USE_GPU_ENV = os.environ.get("LINUMPY_USE_GPU", "true").lower()
_GPU_DISABLED_BY_ENV = _USE_GPU_ENV in ("false", "0", "no")

if not _GPU_DISABLED_BY_ENV:
    try:
        import cupy as cp

        # Test if CUDA is actually available
        try:
            # First, find the GPU with most free memory
            n_devices = cp.cuda.runtime.getDeviceCount()

            if n_devices > 0:
                best_gpu_id = 0
                best_free_memory = 0

                for i in range(n_devices):
                    with cp.cuda.Device(i):
                        free, total = cp.cuda.runtime.memGetInfo()
                        if free > best_free_memory:
                            best_free_memory = free
                            best_gpu_id = i

                # Select the best GPU
                cp.cuda.Device(best_gpu_id).use()

                CUPY_AVAILABLE = True
                GPU_AVAILABLE = True

                # Get device info for selected GPU
                device = cp.cuda.Device(best_gpu_id)
                GPU_DEVICE_NAME = device.name if hasattr(device, 'name') else f"GPU {device.id}"
                mem_info = device.mem_info
                GPU_MEMORY_GB = mem_info[1] / (1024 ** 3)  # Total memory in GB

                if n_devices > 1:
                    # Only show message if there are multiple GPUs
                    import sys

                    print(f"Auto-selected GPU {best_gpu_id}: {GPU_DEVICE_NAME} "
                          f"({best_free_memory / (1024 ** 3):.1f} GB free)", file=sys.stderr)
            else:
                CUPY_AVAILABLE = True
                GPU_AVAILABLE = False

        except cp.cuda.runtime.CUDARuntimeError as e:
            warnings.warn(f"CuPy installed but CUDA not available: {e}")
            CUPY_AVAILABLE = True
            GPU_AVAILABLE = False

    except ImportError:
        pass
else:
    warnings.warn("GPU disabled via LINUMPY_USE_GPU environment variable")


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (cupy or numpy).
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU if available.
        
    Returns
    -------
    module
        cupy if GPU available and use_gpu=True, else numpy
    """
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        return cp
    else:
        import numpy as np
        return np


def to_gpu(array):
    """
    Transfer array to GPU if available.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
        
    Returns
    -------
    array
        CuPy array if GPU available, else original numpy array
    """
    if GPU_AVAILABLE:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return array
        return cp.asarray(array)
    return array


def to_cpu(array):
    """
    Transfer array to CPU (numpy).
    
    Parameters
    ----------
    array : array-like
        Input array (numpy or cupy)
        
    Returns
    -------
    np.ndarray
        NumPy array
    """
    if GPU_AVAILABLE:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    return array


def gpu_info():
    """
    Get information about GPU availability and configuration.
    
    Returns
    -------
    dict
        Dictionary with GPU information
    """
    return {
        "gpu_available": GPU_AVAILABLE,
        "cupy_installed": CUPY_AVAILABLE,
        "device_name": GPU_DEVICE_NAME,
        "memory_gb": GPU_MEMORY_GB,
        "disabled_by_env": _GPU_DISABLED_BY_ENV,
    }


def print_gpu_info():
    """Print GPU availability information."""
    info = gpu_info()
    print("=" * 50)
    print("linumpy GPU Configuration")
    print("=" * 50)
    print(f"  GPU Available:     {info['gpu_available']}")
    print(f"  CuPy Installed:    {info['cupy_installed']}")
    print(f"  Device:            {info['device_name']}")
    print(f"  Memory:            {info['memory_gb']:.1f} GB")
    if info['disabled_by_env']:
        print(f"  NOTE: GPU disabled via environment variable")
    print("=" * 50)


def list_gpus():
    """
    List all available GPUs with memory information.
    
    Returns
    -------
    list of dict
        List of GPU info dictionaries with keys:
        - id: Device ID
        - name: Device name
        - total_gb: Total memory in GB
        - free_gb: Free memory in GB
        - used_gb: Used memory in GB
        - utilization: Memory utilization (0-1)
    """
    if not CUPY_AVAILABLE:
        return []

    import cupy as cp

    gpus = []
    n_devices = cp.cuda.runtime.getDeviceCount()

    for i in range(n_devices):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            device = cp.cuda.Device(i)
            name = device.name if hasattr(device, 'name') else f"GPU {i}"

            gpus.append({
                'id': i,
                'name': name,
                'total_gb': total / (1024 ** 3),
                'free_gb': free / (1024 ** 3),
                'used_gb': (total - free) / (1024 ** 3),
                'utilization': (total - free) / total,
            })

    return gpus


def select_best_gpu(verbose: bool = True):
    """
    Select the GPU with the most free memory.
    
    This function queries all available GPUs and switches to the one
    with the most free memory. Useful when running on multi-GPU systems
    where one GPU may already be in use.
    
    Parameters
    ----------
    verbose : bool
        Print selection information
        
    Returns
    -------
    int or None
        Selected GPU ID, or None if no GPU available
        
    Examples
    --------
    >>> from linumpy.gpu import select_best_gpu
    >>> select_best_gpu()
    Selected GPU 1: NVIDIA RTX A6000 (45.2 GB free / 48.0 GB total)
    1
    """
    global GPU_AVAILABLE, GPU_DEVICE_NAME, GPU_MEMORY_GB

    if not CUPY_AVAILABLE:
        if verbose:
            print("No GPU available (CuPy not installed)")
        return None

    import cupy as cp

    gpus = list_gpus()

    if not gpus:
        if verbose:
            print("No GPUs found")
        return None

    # Find GPU with most free memory
    best_gpu = max(gpus, key=lambda g: g['free_gb'])
    best_id = best_gpu['id']

    # Switch to best GPU
    cp.cuda.Device(best_id).use()

    # Update module globals
    GPU_AVAILABLE = True
    GPU_DEVICE_NAME = best_gpu['name']
    GPU_MEMORY_GB = best_gpu['total_gb']

    if verbose:
        print(f"Selected GPU {best_id}: {best_gpu['name']} "
              f"({best_gpu['free_gb']:.1f} GB free / {best_gpu['total_gb']:.1f} GB total)")

        if len(gpus) > 1:
            print(f"  (Selected from {len(gpus)} available GPUs)")

    return best_id


def select_gpu(device_id: int, verbose: bool = True):
    """
    Select a specific GPU by device ID.
    
    Parameters
    ----------
    device_id : int
        GPU device ID (0, 1, 2, ...)
    verbose : bool
        Print selection information
        
    Returns
    -------
    int or None
        Selected GPU ID, or None if invalid
        
    Examples
    --------
    >>> from linumpy.gpu import select_gpu
    >>> select_gpu(1)
    Selected GPU 1: NVIDIA RTX A6000 (48.0 GB total)
    1
    """
    global GPU_AVAILABLE, GPU_DEVICE_NAME, GPU_MEMORY_GB

    if not CUPY_AVAILABLE:
        if verbose:
            print("No GPU available (CuPy not installed)")
        return None

    import cupy as cp

    n_devices = cp.cuda.runtime.getDeviceCount()

    if device_id < 0 or device_id >= n_devices:
        if verbose:
            print(f"Invalid GPU ID {device_id}. Available: 0-{n_devices - 1}")
        return None

    # Switch to specified GPU
    cp.cuda.Device(device_id).use()

    # Update module globals
    with cp.cuda.Device(device_id):
        free, total = cp.cuda.runtime.memGetInfo()
        device = cp.cuda.Device(device_id)
        name = device.name if hasattr(device, 'name') else f"GPU {device_id}"

        GPU_AVAILABLE = True
        GPU_DEVICE_NAME = name
        GPU_MEMORY_GB = total / (1024 ** 3)

    if verbose:
        print(f"Selected GPU {device_id}: {name} ({GPU_MEMORY_GB:.1f} GB total)")

    return device_id


def print_gpu_status():
    """
    Print detailed status of all available GPUs.
    
    Shows memory usage for each GPU, highlighting the currently selected one.
    """
    if not CUPY_AVAILABLE:
        print("No GPU available (CuPy not installed)")
        return

    import cupy as cp

    gpus = list_gpus()
    current_device = cp.cuda.Device().id

    print("=" * 60)
    print("GPU Status")
    print("=" * 60)

    for gpu in gpus:
        marker = " *" if gpu['id'] == current_device else "  "
        bar_width = 30
        used_bars = int(gpu['utilization'] * bar_width)
        bar = "█" * used_bars + "░" * (bar_width - used_bars)

        print(f"{marker}GPU {gpu['id']}: {gpu['name']}")
        print(f"    Memory: [{bar}] {gpu['utilization'] * 100:.1f}%")
        print(f"    {gpu['used_gb']:.1f} GB used / {gpu['total_gb']:.1f} GB total "
              f"({gpu['free_gb']:.1f} GB free)")

    print("=" * 60)
    print(f"  * = currently selected")


# Import CUDA environment setup functions
from linumpy.gpu.cuda_env import (
    setup_jax_cuda_env,
    get_cuda12_ld_path,
    check_patchelf_needed,
    apply_patchelf_fix,
    verify_jax_cuda,
)

# Expose key components
__all__ = [
    'GPU_AVAILABLE',
    'CUPY_AVAILABLE',
    'GPU_DEVICE_NAME',
    'GPU_MEMORY_GB',
    'get_array_module',
    'to_gpu',
    'to_cpu',
    'gpu_info',
    'print_gpu_info',
    'list_gpus',
    'select_best_gpu',
    'select_gpu',
    'print_gpu_status',
    # CUDA environment setup for JAX
    'setup_jax_cuda_env',
    'get_cuda12_ld_path',
    'check_patchelf_needed',
    'apply_patchelf_fix',
    'verify_jax_cuda',
]

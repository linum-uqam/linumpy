# -*- coding: utf-8 -*-
"""
Thread configuration module for linumpy.

This module MUST be imported before numpy, scipy, or other numerical libraries
to ensure thread limits are respected. It is automatically imported by the
linumpy package __init__.py and can also be imported directly by scripts.

The module configures thread limits for:
- OpenMP (OMP_NUM_THREADS)
- Intel MKL (MKL_NUM_THREADS)
- OpenBLAS (OPENBLAS_NUM_THREADS)
- macOS Accelerate (VECLIB_MAXIMUM_THREADS)
- NumExpr (NUMEXPR_NUM_THREADS)
- Numba (NUMBA_NUM_THREADS)
- Dask (via dask.config)
- SimpleITK (via ProcessObject.SetGlobalDefaultNumberOfThreads)

Environment variables (checked in order of precedence):
1. OMP_NUM_THREADS - if already set, respects it (user/Nextflow override)
2. LINUMPY_MAX_CPUS - explicit maximum CPUs
3. LINUMPY_RESERVED_CPUS - CPUs to reserve for overhead

Known gaps that can cause CPU usage spikes:
1. SimpleITK spawns its own thread pool - configure_sitk() must be called after import
2. Subprocess workers (pqdm, multiprocessing.Pool) re-import libraries
3. Some libraries ignore environment variables set after import
4. CuPy/GPU operations don't respect CPU thread limits

To ensure proper limiting, scripts should call configure_all_libraries() after imports.
"""

import multiprocessing
import os
import sys

# Track if we've already configured
_thread_config_applied = False


def get_max_threads():
    """
    Calculate the maximum number of threads to use based on environment variables.

    Returns:
        int: Maximum number of threads to use
    """
    total_cpus = multiprocessing.cpu_count()

    # Check for explicit max CPUs limit
    max_cpus = os.environ.get('LINUMPY_MAX_CPUS')
    if max_cpus is not None:
        try:
            return max(1, min(int(max_cpus), total_cpus))
        except ValueError:
            pass

    # Check for reserved CPUs
    reserved = os.environ.get('LINUMPY_RESERVED_CPUS')
    if reserved is not None:
        try:
            return max(1, total_cpus - int(reserved))
        except ValueError:
            pass

    # Default: use all CPUs
    return total_cpus


def configure_thread_limits():
    """
    Configure thread limits for numerical libraries.

    This function sets environment variables that control threading in
    numpy, scipy, and other numerical libraries. It must be called
    BEFORE these libraries are imported to be effective.

    If OMP_NUM_THREADS is already set (e.g., by Nextflow beforeScript),
    this function will still set the other threading variables to match.
    """
    # Calculate the thread limit
    max_threads = get_max_threads()

    # If OMP_NUM_THREADS is already set, use that value instead
    if 'OMP_NUM_THREADS' in os.environ:
        try:
            max_threads = int(os.environ['OMP_NUM_THREADS'])
        except ValueError:
            pass

    # Set environment variables for all common threading libraries
    # Set ALL of them unconditionally to ensure consistency
    thread_vars = [
        'OMP_NUM_THREADS',  # OpenMP (used by numpy, scipy, etc.)
        'MKL_NUM_THREADS',  # Intel MKL
        'OPENBLAS_NUM_THREADS',  # OpenBLAS
        'VECLIB_MAXIMUM_THREADS',  # macOS Accelerate
        'NUMEXPR_NUM_THREADS',  # NumExpr
        'NUMBA_NUM_THREADS',  # Numba
        'GOTO_NUM_THREADS',  # GotoBLAS
        'BLIS_NUM_THREADS',  # BLIS
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',  # SimpleITK/ITK
        'XLA_FLAGS',  # JAX/XLA thread pool (set below with special format)
    ]

    for var in thread_vars:
        if var == 'XLA_FLAGS':
            # XLA flags use a special format
            # This limits JAX's XLA thread pool (used by BaSiCPy)
            xla_flags = os.environ.get('XLA_FLAGS', '')
            if f'--xla_cpu_multi_thread_eigen=false' not in xla_flags:
                # Disable multi-threading in XLA's Eigen backend for better control
                new_flags = f'{xla_flags} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={max_threads}'.strip()
                os.environ['XLA_FLAGS'] = new_flags
        else:
            os.environ[var] = str(max_threads)

    # Also set dask configuration via environment variable
    # This limits dask's thread pool before dask is imported
    os.environ['DASK_NUM_WORKERS'] = str(max_threads)

    return max_threads


def configure_dask():
    """
    Configure dask's thread pool after dask is imported.
    Call this after dask has been imported.
    """
    try:
        import dask
        max_threads = int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count()))
        dask.config.set(num_workers=max_threads)
        dask.config.set(scheduler='threads')  # Use thread scheduler, not process
        dask.config.set({'array.slicing.split_large_chunks': False})
    except ImportError:
        pass


def configure_sitk():
    """
    Configure SimpleITK's global thread pool.
    Call this after SimpleITK has been imported.

    NOTE: This is a major source of CPU oversubscription! SimpleITK's
    ImageRegistrationMethod and other filters spawn their own thread pools
    that ignore environment variables.
    """
    try:
        import SimpleITK as sitk
        max_threads = int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count()))
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max_threads)
    except ImportError:
        pass


def apply_threadpool_limits():
    """
    Apply thread limits using threadpoolctl after libraries are loaded.

    This provides runtime control over thread pools that may not respect
    environment variables. Call this after numpy/scipy are imported.

    Returns:
        threadpoolctl.ThreadpoolController context or None if not available
    """
    try:
        from threadpoolctl import threadpool_limits

        # Get the configured thread limit
        max_threads = int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count()))

        # Apply limits globally - this returns a context manager but also applies immediately
        limiter = threadpool_limits(limits=max_threads)
        return limiter
    except ImportError:
        return None


def configure_all_libraries():
    """
    Configure thread limits for ALL known libraries that have been imported.

    This is the most comprehensive approach - call this after all imports
    are complete to ensure all libraries respect the thread limits.

    This addresses the following gaps:
    1. SimpleITK's internal thread pool (major source of CPU spikes)
    2. Dask's scheduler configuration
    3. Runtime threadpool limiting via threadpoolctl
    4. Numba's thread pool

    Returns:
        int: The configured thread limit
    """
    global _thread_config_applied

    max_threads = int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count()))

    # Configure SimpleITK if imported (CRITICAL - major source of CPU spikes)
    if 'SimpleITK' in sys.modules:
        configure_sitk()

    # Configure dask if imported
    if 'dask' in sys.modules:
        configure_dask()

    # Configure numba if imported
    if 'numba' in sys.modules:
        try:
            from numba import set_num_threads
            set_num_threads(max_threads)
        except (ImportError, Exception):
            pass

    # Apply threadpoolctl limits (catches numpy, scipy, etc.)
    apply_threadpool_limits()

    _thread_config_applied = True
    return max_threads


def get_thread_info():
    """
    Get diagnostic information about current thread configuration.
    Useful for debugging CPU usage issues.

    Returns:
        dict: Thread configuration information
    """
    info = {
        'total_cpus': multiprocessing.cpu_count(),
        'configured_threads': int(os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count())),
        'env_vars': {},
        'libraries': {},
    }

    # Check environment variables
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'LINUMPY_MAX_CPUS', 'LINUMPY_RESERVED_CPUS']:
        info['env_vars'][var] = os.environ.get(var, 'NOT SET')

    # Check SimpleITK
    if 'SimpleITK' in sys.modules:
        try:
            import SimpleITK as sitk
            info['libraries']['SimpleITK'] = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        except Exception:
            info['libraries']['SimpleITK'] = 'ERROR'

    # Check threadpoolctl
    try:
        from threadpoolctl import threadpool_info
        info['libraries']['threadpoolctl'] = threadpool_info()
    except ImportError:
        info['libraries']['threadpoolctl'] = 'NOT INSTALLED'

    return info


def print_thread_info():
    """Print thread configuration for debugging."""
    info = get_thread_info()
    print(f"CPU cores: {info['total_cpus']}")
    print(f"Configured threads: {info['configured_threads']}")
    print("Environment variables:")
    for var, val in info['env_vars'].items():
        print(f"  {var}: {val}")
    print("Library configurations:")
    for lib, val in info['libraries'].items():
        print(f"  {lib}: {val}")


def worker_initializer():
    """
    Initializer function for multiprocessing workers.

    Use this as the `initializer` argument for multiprocessing.Pool or
    concurrent.futures.ProcessPoolExecutor to ensure worker processes
    respect thread limits.

    Example:
        from multiprocessing import Pool
        from linumpy._thread_config import worker_initializer

        with Pool(n_workers, initializer=worker_initializer) as pool:
            results = pool.map(my_function, data)

    This is crucial because:
    1. Child processes inherit environment variables but re-import libraries
    2. Libraries like SimpleITK need explicit configuration after import
    3. threadpoolctl limits need to be reapplied in each worker
    """
    # Re-run the initial configuration
    configure_thread_limits()

    # Apply runtime limits
    apply_threadpool_limits()

    # Configure any imported libraries in this worker
    configure_all_libraries()


# Configure thread limits immediately when this module is imported
_configured_threads = configure_thread_limits()

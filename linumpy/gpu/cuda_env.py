"""
CUDA environment setup for JAX GPU support.

This module provides functions to automatically configure LD_LIBRARY_PATH
for JAX CUDA 12 plugin compatibility.

JAX 0.4.23 was compiled with CUDA 12 driver API and requires specific library versions:
  - libcusolver.so.11 (from nvidia-cusolver-cu12==11.5.4.101)
  - libcusparse.so.12 (from nvidia-cusparse-cu12==12.2.0.103)
  - libcufft.so.11 (from nvidia-cufft-cu12==11.0.12.1)
  - libcublas.so.12 (from nvidia-cublas-cu12==12.3.4.1)
  - libcudnn.so.8 (from nvidia-cudnn-cu12==8.9.7.29)

These exact versions must be installed - newer versions have different .so versions
that are INCOMPATIBLE with JAX 0.4.23.

Usage:
    # Before importing JAX (e.g., in scripts that use BaSiCPy):
    from linumpy.gpu.cuda_env import setup_jax_cuda_env
    setup_jax_cuda_env()

    # Then import JAX/BaSiCPy
    import jax
    from basicpy import BaSiC
"""

import os
import site
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    'setup_jax_cuda_env',
    'get_cuda12_ld_path',
    'check_patchelf_needed',
    'apply_patchelf_fix',
    'verify_jax_cuda',
]


def get_site_packages() -> str:
    """Get the primary site-packages directory."""
    return site.getsitepackages()[0]


def get_cuda12_ld_path(include_existing: bool = True) -> Tuple[str, List[str]]:
    """
    Build LD_LIBRARY_PATH for CUDA 12 compatible libraries.

    JAX 0.4.23 needs .so.11 libraries from the -cu12 packages.
    Order matters - -cu12 package paths should be listed.

    Parameters
    ----------
    include_existing : bool
        Whether to append existing LD_LIBRARY_PATH. Default True.

    Returns
    -------
    ld_path : str
        Colon-separated LD_LIBRARY_PATH string
    cuda_paths : list
        List of individual paths that were found
    """
    sp = get_site_packages()
    cuda_paths = []

    # Priority 1: Check for system CUDA 12 installation
    system_cuda_paths = [
        "/usr/local/cuda-12.4/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda/lib64",
    ]
    for cuda_path in system_cuda_paths:
        cublas_path = os.path.join(cuda_path, "libcublas.so.11")
        if os.path.exists(cublas_path):
            cuda_paths.append(cuda_path)
            break

    # Priority 2: -cu12 pip packages (pinned versions with correct library files)
    # JAX 0.4.23 needs specific .so versions from pinned nvidia packages
    cu12_lib_paths = [
        ("nvidia/cublas/lib", "libcublas.so.12"),
        ("nvidia/cuda_runtime/lib", "libcudart.so.12"),
        ("nvidia/cusolver/lib", "libcusolver.so.11"),
        ("nvidia/cusparse/lib", "libcusparse.so.12"),
        ("nvidia/cufft/lib", "libcufft.so.11"),
        ("nvidia/cudnn/lib", "libcudnn.so.8"),
        ("nvidia/nvjitlink/lib", "libnvJitLink.so.12"),
    ]

    for lib_path, check_file in cu12_lib_paths:
        full_path = os.path.join(sp, lib_path)
        if os.path.isdir(full_path):
            # Check for the expected file or any .so file
            expected = os.path.join(full_path, check_file)
            if os.path.exists(expected) or any(Path(full_path).glob("*.so*")):
                if full_path not in cuda_paths:
                    cuda_paths.append(full_path)

    # Priority 3: Check for system cuDNN 8.x (if pip package doesn't have it)
    system_cudnn_paths = ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64", "/usr/lib64"]
    for sys_path in system_cudnn_paths:
        if os.path.exists(os.path.join(sys_path, "libcudnn.so.8")):
            if sys_path not in cuda_paths:
                cuda_paths.insert(0, sys_path)  # System cuDNN first
            break

    # Build path string
    new_ld_path = ':'.join(cuda_paths)

    # Optionally append existing LD_LIBRARY_PATH
    if include_existing:
        existing = os.environ.get('LD_LIBRARY_PATH', '')
        if existing:
            new_ld_path = f"{new_ld_path}:{existing}"

    return new_ld_path, cuda_paths


def check_patchelf_needed() -> Tuple[bool, Optional[str]]:
    """
    Check if patchelf fix is needed for JAX CUDA plugin.

    Returns
    -------
    needs_fix : bool
        True if patchelf fix is needed
    plugin_path : str or None
        Path to the xla_cuda_plugin.so file, or None if not found
    """
    sp = get_site_packages()
    plugin_path = os.path.join(sp, "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so")

    if not os.path.exists(plugin_path):
        return False, None

    # Check if it has executable stack using execstack or readelf
    try:
        result = subprocess.run(
            ['readelf', '-l', plugin_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Look for GNU_STACK with RWE (read-write-execute)
            for line in result.stdout.split('\n'):
                if 'GNU_STACK' in line and 'RWE' in line:
                    return True, plugin_path
        return False, plugin_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Can't check - assume it might need fix
        return True, plugin_path


def apply_patchelf_fix(verbose: bool = False) -> bool:
    """
    Apply patchelf fix to JAX CUDA plugin and jaxlib .so files.

    Returns True if successful, False if patchelf not available.
    """
    try:
        subprocess.run(['patchelf', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        if verbose:
            print("patchelf not installed. Install with: sudo apt install patchelf")
        return False

    sp = get_site_packages()
    patched_count = 0

    # Patch jaxlib
    jaxlib = None
    try:
        import jaxlib
        jaxlib_path = jaxlib.__path__[0]
        for so_file in Path(jaxlib_path).rglob("*.so"):
            try:
                subprocess.run(
                    ['patchelf', '--clear-execstack', str(so_file)],
                    capture_output=True, check=True
                )
                patched_count += 1
            except subprocess.CalledProcessError:
                pass
    except ImportError:
        # jaxlib not installed
        jaxlib = None

    # Patch jax_plugins
    jax_plugins_path = os.path.join(sp, "jax_plugins")
    if os.path.isdir(jax_plugins_path):
        for so_file in Path(jax_plugins_path).rglob("*.so"):
            try:
                subprocess.run(
                    ['patchelf', '--clear-execstack', str(so_file)],
                    capture_output=True, check=True
                )
                patched_count += 1
            except subprocess.CalledProcessError:
                pass

    if verbose:
        print(f"Patched {patched_count} .so files")

    return patched_count > 0


def setup_jax_cuda_env(
        auto_patchelf: bool = True,
        verbose: bool = False,
        warn_on_failure: bool = True,
) -> bool:
    """
    Set up the environment for JAX CUDA support.

    This function should be called BEFORE importing JAX to ensure
    LD_LIBRARY_PATH is set correctly for CUDA 12 library loading.

    Parameters
    ----------
    auto_patchelf : bool
        Automatically apply patchelf fix if needed. Default True.
    verbose : bool
        Print diagnostic information. Default False.
    warn_on_failure : bool
        Issue warnings if setup fails. Default True.

    Returns
    -------
    success : bool
        True if environment was set up successfully.

    Example
    -------
    >>> from linumpy.gpu.cuda_env import setup_jax_cuda_env
    >>> setup_jax_cuda_env()
    >>> import jax  # Now JAX should find CUDA libraries
    >>> print(jax.devices())
    """
    # Check if JAX is already imported - warn that it may be too late
    if 'jax' in sys.modules:
        if warn_on_failure:
            warnings.warn(
                "JAX is already imported. setup_jax_cuda_env() should be called "
                "BEFORE importing JAX for LD_LIBRARY_PATH changes to take effect. "
                "You may need to restart the Python process."
            )

    # Build and set LD_LIBRARY_PATH
    new_ld_path, cuda_paths = get_cuda12_ld_path(include_existing=True)

    if not cuda_paths:
        if warn_on_failure:
            warnings.warn(
                "No CUDA 12 libraries found. Install with:\n"
                "  pip install --extra-index-url https://pypi.nvidia.com \\\n"
                "      nvidia-cusolver nvidia-cufft nvidia-cusparse \\\n"
                "      nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \\\n"
                "      nvidia-nvjitlink-cu12 nvidia-cudnn-cu12\n"
                "Or run: source scripts/fix_jax_cuda_plugin.sh"
            )
        return False

    os.environ['LD_LIBRARY_PATH'] = new_ld_path

    if verbose:
        print(f"Set LD_LIBRARY_PATH with {len(cuda_paths)} CUDA library paths")
        for p in cuda_paths:
            print(f"  {p}")

    # Check and apply patchelf fix if needed
    if auto_patchelf:
        needs_fix, plugin_path = check_patchelf_needed()
        if needs_fix:
            if verbose:
                print("Applying patchelf fix...")
            success = apply_patchelf_fix(verbose=verbose)
            if not success and warn_on_failure:
                warnings.warn(
                    "Could not apply patchelf fix. JAX CUDA may fail with "
                    "'cannot enable executable stack' error. "
                    "Install patchelf: sudo apt install patchelf"
                )

    return True


def verify_jax_cuda(verbose: bool = True) -> bool:
    """
    Verify that JAX CUDA is working correctly.

    This imports JAX and tests basic GPU operations including SVD
    which is used by BaSiCPy.

    Parameters
    ----------
    verbose : bool
        Print diagnostic information. Default True.

    Returns
    -------
    working : bool
        True if JAX CUDA is working.
    """
    try:
        import jax
        devices = jax.devices()

        has_gpu = any('cuda' in str(d).lower() for d in devices)

        if verbose:
            print(f"JAX devices: {devices}")

        if not has_gpu:
            if verbose:
                print("⚠️  JAX is using CPU only")
            return False

        # Test SVD (used by BaSiCPy)
        import jax.numpy as jnp
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = jnp.linalg.svd(a)

        if verbose:
            print(f"✅ JAX GPU working - SVD test passed")
            print(f"   Singular values: {result[1]}")

        return True

    except Exception as e:
        if verbose:
            print(f"❌ JAX CUDA verification failed: {e}")
        return False

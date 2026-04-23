"""
CUDA environment setup for GPU support.

This module provides functions to automatically configure LD_LIBRARY_PATH
for CUDA 12 library compatibility (used by torch, cupy, etc.).

Usage:
    # Before importing GPU-aware libraries:
    from linumpy.gpu.cuda_env import setup_jax_cuda_env
    setup_jax_cuda_env()
"""

import contextlib
import ctypes
import os
import site
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

__all__ = [
    "apply_patchelf_fix",
    "check_patchelf_needed",
    "ensure_cuda_env",
    "get_cuda12_ld_path",
    "get_nvidia_lib_paths",
    "preload_cuda_libraries",
    "setup_jax_cuda_env",
]

# --- Sentinel used by ensure_cuda_env ---
_CUDA_ENV_SENTINEL = "_LINUMPY_CUDA_ENV_READY"

# Pip nvidia library sub-directories (relative to site-packages).
_NVIDIA_LIB_DIRS = [
    "nvidia/cublas/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/cufft/lib",
    "nvidia/cudnn/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/nccl/lib",  # must be >=2.21.5 for torch compat (ncclCommWindowDeregister)
]

# Preload order: dependencies first; loaded with RTLD_GLOBAL.
_CUDA_LIBS_TO_PRELOAD = [
    "libcudart.so.12",
    "libnvJitLink.so.12",  # CUDA 13 ships .so.13 with different ABI
    "libnccl.so.2",  # CUDA 13 missing ncclCommWindowDeregister in older .so
    "libcudnn.so.8",  # JAX 0.4.23 / torch expect 8.x; CUDA 13 ships 9.x
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcusolver.so.11",  # JAX 0.4.23 needs .so.11
    "libcusparse.so.12",
    "libcufft.so.11",  # JAX 0.4.23 needs .so.11
]


def get_nvidia_lib_paths() -> list[str]:
    """Return all existing pip-installed nvidia CUDA library directories.

    Uses both ``sysconfig`` (most reliable in uv/virtualenv environments) and
    ``site.getsitepackages()`` as a fallback.
    """
    sp_set: set[str] = set()
    with contextlib.suppress(Exception):
        for key in ("purelib", "platlib"):
            p = sysconfig.get_path(key)
            if p:
                sp_set.add(p)
    with contextlib.suppress(Exception):
        sp_set.update(site.getsitepackages())
    paths: list[str] = []
    for sp in sp_set:
        for lib_dir in _NVIDIA_LIB_DIRS:
            p = Path(sp) / lib_dir
            if p.is_dir():
                paths.append(str(p))
    return paths


def ensure_cuda_env() -> None:
    """Ensure ``LD_LIBRARY_PATH`` contains pip nvidia libs; re-exec if not.

    Must be called **before** any GPU-aware import (torch, jax, cupy).
    Setting ``LD_LIBRARY_PATH`` mid-process via ``os.environ`` is unreliable
    because glibc initialises its linker search-path cache at process startup
    and may not re-read the variable on every ``dlopen()``.  The only
    guaranteed fix is to have the correct value in the environment when the
    process *starts*.

    This function sets the variable and uses ``os.execv()`` to replace the
    current process with an identical one that inherits the corrected
    environment.  A sentinel env-var (``_LINUMPY_CUDA_ENV_READY``) prevents
    infinite re-exec.
    """
    if os.environ.get(_CUDA_ENV_SENTINEL):
        return  # already handled in a previous exec
    nvidia_paths = get_nvidia_lib_paths()
    if not nvidia_paths:
        return  # pip nvidia packages not installed — nothing to do
    new_prefix = ":".join(nvidia_paths)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{new_prefix}:{current}" if current else new_prefix
    os.environ[_CUDA_ENV_SENTINEL] = "1"
    # Replace the current process; the new one starts with correct LD_LIBRARY_PATH
    os.execv(sys.executable, [sys.executable, *sys.argv])


def preload_cuda_libraries() -> bool:
    """Load pip nvidia libs into the process with ``RTLD_GLOBAL`` via ctypes.

    Secondary defence after :func:`ensure_cuda_env` has already set
    ``LD_LIBRARY_PATH``.  Explicitly registering symbols with ``RTLD_GLOBAL``
    ensures they are in the process namespace before every subsequent
    ``dlopen()``, even if the loader cache was seeded from a stale path.

    Returns
    -------
    bool
        ``True`` if at least one library directory was found.
    """
    search_paths = get_nvidia_lib_paths()
    if not search_paths:
        return False
    for lib in _CUDA_LIBS_TO_PRELOAD:
        for path in search_paths:
            lib_path = Path(path) / lib
            if lib_path.exists():
                with contextlib.suppress(Exception):
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                break
    return True


def get_site_packages() -> str:
    """Get the primary site-packages directory."""
    return site.getsitepackages()[0]


def get_cuda12_ld_path(include_existing: bool = True) -> tuple[str, list[str]]:
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
        cublas_path = Path(cuda_path) / "libcublas.so.11"
        if cublas_path.exists():
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
        full_path = Path(sp) / lib_path
        if full_path.is_dir():
            # Check for the expected file or any .so file
            expected = full_path / check_file
            if (expected.exists() or any(full_path.glob("*.so*"))) and str(full_path) not in cuda_paths:
                cuda_paths.append(str(full_path))

    # Priority 3: Check for system cuDNN 8.x (if pip package doesn't have it)
    system_cudnn_paths = ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64", "/usr/lib64"]
    for sys_path in system_cudnn_paths:
        if (Path(sys_path) / "libcudnn.so.8").exists():
            if sys_path not in cuda_paths:
                cuda_paths.insert(0, sys_path)  # System cuDNN first
            break

    # Build path string
    new_ld_path = ":".join(cuda_paths)

    # Optionally append existing LD_LIBRARY_PATH
    if include_existing:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            new_ld_path = f"{new_ld_path}:{existing}"

    return new_ld_path, cuda_paths


def check_patchelf_needed() -> tuple[bool, Path | None]:
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
    plugin_path = Path(sp) / "jax_plugins" / "xla_cuda12" / "xla_cuda_plugin.so"

    if not plugin_path.exists():
        return False, None

    # Check if it has executable stack using execstack or readelf
    try:
        result = subprocess.run(["readelf", "-l", plugin_path], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Look for GNU_STACK with RWE (read-write-execute)
            for line in result.stdout.split("\n"):
                if "GNU_STACK" in line and "RWE" in line:
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
        subprocess.run(["patchelf", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        if verbose:
            print("patchelf not installed. Install with: sudo apt install patchelf")
        return False

    sp = get_site_packages()
    patched_count = 0

    # Patch jax_plugins
    jax_plugins_path = Path(sp) / "jax_plugins"
    if jax_plugins_path.is_dir():
        for so_file in Path(jax_plugins_path).rglob("*.so"):
            try:
                subprocess.run(["patchelf", "--clear-execstack", str(so_file)], capture_output=True, check=True)
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
    """
    # Ensure LD_LIBRARY_PATH is correct; re-exec if needed so the linker
    # search-path cache is built with the pip nvidia paths from the start.
    ensure_cuda_env()

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
                "Or run: source scripts/fix_jax_cuda_plugin.sh",
                stacklevel=2,
            )
        return False

    os.environ["LD_LIBRARY_PATH"] = new_ld_path

    if verbose:
        print(f"Set LD_LIBRARY_PATH with {len(cuda_paths)} CUDA library paths")
        for p in cuda_paths:
            print(f"  {p}")

    # Check and apply patchelf fix if needed
    if auto_patchelf:
        needs_fix, _plugin_path = check_patchelf_needed()
        if needs_fix:
            if verbose:
                print("Applying patchelf fix...")
            success = apply_patchelf_fix(verbose=verbose)
            if not success and warn_on_failure:
                warnings.warn(
                    "Could not apply patchelf fix. JAX CUDA may fail with "
                    "'cannot enable executable stack' error. "
                    "Install patchelf: sudo apt install patchelf",
                    stacklevel=2,
                )

    return True

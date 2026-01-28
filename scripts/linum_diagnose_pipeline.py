#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script for linumpy 3D reconstruction pipeline performance.

This script checks the server configuration to identify bottlenecks:
- CPU core detection and thread configuration
- GPU availability and CUDA setup
- Memory availability
- Nextflow parameter recommendations
- Performance baseline tests

Usage:
    linum_diagnose_pipeline.py                    # Full diagnostic
    linum_diagnose_pipeline.py --quick            # Quick system check only
    linum_diagnose_pipeline.py --benchmark        # Include performance benchmarks
    linum_diagnose_pipeline.py --debug-cuda       # Detailed CUDA library debugging
    linum_diagnose_pipeline.py --verbose          # Show full error tracebacks
    linum_diagnose_pipeline.py --output report.txt  # Save results to file
"""

import argparse
import glob
import json
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime


def get_terminal_width():
    """Get terminal width for formatting."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def print_header(title: str):
    """Print a section header."""
    width = get_terminal_width()
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


class SystemDiagnostics:
    """System diagnostics collector."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {},
            "memory": {},
            "gpu": {},
            "python": {},
            "nextflow": {},
            "linumpy": {},
            "issues": [],
            "recommendations": [],
        }

    def check_cpu(self):
        """Check CPU configuration."""
        print_header("CPU Configuration")

        # Physical CPU info
        total_cpus = multiprocessing.cpu_count()
        self.results["cpu"]["total_cores"] = total_cpus
        print(f"  Total CPU cores detected: {total_cpus}")

        # Check environment variables that control threading
        print_subheader("Thread Environment Variables")
        thread_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'NUMBA_NUM_THREADS',
            'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
            'XLA_FLAGS',
            'LINUMPY_MAX_CPUS',
            'LINUMPY_RESERVED_CPUS',
        ]

        for var in thread_vars:
            value = os.environ.get(var, "(not set)")
            self.results["cpu"][var] = value
            print(f"  {var}: {value}")

        return total_cpus

    def check_memory(self):
        """Check memory availability."""
        print_header("Memory Configuration")

        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)

            self.results["memory"]["total_gb"] = round(total_gb, 1)
            self.results["memory"]["available_gb"] = round(available_gb, 1)
            self.results["memory"]["percent_used"] = mem.percent

            print(f"  Total RAM: {total_gb:.1f} GB")
            print(f"  Available RAM: {available_gb:.1f} GB")
            print(f"  Memory usage: {mem.percent}%")

            if available_gb < 16:
                print(f"  ⚠️  Low available memory - may cause swapping")
                self.results["issues"].append(f"Low available memory: {available_gb:.1f} GB")

            return total_gb, available_gb

        except ImportError:
            print("  ⚠️  psutil not installed - cannot check memory")
            print("     Install with: pip install psutil")
            self.results["memory"]["error"] = "psutil not installed"
            return None, None

    def check_gpu(self):
        """Check GPU configuration and CUDA availability."""
        print_header("GPU Configuration")

        # Check nvidia-smi
        print_subheader("NVIDIA Driver")
        try:
            simple_result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True, text=True, timeout=30
            )
            if simple_result.returncode == 0 and simple_result.stdout.strip():
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,cuda_version',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpus = []
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            try:
                                gpu_info = {
                                    "id": i,
                                    "name": parts[0],
                                    "memory_total_mb": int(float(parts[1])),
                                    "memory_free_mb": int(float(parts[2])),
                                    "driver_version": parts[3],
                                    "cuda_version": parts[4],
                                }
                                gpus.append(gpu_info)
                                print(f"  GPU {i}: {parts[0]}")
                                print(f"    Memory: {int(float(parts[1]))/1024:.1f} GB total, {int(float(parts[2]))/1024:.1f} GB free")
                                print(f"    Driver: {parts[3]}, CUDA: {parts[4]}")
                            except (ValueError, IndexError):
                                print(f"  ⚠️  Could not parse GPU {i} info: {line}")

                    self.results["gpu"]["available"] = len(gpus) > 0
                    self.results["gpu"]["devices"] = gpus
                    if gpus:
                        print(f"  ✅ Found {len(gpus)} GPU(s)")
                else:
                    gpus = []
                    for line in simple_result.stdout.strip().split('\n'):
                        if line.startswith('GPU '):
                            gpus.append({"name": line})
                            print(f"  {line}")
                    self.results["gpu"]["available"] = len(gpus) > 0
                    self.results["gpu"]["devices"] = gpus
                    if gpus:
                        print(f"  ✅ Found {len(gpus)} GPU(s)")
            else:
                if simple_result.stderr:
                    print(f"  ⚠️  nvidia-smi error: {simple_result.stderr.strip()[:100]}")
                else:
                    print("  ⚠️  nvidia-smi returned no output")
                self.results["gpu"]["available"] = False

        except FileNotFoundError:
            print("  ⚠️  nvidia-smi not found - no NVIDIA driver installed")
            self.results["gpu"]["available"] = False
        except subprocess.TimeoutExpired:
            print("  ⚠️  nvidia-smi timed out - GPU may be busy or stuck")
            self.results["gpu"]["available"] = False
        except Exception as e:
            print(f"  ⚠️  nvidia-smi error: {e}")
            self.results["gpu"]["available"] = False

        # Check JAX (used by BaSiCPy for fix_illumination)
        print_subheader("JAX (BaSiCPy backend)")
        self._check_jax_gpu()

        # Check CuPy
        print_subheader("CuPy (GPU Python)")
        self._check_cupy()

        # Check linumpy GPU module
        print_subheader("Linumpy GPU Module")
        self._check_linumpy_gpu()

    def _check_jax_gpu(self):
        """Check JAX GPU in a subprocess with proper CUDA 12 library paths."""
        try:
            new_ld_path, cuda12_paths = self._get_cuda12_ld_path(debug=False)

            if cuda12_paths:
                print(f"  Found {len(cuda12_paths)} CUDA 12 library paths")

            jax_check_code = '''
import sys
import os
import ctypes

# Preload CUDA libraries before importing JAX
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
paths = [p for p in ld_path.split(':') if p]
for lib in ['libcudart.so.12', 'libcublas.so.12', 'libcusolver.so.12']:
    for path in paths:
        lib_path = os.path.join(path, lib)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except:
                pass
            break

try:
    import jax
    print(f"VERSION:{jax.__version__}")
    devices = jax.devices()
    device_strs = [str(d) for d in devices]
    platforms = [str(d.platform) for d in devices]
    print(f"DEVICES:{','.join(device_strs)}")
    print(f"PLATFORMS:{','.join(platforms)}")
    has_gpu = 'gpu' in platforms or 'cuda' in platforms or any('cuda' in d.lower() for d in device_strs)
    print(f"HAS_GPU:{has_gpu}")
except Exception as e:
    print(f"ERROR:{e}")
    sys.exit(1)
'''
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = new_ld_path

            result = subprocess.run(
                [sys.executable, '-c', jax_check_code],
                capture_output=True, text=True, timeout=60, env=env
            )

            if result.returncode == 0:
                jax_version = None
                jax_devices = []
                jax_has_gpu = False

                for line in result.stdout.strip().split('\n'):
                    if line.startswith('VERSION:'):
                        jax_version = line.split(':', 1)[1]
                    elif line.startswith('DEVICES:'):
                        jax_devices = line.split(':', 1)[1].split(',') if line.split(':', 1)[1] else []
                    elif line.startswith('HAS_GPU:'):
                        jax_has_gpu = line.split(':', 1)[1] == 'True'

                print(f"  JAX version: {jax_version}")
                self.results["gpu"]["jax_version"] = jax_version
                print(f"  JAX devices: {jax_devices}")

                if jax_has_gpu:
                    print("  ✅ JAX GPU support is enabled")
                    self.results["gpu"]["jax_gpu"] = True
                else:
                    print("  ⚠️  JAX is using CPU only")
                    print("     To enable GPU: pip install 'jax[cuda12]<=0.4.23'")
                    self.results["gpu"]["jax_gpu"] = False
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                print(f"  ⚠️  JAX GPU check failed")
                self.results["gpu"]["jax_gpu"] = False
                self._handle_jax_error(error_msg)

        except subprocess.TimeoutExpired:
            print("  ⚠️  JAX check timed out")
            self.results["gpu"]["jax_gpu"] = False
        except Exception as e:
            print(f"  ⚠️  JAX check error: {e}")
            self.results["gpu"]["jax_gpu"] = False

    def _handle_jax_error(self, error_msg: str):
        """Handle JAX error messages with helpful guidance."""
        if "libcublas" in error_msg.lower() or "cannot open shared object" in error_msg or "cusolver" in error_msg.lower():
            print("     CUDA library issue. JAX 0.4.23 requires specific library versions.")
            print("")
            print("     Run the fix script:")
            print("       source scripts/fix_jax_cuda_plugin.sh")
            print("")
            print("     This installs pinned nvidia package versions:")
            print("       nvidia-cusolver-cu12==11.5.4.101  (libcusolver.so.11)")
            print("       nvidia-cublas-cu12==12.3.4.1     (libcublas.so.12)")
            print("       nvidia-cudnn-cu12==8.9.7.29      (libcudnn.so.8)")
            print("       etc.")
            print("")
            print("     Then set LD_LIBRARY_PATH - see docs/GPU_ACCELERATION.md")
            self.results["issues"].append(
                "CUDA library issue - run: source scripts/fix_jax_cuda_plugin.sh"
            )
        elif "cannot enable executable stack" in error_msg:
            print("     JAX CUDA plugin blocked by kernel security.")
            print("     Fix with: sudo apt install patchelf")
            print("     Then: source scripts/fix_jax_cuda_plugin.sh")
            self.results["issues"].append("JAX CUDA plugin needs patchelf fix")
        elif "PJRT_Api not found" in error_msg or "pjrt_plugin" in error_msg.lower():
            print("     JAX CUDA plugin conflict.")
            print("     Run: source scripts/fix_jax_cuda_plugin.sh")
            self.results["issues"].append("JAX CUDA plugin conflict")
        elif "triton" in error_msg.lower():
            print("     jax-cuda13-plugin requires newer JAX. Uninstall it:")
            print("       pip uninstall jax-cuda13-plugin -y")
            self.results["issues"].append("jax-cuda13-plugin incompatible")
        else:
            print(f"     Error: {error_msg[:200]}")

    def _check_cupy(self):
        """Check CuPy GPU support."""
        try:
            import cupy as cp
            print(f"  ✅ CuPy version: {cp.__version__}")
            self.results["gpu"]["cupy_version"] = cp.__version__

            try:
                n_devices = cp.cuda.runtime.getDeviceCount()
                print(f"  ✅ CUDA devices available: {n_devices}")

                for i in range(n_devices):
                    with cp.cuda.Device(i):
                        free, total = cp.cuda.runtime.memGetInfo()
                        print(f"    Device {i}: {free/(1024**3):.1f} GB free / {total/(1024**3):.1f} GB total")

                test_array = cp.random.rand(1000, 1000)
                _ = cp.fft.fft2(test_array)
                cp.cuda.Stream.null.synchronize()
                print("  ✅ GPU computation test passed")
                self.results["gpu"]["cupy_working"] = True

            except Exception as e:
                print(f"  ⚠️  CuPy CUDA error: {e}")
                self.results["gpu"]["cupy_working"] = False
                self.results["issues"].append(f"CuPy CUDA error: {e}")

        except ImportError:
            print("  ⚠️  CuPy not installed - GPU acceleration disabled for linumpy")
            print("     Install with: pip install cupy-cuda12x (or appropriate CUDA version)")
            self.results["gpu"]["cupy_version"] = None
            self.results["issues"].append("CuPy not installed - linumpy GPU acceleration disabled")

    def _check_linumpy_gpu(self):
        """Check linumpy GPU module."""
        try:
            from linumpy.gpu import GPU_AVAILABLE, GPU_DEVICE_NAME, GPU_MEMORY_GB
            print(f"  GPU_AVAILABLE: {GPU_AVAILABLE}")
            if GPU_AVAILABLE:
                print(f"  GPU_DEVICE_NAME: {GPU_DEVICE_NAME}")
                print(f"  GPU_MEMORY_GB: {GPU_MEMORY_GB:.1f}")
            self.results["gpu"]["linumpy_gpu_available"] = GPU_AVAILABLE
        except ImportError as e:
            print(f"  ⚠️  Cannot import linumpy.gpu: {e}")
            self.results["gpu"]["linumpy_gpu_available"] = False
        except Exception as e:
            print(f"  ⚠️  Error checking linumpy.gpu: {e}")
            self.results["gpu"]["linumpy_gpu_available"] = False

    def check_python_packages(self):
        """Check critical Python packages."""
        print_header("Python Environment")

        print(f"  Python version: {sys.version}")
        self.results["python"]["version"] = sys.version

        print_subheader("Critical Packages")

        packages = [
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("basicpy", "basicpy"),
            ("jax", "jax"),
            ("jaxlib", "jaxlib"),
            ("pqdm", "pqdm"),
            ("dask", "dask"),
            ("zarr", "zarr"),
            ("threadpoolctl", "threadpoolctl"),
            ("psutil", "psutil"),
        ]

        for name, import_name in packages:
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  ✅ {name}: {version}")
                self.results["python"][name] = version
            except ImportError:
                print(f"  ❌ {name}: NOT INSTALLED")
                self.results["python"][name] = None
                if name in ["basicpy", "pqdm"]:
                    self.results["issues"].append(f"{name} not installed - required for pipeline")

        # Check numpy BLAS backend
        print_subheader("NumPy BLAS Configuration")
        try:
            import numpy as np
            try:
                blas_info = np.show_config(mode='dicts')
                if blas_info and 'Build Dependencies' in blas_info:
                    blas = blas_info.get('Build Dependencies', {}).get('blas', {})
                    print(f"  BLAS: {blas.get('name', 'unknown')}")
                    self.results["python"]["blas"] = blas.get('name', 'unknown')
            except Exception:
                print("  (Could not determine BLAS configuration)")
        except Exception:
            print("  (Could not check BLAS)")

    def check_nextflow_config(self):
        """Check Nextflow configuration recommendations."""
        print_header("Nextflow Configuration")

        total_cpus = self.results["cpu"]["total_cores"]
        total_memory = self.results["memory"].get("total_gb", 0)

        print_subheader("Current Environment")

        nf_process_name = os.environ.get('NXF_TASK_NAME', 'Not in Nextflow process')
        print(f"  NXF_TASK_NAME: {nf_process_name}")

        try:
            result = subprocess.run(['nextflow', '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.strip().split('\n')[0]
                print(f"  Nextflow: {version_line}")
                self.results["nextflow"]["installed"] = True
            else:
                print("  Nextflow: installed but version check failed")
                self.results["nextflow"]["installed"] = True
        except FileNotFoundError:
            print("  Nextflow: not found in PATH")
            self.results["nextflow"]["installed"] = False
        except Exception:
            print("  Nextflow: could not check version")

        print_subheader("Suggested Nextflow Parameters")

        reserved_cpus = max(2, total_cpus // 12)
        available_cpus = total_cpus - reserved_cpus
        suggested_processes = min(16, max(1, available_cpus // 3))

        suggestions = {
            "params.processes": suggested_processes,
            "params.enable_cpu_limits": True,
            "params.reserved_cpus": reserved_cpus,
            "params.use_gpu": self.results["gpu"].get("available", False),
        }

        for param, value in suggestions.items():
            print(f"  {param} = {value}")

        self.results["nextflow"]["suggestions"] = suggestions

        print_subheader("Notes")
        print(f"  • With {total_cpus} CPU cores, you can run up to {suggested_processes} parallel processes")
        jax_status = 'GPU-enabled' if self.results['gpu'].get('jax_gpu') else 'CPU-only'
        print(f"  • fix_illumination step uses BaSiC algorithm (JAX-based, {jax_status})")
        print(f"  • Each BaSiC process typically uses ~3 CPU threads")
        if total_memory:
            print(f"  • With {total_memory:.0f} GB RAM, memory should not be a bottleneck")

    def run_benchmarks(self):
        """Run performance benchmarks."""
        print_header("Performance Benchmarks")

        print_subheader("BaSiC Algorithm (fix_illumination bottleneck)")
        self._run_basic_benchmark()

        print_subheader("Parallel Processing (pqdm)")
        self._run_pqdm_benchmark()

        if self.results["gpu"].get("cupy_working"):
            print_subheader("GPU Performance")
            self._run_gpu_benchmark()

    def _get_cuda12_ld_path(self, debug=False):
        """Build LD_LIBRARY_PATH for CUDA 12 compatible libraries."""
        import site
        site_packages = site.getsitepackages()[0]

        cuda_paths = []

        # Method 1: Check for system CUDA 12 installation
        system_cuda_paths = [
            "/usr/local/cuda-12.4/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda/lib64",
        ]
        for cuda_path in system_cuda_paths:
            cublas_path = os.path.join(cuda_path, "libcublas.so.12")
            if os.path.exists(cublas_path):
                cuda_paths.append(cuda_path)
                if debug:
                    print(f"    Found system CUDA 12: {cuda_path}")
                break

        # Method 2: -cu12 pip packages (pinned versions with correct library files)
        # JAX 0.4.23 needs specific library versions from pinned nvidia packages
        cu12_lib_dirs = [
            "nvidia/cublas/lib",
            "nvidia/cuda_runtime/lib",
            "nvidia/cusolver/lib",
            "nvidia/cusparse/lib",
            "nvidia/cufft/lib",
            "nvidia/cudnn/lib",
            "nvidia/nvjitlink/lib",
        ]
        for lib_dir in cu12_lib_dirs:
            full_path = os.path.join(site_packages, lib_dir)
            if os.path.isdir(full_path) and full_path not in cuda_paths:
                cuda_paths.append(full_path)
                if debug:
                    print(f"    Found nvidia lib: {full_path}")

        # Build clean LD_LIBRARY_PATH
        new_ld_path = ':'.join(cuda_paths)

        return new_ld_path, cuda_paths

    def _run_basic_benchmark(self):
        """Run BaSiC benchmark in subprocess with proper LD_LIBRARY_PATH for CUDA."""
        try:
            new_ld_path, cuda_paths = self._get_cuda12_ld_path(debug=False)

            if not cuda_paths:
                print("  ⚠️  No CUDA libraries found")
                print("     Run: source scripts/fix_jax_cuda_plugin.sh")
                return

            # Show paths being used
            print(f"  Using {len(cuda_paths)} CUDA library paths")

            benchmark_code = '''
import sys
import os
import ctypes

# Debug: print LD_LIBRARY_PATH
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
paths = [p for p in ld_path.split(':') if p]
print(f"DEBUG_LD_PATHS:{len(paths)}")

# Debug: Check if key .so.12 files exist in LD_LIBRARY_PATH
for p in paths[:3]:  # Check first 3 paths
    import glob
    so12_files = glob.glob(os.path.join(p, "*.so.12"))
    if so12_files:
        print(f"DEBUG_SO12_PATH:{p}:{len(so12_files)}")

# CRITICAL: Preload CUDA libraries BEFORE importing JAX
# This ensures the correct libraries from cu13/lib are used
print("DEBUG_STAGE:preloading_cuda")
for lib in ['libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12', 'libcusolver.so.12', 'libcufft.so.12', 'libcusparse.so.12']:
    for path in paths:
        lib_path = os.path.join(path, lib)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except:
                pass
            break

import time
import numpy as np

try:
    # Try to import JAX first to get better error messages
    print("DEBUG_STAGE:importing_jax")
    import jax
    
    # Try to initialize JAX with CUDA before importing BaSiC
    print("DEBUG_STAGE:checking_devices")
    devices = jax.devices()
    device_strs = [str(d) for d in devices]
    print(f"DEBUG_JAX_DEVICES:{','.join(device_strs)}")
    
    has_gpu = any('cuda' in str(d).lower() for d in devices)
    mode = "GPU" if has_gpu else "CPU"
    print(f"DEBUG_JAX_MODE:{mode}")
    
    print("DEBUG_STAGE:importing_basicpy")
    from basicpy import BaSiC

    tiles = np.random.rand(16, 256, 256).astype(np.float32) * 1000

    print("DEBUG_STAGE:fitting")
    start = time.perf_counter()
    optimizer = BaSiC(get_darkfield=False, max_iterations=100)
    optimizer.fit(tiles)
    elapsed = time.perf_counter() - start

    print(f"SUCCESS:{elapsed:.2f}:{mode}")
except Exception as e:
    import traceback
    print(f"ERROR:{e}")
    traceback.print_exc()
    sys.exit(1)
'''

            # Create a clean environment with ONLY our verified CUDA paths
            # Don't use os.environ.copy() to avoid inheriting polluted paths
            env = {}
            # Copy essential env vars but NOT LD_LIBRARY_PATH
            essential_vars = ['PATH', 'HOME', 'USER', 'LANG', 'TERM', 'SHELL',
                            'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYENV_ROOT']
            for var in essential_vars:
                if var in os.environ:
                    env[var] = os.environ[var]

            # Set our clean LD_LIBRARY_PATH
            env['LD_LIBRARY_PATH'] = new_ld_path

            # Debug: show what we're setting
            if self.verbose:
                print(f"  Setting clean LD_LIBRARY_PATH: {new_ld_path}")

            result = subprocess.run(
                [sys.executable, '-c', benchmark_code],
                capture_output=True, text=True, timeout=120, env=env
            )

            # Parse debug info and result
            success_line = None
            jax_mode = None
            last_stage = None
            for line in result.stdout.strip().split('\n'):
                if line.startswith('DEBUG_LD_PATHS:'):
                    n_paths = line.split(':')[1]
                    print(f"  LD_LIBRARY_PATH has {n_paths} entries")
                elif line.startswith('DEBUG_SO12_PATH:'):
                    parts = line.split(':')
                    path, count = parts[1], parts[2]
                    print(f"  Found {count} .so.12 files in: {os.path.basename(path)}/")
                elif line.startswith('DEBUG_JAX_DEVICES:'):
                    devices = line.split(':')[1]
                    print(f"  JAX devices: {devices}")
                elif line.startswith('DEBUG_JAX_MODE:'):
                    jax_mode = line.split(':')[1]
                elif line.startswith('DEBUG_STAGE:'):
                    last_stage = line.split(':')[1]
                elif line.startswith('SUCCESS:'):
                    success_line = line

            if success_line:
                parts = success_line.split(':')
                elapsed = float(parts[1])
                mode = parts[2]
                print(f"  BaSiC fit 16 tiles @ (256, 256): {elapsed:.2f}s ({mode})")
                self.results["linumpy"]["basic_16x256"] = elapsed
                if mode == "GPU":
                    print("  ✅ JAX GPU acceleration working")
                else:
                    print("  ⚠️  Running on CPU - check CUDA 12 library setup")
            else:
                # Real failure - analyze the error
                error_out = result.stderr + result.stdout
                if last_stage:
                    print(f"  Failed during stage: {last_stage}")
                self._handle_basic_error(error_out)

                # Show full traceback if verbose
                if self.verbose:
                    print("\n  --- Full subprocess output ---")
                    print("  STDOUT:")
                    for line in result.stdout.split('\n')[-20:]:
                        print(f"    {line}")
                    print("  STDERR:")
                    for line in result.stderr.split('\n')[-30:]:
                        print(f"    {line}")

        except subprocess.TimeoutExpired:
            print(f"  ⚠️  BaSiC benchmark timed out (>120s)")
        except Exception as e:
            print(f"  ⚠️  BaSiC benchmark failed: {e}")

    def _handle_basic_error(self, error_out: str):
        """Handle BaSiC benchmark errors with specific guidance."""
        error_lower = error_out.lower()

        if "libcublas" in error_lower or "cannot open shared object" in error_lower:
            print("  ❌ BaSiC failed: CUDA 12 libraries not in LD_LIBRARY_PATH")
            print("")
            print("     The nvidia-*-cu12 packages are installed but not accessible.")
            print("     You need to set LD_LIBRARY_PATH before running:")
            print("")
            print("     Quick test:")
            print("       jax_cuda12_env() {")
            print("         local sp=$(python -c \"import site; print(site.getsitepackages()[0])\")")
            print("         echo \"${sp}/nvidia/cublas/lib:${sp}/nvidia/cuda_runtime/lib:${sp}/nvidia/cudnn/lib:${sp}/nvidia/cufft/lib:${sp}/nvidia/cusolver/lib:${sp}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}\"")
            print("       }")
            print("       LD_LIBRARY_PATH=$(jax_cuda12_env) linum_diagnose_pipeline.py --benchmark")
            print("")
            print("     See docs/GPU_ACCELERATION.md for permanent setup.")
        elif "cuda_plugin_extension" in error_lower and "not found" in error_lower:
            # This is a JAX plugin loading issue, not a library path issue
            print("  ❌ BaSiC failed: JAX CUDA plugin extension not found")
            print("")
            print("     The JAX CUDA plugin failed to load its native extension.")
            print("     This happens when the plugin can't find compatible CUDA libraries at load time.")
            print("")
            print("     The 'cuda_plugin_extension is not found' message means JAX fell back to CPU")
            print("     but then tried to use GPU operations which failed.")
            print("")
            print("     FIX: Reinstall JAX CUDA plugin with correct LD_LIBRARY_PATH set FIRST:")
            print("")
            print("       # 1. Set LD_LIBRARY_PATH before reinstalling")
            print("       SP=$(python -c \"import site; print(site.getsitepackages()[0])\")")
            print("       export LD_LIBRARY_PATH=\"${SP}/nvidia/cublas/lib:${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/nvjitlink/lib:${SP}/nvidia/cudnn/lib:${SP}/nvidia/cu13/lib\"")
            print("")
            print("       # 2. Reinstall JAX CUDA plugin")
            print("       pip uninstall jax-cuda12-plugin jax-cuda12-pjrt -y")
            print("       pip install jax-cuda12-plugin==0.4.23")
            print("")
            print("       # 3. Test")
            print("       python -c \"import jax; print(jax.devices())\"")
        elif "build_gesvd_descriptor" in error_lower or "cusolver" in error_lower:
            print("  ❌ BaSiC failed: CUDA cusolver library issue")
            print("")
            if "nonetype" in error_lower or "none" in error_lower:
                print("     JAX couldn't initialize the cusolver library.")
                print("     This is usually caused by the CUDA plugin failing to load.")
                print("")
                print("     Check for 'cuda_plugin_extension is not found' warning when importing JAX.")
                print("     If present, the plugin needs to be reinstalled with LD_LIBRARY_PATH set.")
                print("")
                print("     FIX:")
                print("       # Set LD_LIBRARY_PATH first")
                print("       SP=$(python -c \"import site; print(site.getsitepackages()[0])\")")
                print("       export LD_LIBRARY_PATH=\"${SP}/nvidia/cublas/lib:${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/cusolver/lib:${SP}/nvidia/cudnn/lib\"")
                print("")
                print("       # Or run the fix script:")
                print("       source scripts/fix_jax_cuda_plugin.sh")
            else:
                print("     cusolver initialization failed. Check JAX CUDA plugin status:")
                print("       python -c \"import jax; print(jax.devices())\"")
        elif "executable stack" in error_lower or "enable executable stack" in error_lower:
            print("  ❌ BaSiC failed: Kernel blocking JAX CUDA plugin (executable stack)")
            print("")
            print("     The kernel is blocking xla_cuda_plugin.so because it has an executable stack.")
            print("     This is a security feature in modern Linux kernels.")
            print("")
            print("     FIX:")
            print("       # Install patchelf")
            print("       sudo apt install patchelf")
            print("")
            print("       # Clear the executable stack flag")
            print("       patchelf --clear-execstack $(python -c \"import jax_plugins.xla_cuda12; print(jax_plugins.xla_cuda12.__path__[0])\")/xla_cuda_plugin.so")
            print("")
            print("       # Test")
            print("       python -c \"import jax; print(jax.devices())\"")
            print("")
            print("     NOTE: You need to re-run patchelf after reinstalling jax-cuda12-plugin.")
        elif "cuda_plugin_extension" in error_lower or "xla_cuda12" in error_lower:
            print("  ❌ BaSiC failed: JAX CUDA plugin issue")
            print("")
            # Check for specific sub-errors
            if "undefined symbol" in error_lower:
                print("     The JAX CUDA plugin has undefined symbols - likely a library version mismatch.")
                print("")
                print("     Run: source scripts/fix_jax_cuda_plugin.sh")
                print("     This installs pinned nvidia package versions compatible with JAX 0.4.23.")
            elif "no module" in error_lower or "import" in error_lower:
                print("     The JAX CUDA plugin module failed to import.")
                print("")
                print("     Run: source scripts/fix_jax_cuda_plugin.sh")
            else:
                print("     The JAX CUDA 12 plugin failed to load.")
                print("")
                print("     Run: source scripts/fix_jax_cuda_plugin.sh")
            print("")
            # Show the actual error for debugging
            print("     Full error (for debugging):")
            # Extract just the last exception line
            for line in reversed(error_out.split('\n')):
                line = line.strip()
                if line and not line.startswith('Traceback') and not line.startswith('File '):
                    print(f"       {line[:150]}")
                    break
            # In verbose mode, show the full traceback
            if self.verbose:
                print("")
                print("     Complete output (--verbose mode):")
                for line in error_out.split('\n')[-30:]:  # Last 30 lines
                    print(f"       {line}")
        else:
            # Show the actual error
            for line in error_out.split('\n'):
                if line.startswith('ERROR:'):
                    print(f"  ❌ BaSiC failed: {line[6:]}")
                    break
            else:
                print(f"  ❌ BaSiC failed: {error_out[:200]}")
            # In verbose mode, show more
            if self.verbose:
                print("")
                print("     Complete output (--verbose mode):")
                for line in error_out.split('\n')[-30:]:
                    print(f"       {line}")

    def _run_pqdm_benchmark(self):
        """Run pqdm parallel processing benchmark."""
        try:
            from pqdm.processes import pqdm
            import numpy as np

            def dummy_task(i):
                arr = np.random.rand(500, 500)
                for _ in range(10):
                    arr = np.fft.fft2(arr)
                return i

            for n_jobs in [1, 4, 8, 16]:
                start = time.perf_counter()
                results = pqdm(range(16), dummy_task, n_jobs=n_jobs,
                              desc=f"pqdm n_jobs={n_jobs}", disable=True)
                elapsed = time.perf_counter() - start
                print(f"  pqdm with n_jobs={n_jobs}: {elapsed:.2f}s for 16 tasks")
                self.results["linumpy"][f"pqdm_njobs{n_jobs}"] = round(elapsed, 2)

        except Exception as e:
            print(f"  ⚠️  pqdm benchmark failed: {e}")

    def _run_gpu_benchmark(self):
        """Run GPU performance benchmark."""
        try:
            import cupy as cp
            import numpy as np

            size = 2048

            cpu_data = np.random.rand(size, size).astype(np.float32)
            start = time.perf_counter()
            for _ in range(5):
                _ = np.fft.fft2(cpu_data)
            cpu_time = (time.perf_counter() - start) / 5

            gpu_data = cp.asarray(cpu_data)
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                _ = cp.fft.fft2(gpu_data)
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.perf_counter() - start) / 5

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0

            print(f"  FFT {size}x{size}: CPU {cpu_time*1000:.1f}ms, GPU {gpu_time*1000:.1f}ms ({speedup:.1f}x speedup)")

            self.results["linumpy"]["fft_cpu_ms"] = round(cpu_time * 1000, 1)
            self.results["linumpy"]["fft_gpu_ms"] = round(gpu_time * 1000, 1)
            self.results["linumpy"]["fft_speedup"] = round(speedup, 1)

        except Exception as e:
            print(f"  ⚠️  GPU benchmark failed: {e}")

    def debug_cuda_libraries(self):
        """Show detailed CUDA library debugging information."""
        import site

        print_header("CUDA Library Debug Information")

        # Get site-packages
        sp = site.getsitepackages()[0]
        print(f"  Site-packages: {sp}")

        # Check current LD_LIBRARY_PATH
        print_subheader("Current LD_LIBRARY_PATH")
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if ld_path:
            for i, p in enumerate(ld_path.split(':')):
                if p:
                    print(f"  [{i}] {p}")
        else:
            print("  (not set)")

        # Check for nvidia packages installed
        print_subheader("Installed NVIDIA Packages")
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_path):
            subdirs = sorted(os.listdir(nvidia_path))
            for subdir in subdirs:
                subdir_path = os.path.join(nvidia_path, subdir)
                if os.path.isdir(subdir_path):
                    lib_path = os.path.join(subdir_path, "lib")
                    if os.path.isdir(lib_path):
                        so_files = glob.glob(os.path.join(lib_path, "*.so*"))
                        so12_count = len([f for f in so_files if '.so.12' in f])
                        so11_count = len([f for f in so_files if '.so.11' in f])
                        so13_count = len([f for f in so_files if '.so.13' in f])
                        version_info = []
                        if so11_count: version_info.append(f"{so11_count} .so.11")
                        if so12_count: version_info.append(f"{so12_count} .so.12")
                        if so13_count: version_info.append(f"{so13_count} .so.13")
                        print(f"  nvidia/{subdir}/lib: {len(so_files)} .so files "
                              f"({', '.join(version_info) if version_info else 'no versioned'})")
        else:
            print("  No nvidia packages found in site-packages")

        # Check what's in the individual nvidia/xxx/lib paths
        print_subheader("Libraries in -cu12 Package Directories")
        cu12_dirs = ['cublas', 'cuda_runtime', 'nvjitlink', 'cudnn', 'cufft']
        for dir_name in cu12_dirs:
            lib_path = os.path.join(sp, "nvidia", dir_name, "lib")
            if os.path.isdir(lib_path):
                so_files = glob.glob(os.path.join(lib_path, "lib*.so*"))
                versioned_files = [os.path.basename(f) for f in so_files if '.so.' in os.path.basename(f)]
                if versioned_files:
                    print(f"  nvidia/{dir_name}/lib: {', '.join(sorted(versioned_files)[:4])}")

        # Check nvidia/cu13/lib (from non-suffixed packages - should NOT be installed)
        print_subheader("Non-suffixed Package Libraries (nvidia/cu13/lib)")
        cu13_lib = os.path.join(sp, "nvidia", "cu13", "lib")
        if os.path.isdir(cu13_lib):
            so_files = glob.glob(os.path.join(cu13_lib, "*.so*"))
            print(f"  ⚠️  nvidia/cu13/lib exists with {len(so_files)} files")
            print("  This may indicate non-suffixed nvidia packages are installed.")
            print("  These are INCOMPATIBLE with JAX 0.4.23.")
            print("  Run: source scripts/fix_jax_cuda_plugin.sh")
        else:
            print("  ✅ nvidia/cu13/lib does not exist (correct)")

        # Check key libraries needed by JAX
        print_subheader("Key Libraries for JAX CUDA 12 Plugin")
        # JAX 0.4.23 needs these specific library versions from pinned nvidia packages
        # Format: (lib_name, pkg_name, alt_names) - alt_names for case variations
        key_libs = [
            ("libcusolver.so.11", "nvidia-cusolver-cu12==11.5.4.101", []),
            ("libcublas.so.12", "nvidia-cublas-cu12==12.3.4.1", []),
            ("libcublasLt.so.12", "nvidia-cublas-cu12==12.3.4.1", []),
            ("libcudnn.so.8", "nvidia-cudnn-cu12==8.9.7.29", []),
            ("libcufft.so.11", "nvidia-cufft-cu12==11.0.12.1", []),
            ("libcusparse.so.12", "nvidia-cusparse-cu12==12.2.0.103", []),
            ("libnvjitlink.so.12", "nvidia-nvjitlink-cu12==12.3.101", ["libnvJitLink.so.12"]),
            ("libcudart.so.12", "nvidia-cuda-runtime-cu12==12.3.101", []),
        ]
        # Build list of all paths to check
        check_paths = []
        # Individual -cu12 package paths
        cu12_pkg_paths = [
            "nvidia/cublas/lib", "nvidia/cuda_runtime/lib", "nvidia/nvjitlink/lib",
            "nvidia/cudnn/lib", "nvidia/cufft/lib", "nvidia/cusolver/lib",
            "nvidia/cusparse/lib",
        ]
        for pkg_path in cu12_pkg_paths:
            full_path = os.path.join(sp, pkg_path)
            if os.path.isdir(full_path) and full_path not in check_paths:
                check_paths.append(full_path)
        # Add LD_LIBRARY_PATH entries
        for p in ld_path.split(':'):
            if p and os.path.isdir(p) and p not in check_paths:
                check_paths.append(p)

        missing_libs = []
        found_wrong_version = []

        for lib, pkg_name, alt_names in key_libs:
            found = False
            found_path = None
            wrong_version = None

            # Check all names (including alternates for case variations)
            names_to_check = [lib] + alt_names

            for check_path in check_paths:
                for name in names_to_check:
                    lib_path = os.path.join(check_path, name)
                    if os.path.exists(lib_path):
                        # Get a nice short path name for display
                        rel_path = check_path.replace(sp + "/", "")
                        print(f"  ✅ {lib} found in {rel_path}")
                        found = True
                        found_path = check_path
                        break
                if found:
                    break

            if not found:
                # Check for wrong version (.so.13 instead of .so.12, etc.)
                base_name = lib.rsplit('.so.', 1)[0]
                for check_path in check_paths:
                    wrong_files = glob.glob(os.path.join(check_path, f"{base_name}.so.*"))
                    # Also check case variations
                    if 'nvjitlink' in base_name.lower():
                        wrong_files += glob.glob(os.path.join(check_path, "libnvJitLink.so.*"))
                    for wf in wrong_files:
                        wf_base = os.path.basename(wf)
                        if wf_base not in names_to_check and '.so.' in wf_base:
                            wrong_version = wf_base
                            break
                    if wrong_version:
                        break

            if not found:
                if wrong_version:
                    print(f"  ❌ {lib} NOT FOUND (have {wrong_version} instead)")
                    found_wrong_version.append((lib, wrong_version, pkg_name))
                else:
                    print(f"  ❌ {lib} NOT FOUND")
                    missing_libs.append((lib, pkg_name))

        # Check JAX plugins
        print_subheader("JAX CUDA Plugins")
        jax_plugins_path = os.path.join(sp, "jax_plugins")
        if os.path.isdir(jax_plugins_path):
            plugins = os.listdir(jax_plugins_path)
            for plugin in sorted(plugins):
                plugin_path = os.path.join(jax_plugins_path, plugin)
                if os.path.isdir(plugin_path):
                    so_files = glob.glob(os.path.join(plugin_path, "*.so"))
                    print(f"  {plugin}: {len(so_files)} .so files")
                    # Check for xla_cuda_plugin.so
                    xla_plugin = os.path.join(plugin_path, "xla_cuda_plugin.so")
                    if os.path.exists(xla_plugin):
                        # Check if patchelf was applied by looking at execstack
                        try:
                            result = subprocess.run(
                                ["readelf", "-l", xla_plugin],
                                capture_output=True, text=True, timeout=5
                            )
                            if "GNU_STACK" in result.stdout:
                                for line in result.stdout.split('\n'):
                                    if "GNU_STACK" in line:
                                        if "RWE" in line:
                                            print(f"    ⚠️ xla_cuda_plugin.so has executable stack (needs patchelf)")
                                        else:
                                            print(f"    ✅ xla_cuda_plugin.so stack is non-executable")
                                        break
                        except Exception:
                            pass
        else:
            print("  No jax_plugins directory found")

        # Show ldd output for the JAX plugin
        print_subheader("JAX CUDA Plugin Dependencies (ldd)")
        try:
            xla_plugin = os.path.join(sp, "jax_plugins", "xla_cuda12", "xla_cuda_plugin.so")
            if os.path.exists(xla_plugin):
                # Build LD_LIBRARY_PATH with nvidia package paths
                nvidia_paths = []
                for pkg_dir in ['cublas', 'cuda_runtime', 'cusolver', 'cusparse', 'cufft', 'cudnn', 'nvjitlink']:
                    pkg_path = os.path.join(sp, "nvidia", pkg_dir, "lib")
                    if os.path.isdir(pkg_path):
                        nvidia_paths.append(pkg_path)
                test_ld_path = ':'.join(nvidia_paths)
                if ld_path:
                    test_ld_path = f"{test_ld_path}:{ld_path}"

                env = os.environ.copy()
                env['LD_LIBRARY_PATH'] = test_ld_path

                result = subprocess.run(
                    ["ldd", xla_plugin],
                    capture_output=True, text=True, timeout=10, env=env
                )
                # Show only CUDA-related or "not found" lines
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if 'cuda' in line.lower() or 'cublas' in line.lower() or \
                       'cusolver' in line.lower() or 'cudnn' in line.lower() or \
                       'cufft' in line.lower() or 'cusparse' in line.lower() or \
                       'nvjit' in line.lower() or 'not found' in line.lower():
                        print(f"  {line}")
            else:
                print(f"  xla_cuda_plugin.so not found at expected location")
        except Exception as e:
            print(f"  Error checking ldd: {e}")

        # Recommendations based on findings
        print_subheader("Recommendations")

        # Check if we have version mismatches (CUDA 13 libs instead of CUDA 12)
        if found_wrong_version:
            print("  ⚠️  CUDA LIBRARY VERSION MISMATCH DETECTED")
            print("")
            print("  Your nvidia packages have CUDA 13 libraries (.so.13), but")
            print("  JAX 0.4.23's CUDA 12 plugin needs CUDA 12 libraries (.so.12).")
            print("")
            print("  Found wrong versions:")
            for needed, have, pkg in found_wrong_version:
                print(f"    - Need {needed}, have {have}")
            print("")
            print("  RECOMMENDED FIX (automated):")
            print("    source scripts/fix_jax_cuda_plugin.sh")
            print("")
            print("  This script will:")
            print("    1. Install hybrid nvidia packages (mix of -cu12 and non-suffixed)")
            print("    2. Reinstall JAX 0.4.23 with CUDA 12 support")
            print("    3. Apply patchelf fix for modern Linux kernels")
            print("    4. Set LD_LIBRARY_PATH correctly")
            print("    5. Test the installation")
            print("")
            print("  See docs/GPU_ACCELERATION.md for manual setup details.")
        elif missing_libs:
            print("  ❌ Some CUDA libraries are missing")
            print("")
            print("  Missing libraries:")
            for lib, pkg in missing_libs:
                print(f"    - {lib} (from {pkg})")
            print("")
            print("  RECOMMENDED FIX (automated):")
            print("    source scripts/fix_jax_cuda_plugin.sh")
            print("")
            print("  Or install packages manually:")
            print("    pip install --extra-index-url https://pypi.nvidia.com \\")
            print("        nvidia-cusolver nvidia-cublas nvidia-cuda-runtime \\")
            print("        nvidia-cufft nvidia-cusparse nvidia-nvjitlink nvidia-cudnn-cu12")
        else:
            cu13_lib = os.path.join(sp, "nvidia", "cu13", "lib")
            if os.path.isdir(cu13_lib):
                print("  ✅ All required CUDA 12 libraries found!")
                print("")
                print(f"  Make sure LD_LIBRARY_PATH includes:")
                print(f"    export LD_LIBRARY_PATH={cu13_lib}:$LD_LIBRARY_PATH")
            else:
                print("  ℹ️  Using individual nvidia/xxx/lib paths")
                print("")
                print("  Your current LD_LIBRARY_PATH looks correct.")

    def generate_report(self):
        """Generate summary report with recommendations."""
        print_header("Summary")

        if self.results["issues"]:
            print_subheader("Issues Found")
            for issue in self.results["issues"]:
                print(f"  ⚠️  {issue}")

        print_subheader("System Summary")
        total_cpus = self.results["cpu"]["total_cores"]
        total_memory = self.results["memory"].get("total_gb", 0)
        gpu_available = self.results["gpu"].get("available", False)
        jax_gpu = self.results["gpu"].get("jax_gpu", False)
        cupy_working = self.results["gpu"].get("cupy_working", False)

        print(f"  CPU cores: {total_cpus}")
        print(f"  Total RAM: {total_memory:.1f} GB")
        print(f"  NVIDIA GPU: {'Available' if gpu_available else 'Not available'}")
        print(f"  JAX GPU (BaSiCPy): {'Enabled' if jax_gpu else 'CPU-only'}")
        print(f"  CuPy GPU (linumpy): {'Working' if cupy_working else 'Not available'}")

        return self.results


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Quick system check only (no benchmarks)")
    p.add_argument("--benchmark", action="store_true",
                   help="Include performance benchmarks")
    p.add_argument("--debug-cuda", action="store_true",
                   help="Show detailed CUDA library debugging info")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show verbose error output for debugging")
    p.add_argument("--output", "-o", type=str,
                   help="Save results to JSON file")
    args = p.parse_args()

    print("=" * get_terminal_width())
    print(" LINUMPY 3D RECONSTRUCTION PIPELINE DIAGNOSTICS")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * get_terminal_width())

    diag = SystemDiagnostics(verbose=args.verbose)

    if args.debug_cuda:
        diag.debug_cuda_libraries()
        return

    diag.check_cpu()
    diag.check_memory()
    diag.check_gpu()
    diag.check_python_packages()
    diag.check_nextflow_config()

    if args.benchmark or not args.quick:
        diag.run_benchmarks()

    results = diag.generate_report()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "=" * get_terminal_width())
    print(" DIAGNOSTICS COMPLETE")
    print("=" * get_terminal_width() + "\n")


if __name__ == "__main__":
    main()

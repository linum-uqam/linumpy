#!/usr/bin/env python3
"""
Diagnostic script for linumpy 3D reconstruction pipeline performance.

This script checks the server configuration to identify bottlenecks:
- CPU core detection and thread configuration
- GPU availability and CUDA setup (CuPy for linumpy GPU ops; PyTorch for BaSiCPy)
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
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any


def get_terminal_width() -> Any:
    """Get terminal width for formatting."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def print_header(title: str) -> None:
    """Print a section header."""
    width = get_terminal_width()
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


class SystemDiagnostics:
    """System diagnostics collector."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.results: dict[str, Any] = {
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

    def check_cpu(self) -> Any:
        """Check CPU configuration."""
        print_header("CPU Configuration")

        # Physical CPU info
        total_cpus = os.process_cpu_count() or os.cpu_count() or 1
        self.results["cpu"]["total_cores"] = total_cpus
        print(f"  Total CPU cores detected: {total_cpus}")

        # Check environment variables that control threading
        print_subheader("Thread Environment Variables")
        thread_vars = [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
            "LINUMPY_MAX_CPUS",
            "LINUMPY_RESERVED_CPUS",
        ]

        for var in thread_vars:
            value = os.environ.get(var, "(not set)")
            self.results["cpu"][var] = value
            print(f"  {var}: {value}")

        return total_cpus

    def check_memory(self) -> Any:
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
                print("  ⚠️  Low available memory - may cause swapping")
                self.results["issues"].append(f"Low available memory: {available_gb:.1f} GB")

            return total_gb, available_gb

        except ImportError:
            print("  ⚠️  psutil not installed - cannot check memory")
            print("     Install with: pip install psutil")
            self.results["memory"]["error"] = "psutil not installed"
            return None, None

    def check_gpu(self) -> None:
        """Check GPU configuration and CUDA availability."""
        print_header("GPU Configuration")

        # Check nvidia-smi
        print_subheader("NVIDIA Driver")
        try:
            simple_result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
            if simple_result.returncode == 0 and simple_result.stdout.strip():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.free,driver_version,cuda_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpus = []
                    for i, line in enumerate(result.stdout.strip().split("\n")):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
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
                                print(
                                    f"    Memory: {int(float(parts[1])) / 1024:.1f} GB total, "
                                    f"{int(float(parts[2])) / 1024:.1f} GB free"
                                )
                                print(f"    Driver: {parts[3]}, CUDA: {parts[4]}")
                            except ValueError, IndexError:
                                print(f"  ⚠️  Could not parse GPU {i} info: {line}")

                    self.results["gpu"]["available"] = len(gpus) > 0
                    self.results["gpu"]["devices"] = gpus
                    if gpus:
                        print(f"  ✅ Found {len(gpus)} GPU(s)")
                else:
                    gpus = []
                    for line in simple_result.stdout.strip().split("\n"):
                        if line.startswith("GPU "):
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

        # Check CuPy
        print_subheader("CuPy (GPU Python)")
        self._check_cupy()

        # Check linumpy GPU module
        print_subheader("Linumpy GPU Module")
        self._check_linumpy_gpu()

    def _check_cupy(self) -> None:
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
                        print(f"    Device {i}: {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")

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

    def _check_linumpy_gpu(self) -> None:
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

    def check_python_packages(self) -> None:
        """Check critical Python packages."""
        print_header("Python Environment")

        print(f"  Python version: {sys.version}")
        self.results["python"]["version"] = sys.version

        print_subheader("Critical Packages")

        packages = [
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("basicpy", "basicpy"),
            ("torch", "torch"),
            ("pqdm", "pqdm"),
            ("dask", "dask"),
            ("zarr", "zarr"),
            ("threadpoolctl", "threadpoolctl"),
            ("psutil", "psutil"),
        ]

        for name, import_name in packages:
            try:
                mod = __import__(import_name)
                version = getattr(mod, "__version__", "unknown")
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
                blas_info = np.show_config(mode="dicts")
                if blas_info and "Build Dependencies" in blas_info:
                    blas = blas_info.get("Build Dependencies", {}).get("blas", {})
                    print(f"  BLAS: {blas.get('name', 'unknown')}")
                    self.results["python"]["blas"] = blas.get("name", "unknown")
            except Exception:
                print("  (Could not determine BLAS configuration)")
        except Exception:
            print("  (Could not check BLAS)")

    def check_nextflow_config(self) -> None:
        """Check Nextflow configuration recommendations."""
        print_header("Nextflow Configuration")

        total_cpus = self.results["cpu"]["total_cores"]
        total_memory = self.results["memory"].get("total_gb", 0)

        print_subheader("Current Environment")

        nf_process_name = os.environ.get("NXF_TASK_NAME", "Not in Nextflow process")
        print(f"  NXF_TASK_NAME: {nf_process_name}")

        try:
            result = subprocess.run(["nextflow", "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.strip().split("\n")[0]
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
        print("  • fix_illumination step uses BaSiC algorithm")
        print("  • Each BaSiC process typically uses ~3 CPU threads")
        if total_memory:
            print(f"  • With {total_memory:.0f} GB RAM, memory should not be a bottleneck")

    def run_benchmarks(self) -> None:
        """Run performance benchmarks."""
        print_header("Performance Benchmarks")

        print_subheader("BaSiC Algorithm (fix_illumination bottleneck)")
        self._run_basic_benchmark()

        print_subheader("Parallel Processing (pqdm)")
        self._run_pqdm_benchmark()

        if self.results["gpu"].get("cupy_working"):
            print_subheader("GPU Performance")
            self._run_gpu_benchmark()

    def _run_basic_benchmark(self) -> None:
        """Run BaSiC benchmark (BaSiCPy 2.0+ uses PyTorch backend)."""
        benchmark_code = """
import sys
import time
import numpy as np

try:
    import torch
    has_cuda = torch.cuda.is_available()
    mode = "GPU" if has_cuda else "CPU"
    print(f"DEBUG_TORCH_MODE:{mode}")
    if has_cuda:
        print(f"DEBUG_TORCH_DEVICE:{torch.cuda.get_device_name(0)}")

    from basicpy import BaSiC
    tiles = np.random.rand(16, 256, 256).astype(np.float32) * 1000

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
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", benchmark_code],
                capture_output=True,
                text=True,
                timeout=120,
                env=dict(os.environ),
            )

            success_line = None
            for line in result.stdout.strip().split("\n"):
                if line.startswith("DEBUG_TORCH_MODE:"):
                    mode = line.split(":", 1)[1]
                    print(f"  PyTorch mode: {mode}")
                elif line.startswith("DEBUG_TORCH_DEVICE:"):
                    device = line.split(":", 1)[1]
                    print(f"  GPU device: {device}")
                elif line.startswith("SUCCESS:"):
                    success_line = line

            if success_line:
                parts = success_line.split(":")
                elapsed = float(parts[1])
                mode = parts[2]
                print(f"  BaSiC fit 16 tiles @ (256, 256): {elapsed:.2f}s ({mode})")
                self.results["linumpy"]["basic_16x256"] = elapsed
                if mode == "GPU":
                    print("  ✅ PyTorch GPU acceleration working")
                else:
                    print("  ⚠️  Running on CPU (no CUDA GPU detected)")
            else:
                error_out = result.stderr + result.stdout
                self._handle_basic_error(error_out)
                if self.verbose:
                    print("\n  --- Full subprocess output ---")
                    for line in result.stdout.split("\n")[-20:]:
                        print(f"    {line}")
                    for line in result.stderr.split("\n")[-30:]:
                        print(f"    {line}")

        except subprocess.TimeoutExpired:
            print("  ⚠️  BaSiC benchmark timed out (>120s)")
        except Exception as e:
            print(f"  ⚠️  BaSiC benchmark failed: {e}")

    def _handle_basic_error(self, error_out: str) -> None:
        """Handle BaSiC benchmark errors with specific guidance."""
        error_lower = error_out.lower()

        if "no module named 'basicpy'" in error_lower:
            print("  ❌ BaSiC failed: basicpy not installed")
            print("     Install: pip install basicpy")
        elif "no module named 'torch'" in error_lower:
            print("  ❌ BaSiC failed: PyTorch not installed")
            print("     Install: pip install torch")
        elif "cuda" in error_lower and "out of memory" in error_lower:
            print("  ❌ BaSiC failed: GPU out of memory")
            print("     BaSiC will automatically fall back to CPU if GPU is unavailable.")
        else:
            for line in error_out.split("\n"):
                if line.startswith("ERROR:"):
                    print(f"  ❌ BaSiC failed: {line[6:]}")
                    break
            else:
                print(f"  ❌ BaSiC failed: {error_out[:200]}")
            if self.verbose:
                print("\n     Complete output (--verbose mode):")
                for line in error_out.split("\n")[-30:]:
                    print(f"       {line}")

    def _run_pqdm_benchmark(self) -> Any:
        """Run pqdm parallel processing benchmark."""
        try:
            import numpy as np
            from pqdm.processes import pqdm

            def dummy_task(i: Any) -> Any:
                arr = np.random.rand(500, 500)
                for _ in range(10):
                    arr = np.fft.fft2(arr)
                return i

            for n_jobs in [1, 4, 8, 16]:
                start = time.perf_counter()
                _results = pqdm(range(16), dummy_task, n_jobs=n_jobs, desc=f"pqdm n_jobs={n_jobs}", disable=True)
                elapsed = time.perf_counter() - start
                print(f"  pqdm with n_jobs={n_jobs}: {elapsed:.2f}s for 16 tasks")
                self.results["linumpy"][f"pqdm_njobs{n_jobs}"] = round(elapsed, 2)

        except Exception as e:
            print(f"  ⚠️  pqdm benchmark failed: {e}")

    def _run_gpu_benchmark(self) -> None:
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

            print(f"  FFT {size}x{size}: CPU {cpu_time * 1000:.1f}ms, GPU {gpu_time * 1000:.1f}ms ({speedup:.1f}x speedup)")

            self.results["linumpy"]["fft_cpu_ms"] = round(cpu_time * 1000, 1)
            self.results["linumpy"]["fft_gpu_ms"] = round(gpu_time * 1000, 1)
            self.results["linumpy"]["fft_speedup"] = round(speedup, 1)

        except Exception as e:
            print(f"  ⚠️  GPU benchmark failed: {e}")

    def debug_cuda_libraries(self) -> None:
        """Show detailed CUDA library debugging information."""
        import site

        print_header("CUDA Library Debug Information")

        # Get site-packages
        sp = site.getsitepackages()[0]
        print(f"  Site-packages: {sp}")

        # Check current LD_LIBRARY_PATH
        print_subheader("Current LD_LIBRARY_PATH")
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_path:
            for i, p in enumerate(ld_path.split(":")):
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
                        so12_count = len([f for f in so_files if ".so.12" in f])
                        so11_count = len([f for f in so_files if ".so.11" in f])
                        so13_count = len([f for f in so_files if ".so.13" in f])
                        version_info = []
                        if so11_count:
                            version_info.append(f"{so11_count} .so.11")
                        if so12_count:
                            version_info.append(f"{so12_count} .so.12")
                        if so13_count:
                            version_info.append(f"{so13_count} .so.13")
                        print(
                            f"  nvidia/{subdir}/lib: {len(so_files)} .so files "
                            f"({', '.join(version_info) if version_info else 'no versioned'})"
                        )
        else:
            print("  No nvidia packages found in site-packages")

        # Check what's in the individual nvidia/xxx/lib paths
        print_subheader("Libraries in -cu12 Package Directories")
        cu12_dirs = ["cublas", "cuda_runtime", "nvjitlink", "cudnn", "cufft"]
        for dir_name in cu12_dirs:
            lib_path = os.path.join(sp, "nvidia", dir_name, "lib")
            if os.path.isdir(lib_path):
                so_files = glob.glob(os.path.join(lib_path, "lib*.so*"))
                versioned_files = [os.path.basename(f) for f in so_files if ".so." in os.path.basename(f)]
                if versioned_files:
                    print(f"  nvidia/{dir_name}/lib: {', '.join(sorted(versioned_files)[:4])}")

        # PyTorch CUDA availability (BaSiCPy / fix_illumination uses the PyTorch backend)
        print_subheader("PyTorch CUDA")
        try:
            import torch  # type: ignore[import-not-found]

            print(f"  torch: {torch.__version__}")
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"    [{i}] {torch.cuda.get_device_name(i)}")
            else:
                print("  ℹ️  Install a CUDA build of PyTorch to enable BaSiCPy GPU acceleration.")
                print("     See docs/GPU_ACCELERATION.md (BaSiCPy section).")
        except ImportError:
            print("  torch not installed (BaSiCPy will run on CPU)")

    def generate_report(self) -> Any:
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
        cupy_working = self.results["gpu"].get("cupy_working", False)

        print(f"  CPU cores: {total_cpus}")
        print(f"  Total RAM: {total_memory:.1f} GB")
        print(f"  NVIDIA GPU: {'Available' if gpu_available else 'Not available'}")
        print(f"  CuPy GPU (linumpy): {'Working' if cupy_working else 'Not available'}")

        return self.results


def main() -> None:
    """Run function."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Quick system check only (no benchmarks)")
    p.add_argument("--benchmark", action="store_true", help="Include performance benchmarks")
    p.add_argument("--debug-cuda", action="store_true", help="Show detailed CUDA library debugging info")
    p.add_argument("--verbose", "-v", action="store_true", help="Show verbose error output for debugging")
    p.add_argument("--output", "-o", type=str, help="Save results to JSON file")
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
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "=" * get_terminal_width())
    print(" DIAGNOSTICS COMPLETE")
    print("=" * get_terminal_width() + "\n")


if __name__ == "__main__":
    main()
